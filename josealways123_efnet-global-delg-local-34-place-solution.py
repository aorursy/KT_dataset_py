DEBUG = False
#if true check parameters
import os
import re, csv, copy, gc, operator
import numpy as np
import pandas as pd
import random
import math
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import backend as K
from tqdm.notebook import tqdm as tqdm
import shutil
import pathlib
import pydegensac
import cv2
import PIL
from scipy import spatial
!pip install -q ../input/whl-files/Keras_Applications-1.0.8-py3-none-any.whl
!pip install -q ../input/whl-files/efficientnet-1.1.0-py3-none-any.whl
import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
IMAGE_SIZE = [512, 512]
NUMBER_OF_CLASSES = 81313
LR = 1e-6

DIM = IMAGE_SIZE[0]
EFNET = 5
QUALITY = 100
MARGIN = 0.3
EMBEDDING_DIMENSION = 2048   #1280 b0 2304 for b6 1536 b3 2048 b5
NUM_TO_RERANK = 3
TOP_K = 3

if DEBUG:
    NUM_PUBLIC_TRAIN_IMAGES = -1
else:
    NUM_PUBLIC_TRAIN_IMAGES = 1580470

DATASET_DIR = '../input/landmark-recognition-2020'
TRAIN_IMAGE_DIR = '../input/landmark-recognition-2020/train'
TEST_IMAGE_DIR = '../input/landmark-recognition-2020/test'
TRAIN_LABELMAP_PATH = '../input/landmark-recognition-2020/train.csv'
MODEL_PATH = '../input/tpu-train-final-dataset/fold-0_epoch-3_valloss-1.3284_loss-0.0056_margin-0.10_scale-64_logweight.h5'
class ArcMarginProduct(tf.keras.layers.Layer):
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]
def get_model(ef):
    margin = ArcMarginProduct(
        n_classes = NUMBER_OF_CLASSES, 
        s = 64, 
        m = 0.1, 
        name='head/arc_margin', 
        dtype='float32'
        )
    inp = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3), name = 'inp1')
    label = tf.keras.layers.Input(shape = (), name = 'inp2')
    x = EFNS[ef](weights = None, include_top = False)(inp)   
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = margin([x, label])

    output = tf.keras.layers.Softmax(dtype='float32')(x)

    model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])

    opt = tf.keras.optimizers.Adam(learning_rate = LR)

    model.compile(
        optimizer = opt,
        loss = [tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        ) 

    return model
MODEL = get_model(ef = EFNET)
MODEL.load_weights(MODEL_PATH)
embedding_model = tf.keras.models.Model(inputs = MODEL.input[0], outputs = MODEL.layers[-4].output)
print(embedding_model.layers[2].output.shape)
MAX_INLIER_SCORE = 35
MAX_REPROJECTION_ERROR = 7.0
MAX_RANSAC_ITERATIONS = 8500000
HOMOGRAPHY_CONFIDENCE = 0.99

SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'
DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)
DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])
DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)
DELG_INPUT_TENSOR_NAMES = ['input_image:0', 'input_scales:0', 'input_abs_thres:0']

LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)
LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],
                                               ['boxes:0', 'features:0'])

def rescore_and_rerank_by_num_inliers(test_image_id,train_ids_labels_and_scores):
    test_image_path = get_image_path('test', test_image_id)
    test_keypoints, test_descriptors = extract_local_features(test_image_path)
    for i in range(len(train_ids_labels_and_scores)):
        train_image_id, label, global_score = train_ids_labels_and_scores[i]
        train_image_path = get_image_path('train', train_image_id)
        train_keypoints, train_descriptors = extract_local_features(
            train_image_path)
        num_inliers = get_num_inliers(test_keypoints, test_descriptors,
                                      train_keypoints, train_descriptors)
        total_score = get_total_score(num_inliers, global_score)
        train_ids_labels_and_scores[i] = (train_image_id, label, total_score)
    train_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)
    return train_ids_labels_and_scores

def get_image_path(subset, image_id):
    name = to_hex(image_id)
    return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2],
                        '{}.jpg'.format(name))

def extract_local_features(image_path):
    image_tensor = load_image_tensor(image_path)
    features = LOCAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR,
                                           DELG_SCORE_THRESHOLD_TENSOR,
                                           LOCAL_FEATURE_NUM_TENSOR)
    keypoints = tf.divide(
        tf.add(
            tf.gather(features[0], [0, 1], axis=1),
            tf.gather(features[0], [2, 3], axis=1)), 2.0).numpy()
    descriptors = tf.nn.l2_normalize(
        features[1], axis=1, name='l2_normalization').numpy()
    return keypoints, descriptors

def get_num_inliers(test_keypoints, test_descriptors, train_keypoints,train_descriptors):
    test_match_kp, train_match_kp = get_putative_matching_keypoints(test_keypoints, test_descriptors, 
                                                                    train_keypoints, train_descriptors)
    if test_match_kp.shape[0] <= 4:
        return 0
    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                            MAX_REPROJECTION_ERROR,
                                            HOMOGRAPHY_CONFIDENCE,
                                            MAX_RANSAC_ITERATIONS)
    except np.linalg.LinAlgError:
        return 0
    return int(copy.deepcopy(mask).astype(np.float32).sum())

def get_putative_matching_keypoints(test_keypoints,
                                    test_descriptors,
                                    train_keypoints,
                                    train_descriptors,
                                    max_distance=0.9):
    train_descriptor_tree = spatial.cKDTree(train_descriptors)
    _, matches = train_descriptor_tree.query(test_descriptors, distance_upper_bound=max_distance)
    test_kp_count = test_keypoints.shape[0]
    train_kp_count = train_keypoints.shape[0]
    test_matching_keypoints = np.array([
        test_keypoints[i,] for i in range(test_kp_count) if matches[i] != train_kp_count])
    train_matching_keypoints = np.array([
        train_keypoints[matches[i],] for i in range(test_kp_count) if matches[i] != train_kp_count])
    return test_matching_keypoints, train_matching_keypoints

def get_total_score(num_inliers, global_score):
    local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE
    return local_score + global_score

def load_image_tensor(image_path):
    return tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB')))
def read_image(image_path, size = (DIM, DIM)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, QUALITY))[1].tostring()
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [1, DIM, DIM, 3])
    return img

def get_predictions(labelmap):
    if DEBUG:
        test_image_paths = [x for x in pathlib.Path(TEST_IMAGE_DIR).rglob('*.jpg')][:100]
        train_image_paths = [x for x in pathlib.Path(TRAIN_IMAGE_DIR).rglob('*.jpg')][:100]        
    else:
        test_image_paths = [x for x in pathlib.Path(TEST_IMAGE_DIR).rglob('*.jpg')]
        train_image_paths = [x for x in pathlib.Path(TRAIN_IMAGE_DIR).rglob('*.jpg')]
    print('Extracting global features of test images')
    test_ids, test_embeddings = extract_global_features(test_image_paths)
    print('Extracting global features of train images')
    train_ids, train_embeddings = extract_global_features(train_image_paths)
    train_ids_labels_and_scores = [None] * test_embeddings.shape[0]
    for test_index in range(test_embeddings.shape[0]):
        distances = spatial.distance.cdist(test_embeddings[np.newaxis, test_index, : ],
                                         train_embeddings, 'cosine')[0]
        partition = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]
        nearest = sorted([(train_ids[p], distances[p]) for p in partition],
                         key = lambda x: x[1])
        train_ids_labels_and_scores[test_index] = [(train_id, labelmap[to_hex(train_id)],
                                                    1.0 - cosine_distance)
                                                   for train_id, cosine_distance in nearest]
    del test_embeddings
    del train_embeddings
    del labelmap
    gc.collect()
    pre_verification_predictions = get_prediction_map(test_ids, train_ids_labels_and_scores)
    
    for test_index, test_id in tqdm(enumerate(test_ids)):
        train_ids_labels_and_scores[test_index]=rescore_and_rerank_by_num_inliers(
            test_id,train_ids_labels_and_scores[test_index])
    post_verification_predictions = get_prediction_map(test_ids, train_ids_labels_and_scores)
    
    return pre_verification_predictions, post_verification_predictions

def extract_global_features(image_paths):
    num_images = len(image_paths)
    ids = num_images * [None]
    embeddings = np.empty((num_images, EMBEDDING_DIMENSION))
    for i, image_path in tqdm(enumerate(image_paths)):           
        ids[i] = int(image_path.name.split('.')[0], 16)
        image_tensor = read_image(str(image_path), size = (DIM, DIM))
        features = embedding_model.predict(image_tensor)
        embeddings[i, :] = tf.nn.l2_normalize(tf.reduce_sum(features, axis=0),
                                              axis=0).numpy()
    return ids, embeddings

def get_prediction_map(test_ids, train_ids_labels_and_scores):
    prediction_map = dict()
    for test_index, test_id in enumerate(test_ids):
        hex_test_id = to_hex(test_id)
        aggregate_scores = {}
        for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:
            if label not in aggregate_scores:
                aggregate_scores[label] = 0
            aggregate_scores[label] += score
        label, score = max(aggregate_scores.items(), key = operator.itemgetter(1))
        prediction_map[hex_test_id] = {'score': score, 'class': label}
    return prediction_map

def to_hex(image_id):
    return '{0:0{1}x}'.format(image_id, 16)

def load_labelmap():
    with open(TRAIN_LABELMAP_PATH, mode = 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        labelmap = {row['id']: row['landmark_id'] for row in csv_reader}
    return labelmap

def save_submission_csv(predictions = None):
    if predictions is None:
        shutil.copyfile(os.path.join(DATASET_DIR, 'sample_submission.csv'), 
                        'submission.csv')
        return
    with open('submission.csv', 'w') as submission_csv:
        csv_writer = csv.DictWriter(submission_csv, fieldnames = ['id', 'landmarks'])
        csv_writer.writeheader()
        for image_id, prediction in predictions.items():
            label = prediction['class']
            score = prediction['score']
            csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})
def main():
    labelmap = load_labelmap()
    num_training_images = len(labelmap.keys())
    print(f'Found {num_training_images} training images')
    if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:
        print(f'Found {NUM_PUBLIC_TRAIN_IMAGES} training images. Copying sample submission')
        save_submission_csv()
        return

    pre_verification_predictions, post_verification_predictions = get_predictions(labelmap)
    save_submission_csv(post_verification_predictions)
    print('Done')

main()
