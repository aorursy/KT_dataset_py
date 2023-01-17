import operator

import gc

import pathlib

import shutil

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import backend as K

from scipy import spatial

import cv2

!pip install ../input/landmark-lib/Keras_Applications-1.0.8-py3-none-any.whl

!pip install ../input/landmark-lib/efficientnet-1.1.0-py3-none-any.whl

import efficientnet.tfkeras as efn

import math



NUMBER_OF_CLASSES = 81313

LR = 0.0001



IMAGE_SIZE = [600, 600]

IMG_H = IMAGE_SIZE[0]

IMG_W = IMAGE_SIZE[1]

S = 64

M = 0.15

EFF = 7

WEIGHTS_PATH = "../input/b7res600drop5ep14/b7res600drop5.h5"

NUM_TO_RERANK = 3

DROPOUT = 0.5



class ArcMarginProduct(tf.keras.layers.Layer):

    '''

    Implements large margin arc distance.

    '''

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





# Function to build our model using fine tunning (efficientnet)

def get_model(eff = 1):

    



    margin = ArcMarginProduct(

        n_classes = NUMBER_OF_CLASSES, 

        s = S, 

        m = M, 

        name='head/arc_margin', 

        dtype='float32'

        )



    inp = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3), name = 'inp1')

    label = tf.keras.layers.Input(shape = (), name = 'inp2')

    if eff == 0:

        x = efn.EfficientNetB0(weights = None, include_top = False)(inp)

    elif eff == 1:

        x = efn.EfficientNetB1(weights = None, include_top = False)(inp)

    elif eff == 2:

        x = efn.EfficientNetB2(weights = None, include_top = False)(inp)

    elif eff == 3:

        x = efn.EfficientNetB3(weights = None, include_top = False)(inp)

    elif eff == 4:

        x = efn.EfficientNetB4(weights = None, include_top = False)(inp)

    elif eff == 5:

        x = efn.EfficientNetB5(weights = None, include_top = False)(inp)

    elif eff == 6:

        x = efn.EfficientNetB6(weights = None, include_top = False)(inp)

    elif eff == 7:

        x = efn.EfficientNetB7(weights = None, include_top = False)(inp)

        

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(DROPOUT)(x)

    x = tf.keras.layers.Dense(512)(x)

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





NUM_EMBEDDING_DIMENSIONS = 512

DATASET_DIR = '../input/landmark-image-train/train_encoded.csv'

TEST_IMAGE_DIR = '../input/landmark-recognition-2020/test'

TRAIN_IMAGE_DIR = '../input/landmark-recognition-2020/train'

MODEL = get_model(eff = EFF)

MODEL.load_weights(WEIGHTS_PATH)

MODEL = tf.keras.models.Model(inputs = MODEL.input[0], outputs = MODEL.layers[-4].output)







NUM_PUBLIC_TEST_IMAGES = 10345 # Used to detect if in session or re-run.



# Read image and resize it

def read_image(image_path, size = (384, 384)):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, size)

    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))[1].tostring()

    img = tf.image.decode_jpeg(img, channels = 3)

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.reshape(img, [1, IMG_H, IMG_W, 3])

    return img



# Function to get training and test embeddings

def generate_embeddings(filepaths):

    image_paths = [x for x in pathlib.Path(filepaths).rglob('*.jpg')]

    num_images = len(image_paths)

    ids = num_images * [None]

    # Generate an empty matrix where we can store the embeddings of each image

    embeddings = np.empty((num_images, NUM_EMBEDDING_DIMENSIONS))

    for i, image_path in enumerate(image_paths):

        ids[i] = image_path.name.split('.')[0]

        image_tensor = read_image(str(image_path), (IMG_H, IMG_W))

        prediction = MODEL.predict(image_tensor)

#         prediction2 = MODEL2.predict(image_tensor)

#         prediction = np.average([prediction1, prediction2], axis = 0)

        embeddings[i, :] = prediction

    return ids, embeddings



# This function get the most similar train images for each test image based on cosine similarity

def get_similarities(train_csv, test_directory, train_directory):

    # Get target dictionary

    df = pd.read_csv(train_csv)

    df = df[['id', 'landmark_id']]

    df.set_index('id', inplace = True)

    df = df.to_dict()['landmark_id']

    # Extract the test ids and global feature for the test images

    test_ids, test_embeddings = generate_embeddings(test_directory)

    # Extract the train ids and global features for the train images

    train_ids, train_embeddings = generate_embeddings(train_directory)

    # Initiate a list were we will store the similar training images for each test image (also score)

    train_ids_labels_and_scores = [None] * test_embeddings.shape[0]

    # Using (slow) for-loop, as distance matrix doesn't fit in memory

    for test_index in range(test_embeddings.shape[0]):

        distances = spatial.distance.cdist(

            test_embeddings[np.newaxis, test_index, : ], train_embeddings, 'cosine')[0]

        # Get the indices of the closest images

        top_k = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]

        # Get the nearest ids and distances using the previous indices

        nearest = sorted([(train_ids[p], distances[p]) for p in top_k], key = lambda x: x[1])

        # Get the labels and score results

        train_ids_labels_and_scores[test_index] = [(df[train_id], 1.0 - cosine_distance) for \

                                                   train_id, cosine_distance in nearest]

        

    del test_embeddings

    del train_embeddings

    gc.collect()

    return test_ids, train_ids_labels_and_scores



# This function aggregate top simlarities and make predictions

def generate_predictions(test_ids, train_ids_labels_and_scores):

    targets = []

    scores = []

    

    # Iterate through each test id

    for test_index, test_id in enumerate(test_ids):

        aggregate_scores = {}

        # Iterate through the similar images with their corresponing score for the given test image

        for target, score in train_ids_labels_and_scores[test_index]:

            if target not in aggregate_scores:

                aggregate_scores[target] = 0

            aggregate_scores[target] += score

        # Get the best score

        target, score = max(aggregate_scores.items(), key = operator.itemgetter(1))

        targets.append(target)

        scores.append(score)

        

    final = pd.DataFrame({'id': test_ids, 'target': targets, 'scores': scores})

    final['landmarks'] = final['target'].astype(str) + ' ' + final['scores'].astype(str)

    final[['id', 'landmarks']].to_csv('submission.csv', index = False)

    return final



def inference_and_save_submission_csv(train_csv, test_directory, train_directory):

    image_paths = [x for x in pathlib.Path(test_directory).rglob('*.jpg')]

    test_len = len(image_paths)

    if test_len == NUM_PUBLIC_TEST_IMAGES:

        # Dummy submission

        shutil.copyfile('../input/landmark-recognition-2020/sample_submission.csv', 'submission.csv')

        return 'Job Done'

    else:

        test_ids, train_ids_labels_and_scores = get_similarities(train_csv, test_directory, train_directory)

        final = generate_predictions(test_ids, train_ids_labels_and_scores)

        return final

    

final = inference_and_save_submission_csv(DATASET_DIR, TEST_IMAGE_DIR, TRAIN_IMAGE_DIR)