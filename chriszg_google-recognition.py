import time

import copy

import csv

import operator

import os

import pathlib

import shutil  # for file operation



import numpy as np

import PIL

import pydegensac  # RANSAC in Python

from scipy import spatial

import tensorflow as tf

import warnings





# warnings.simplefilter("ignore")
# Dataset parameters:

INPUT_DIR = '../input/'

DATASET_DIR = INPUT_DIR + 'landmark-recognition-2020/'

TEST_DIR = DATASET_DIR + 'test/'

TRAIN_DIR = DATASET_DIR + 'train/'

TRAIN_LABELMAP_PATH = DATASET_DIR + 'train.csv'



# DEBUGGING PARAMS

DEBUG = False  # only use small portion (DEBUG_SIZE) of data to get result quickly

DEBUG_SIZE = 500

PUBLIC_TRAIN_SIZE = 1580470  # Used to detect if in session or re-run

MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging



# Retrieval & re-ranking parameters

NUM_TO_RERANK = 5

TOP_K = 3  # Number of retrieved images used to make prediction for a test image



# RANSAC parameters

MAX_INLIER_SCORE = 30

MAX_REPROJECTION_ERROR = 6.0

MAX_RANSAC_ITERATIONS = 5_000  # 5_000_000

HOMOGRAPHY_CONFIDENCE = 0.96  # 0.99



# DELG model

DELG_MODEL_DIR = '../input/delg-saved-models/local_and_global'

DELG_IMAGE_SCALES = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])

DELG_SCORE_THRESHOLD = tf.constant(175.)

DELG_INPUT_TENSOR_NAMES = [

    'input_image:0', 'input_scales:0', 'input_abs_thres:0'

]



# Global feature extraction

NUM_EMBEDDING_DIMENSIONS = 2048



# Local feature extraction

LOCAL_FEATURE_NUM = tf.constant(1000)



# log frequency

LOG_FREQ = 500



# quick submit

QUICK_SUBMIT = True
def image_path(subset, image_id):

    """Get image path by image id."""

    return os.path.join(DATASET_DIR, subset, image_id[0], image_id[1], image_id[2],

                        '{}.jpg'.format(image_id))





def load_image(image_path):

    """Convert image to tensor."""

    return tf.convert_to_tensor(

            np.array(PIL.Image.open(image_path).convert('RGB')))





# load label map

labelmap = None

with open(TRAIN_LABELMAP_PATH, mode='r') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    labelmap = {row['id']: row['landmark_id'] for row in csv_reader}
delg_model = tf.saved_model.load(DELG_MODEL_DIR)



# Global feature extraction function

global_feature_fn = delg_model.prune(DELG_INPUT_TENSOR_NAMES,

    ['global_descriptors:0'])



# Local feature extraction function

local_feature_fn = delg_model.prune(

    DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],

    ['boxes:0', 'features:0'])
def global_features(image_root_dir):

    """Extracts embeddings for all the images in given `image_root_dir`."""

    image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg')]

    if DEBUG:

        image_paths = image_paths[:DEBUG_SIZE]



    num_embeddings = len(image_paths)

    if MAX_NUM_EMBEDDINGS > 0:

        num_embeddings = min(MAX_NUM_EMBEDDINGS, num_embeddings)



    ids = num_embeddings * [None]

    embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))



    start = time.time()

    total = 0  # total time used

    for i, image_path in enumerate(image_paths):

        if i >= num_embeddings:

            break



        ids[i] = image_path.name.split('.')[0]

        image_tensor = load_image(image_path)

        features = global_feature_fn(image_tensor,

            DELG_IMAGE_SCALES, DELG_SCORE_THRESHOLD)



        embeddings[i, :] = tf.nn.l2_normalize(

                tf.reduce_sum(features[0], axis=0, name='sum_pooling'),

                axis=0, name='final_l2_normalization').numpy()



        # logging

        x = i + 1

        if x % LOG_FREQ == 0 or x == num_embeddings:

            end = time.time()

            used = end - start

            total += used

            avg = total / x  # average time used per step

            remain = avg * (num_embeddings-x) / 60.0

            start = end

            print('[{}/{}], [{:.2f}s/{:.2f}mins] used, {:.2f}mins remain'.format(x,

                num_embeddings, used, total/60, remain))



    print('Global features extracted, {:.2f}mins used'.format(total/60))

    return ids, embeddings





def local_features(image_path):

    """Extracts local features for the given `image_path`."""

    image_tensor = load_image(image_path)



    features = local_feature_fn(image_tensor, DELG_IMAGE_SCALES,

                                DELG_SCORE_THRESHOLD, LOCAL_FEATURE_NUM)



    # Shape: (N, 2)

    keypoints = tf.divide(

            tf.add(tf.gather(features[0], [0, 1], axis=1),

                   tf.gather(features[0], [2, 3], axis=1)), 2.0).numpy()



    # Shape: (N, 128)

    descriptors = tf.nn.l2_normalize(

        features[1], axis=1, name='l2_normalization').numpy()



    return keypoints, descriptors
def match_keypoints(test_keypoints, test_descriptors, train_keypoints,

                    train_descriptors, max_distance=0.9):

    """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""

    train_descriptor_tree = spatial.cKDTree(train_descriptors)

    _, matches = train_descriptor_tree.query(

            test_descriptors, distance_upper_bound=max_distance)



    test_kp_count = test_keypoints.shape[0]

    train_kp_count = train_keypoints.shape[0]



    test_matching_keypoints = np.array([

            test_keypoints[i,]

            for i in range(test_kp_count)

            if matches[i] != train_kp_count

    ])



    train_matching_keypoints = np.array([

            train_keypoints[matches[i],]

            for i in range(test_kp_count)

            if matches[i] != train_kp_count

    ])



    return test_matching_keypoints, train_matching_keypoints





def inliers_num(test_keypoints, test_descriptors,

                train_keypoints, train_descriptors):

    """Returns the number of RANSAC inliers."""



    test_match_kp, train_match_kp = match_keypoints(

        test_keypoints, test_descriptors,

        train_keypoints, train_descriptors)



    if test_match_kp.shape[0] <= 4:

        # Min keypoints supported by `pydegensac.findHomography()`

        # RANSAC needs at least 5 keypoints

        return 0



    try:

        # 计算多个点对之间的最优单映射变换矩阵H

        H, mask = pydegensac.findHomography(test_match_kp, train_match_kp,

                                            MAX_REPROJECTION_ERROR,

                                            HOMOGRAPHY_CONFIDENCE,

                                            MAX_RANSAC_ITERATIONS)

    except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.

        return 0



    return int(copy.deepcopy(mask).astype(np.float32).sum())





def score(num_inliers, global_score):

    """Compute total score."""

    local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE

    return local_score + global_score





def rescore(test_image_id, labels_scores):

    """Returns rescored and sorted training images by local feature extraction."""



    test_image_path = image_path('test', test_image_id)

    test_keypoints, test_descriptors = local_features(test_image_path)



    for i in range(len(labels_scores)):

        train_image_id, label, global_score = labels_scores[i]



        train_image_path = image_path('train', train_image_id)

        train_keypoints, train_descriptors = local_features(

            train_image_path)



        num_inliers = inliers_num(test_keypoints, test_descriptors,

                                  train_keypoints, train_descriptors)

        s = score(num_inliers, global_score)

        labels_scores[i] = (train_image_id, label, s)



    labels_scores.sort(key=lambda x: x[2], reverse=True)



    return labels_scores
def predict_map(test_ids, labels_scores):

    """Makes dict from test ids and ranked training ids, labels, scores."""

    prediction_map = dict()



    for test_index, test_id in enumerate(test_ids):

        aggregate_scores = {}

        for _, label, score in labels_scores[test_index][:TOP_K]:

            if label not in aggregate_scores:

                aggregate_scores[label] = 0

            aggregate_scores[label] += score



        label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))



        prediction_map[test_id] = {'score': score, 'class': label}



    return prediction_map





def predict(test_ids, test_embeddings, train_ids, train_embeddings):

    """Gets predictions using embedding similarity and local feature reranking."""

    labels_scores = [None] * test_embeddings.shape[0]

    

    # using global features

    for test_index in range(test_embeddings.shape[0]):

        distances = spatial.distance.cdist(

                test_embeddings[np.newaxis, test_index, :], train_embeddings,

                'cosine')[0]

        

        # get NUM_TO_RERANK entries by distance to re-rank

        partition = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]



        # sort tuple (train_id, distance) by distance

        nearest = sorted([(train_ids[p], distances[p]) for p in partition],

                         key=lambda x: x[1])



        labels_scores[test_index] = [

                (train_id, labelmap[train_id], 1. - cosine_distance)

                for train_id, cosine_distance in nearest

        ]

        

    # using local features to rescore

    pre_verify_preds = predict_map(test_ids, labels_scores)



    for test_index, test_id in enumerate(test_ids):

        labels_scores[test_index] = rescore(

            test_id, labels_scores[test_index])



    post_verify_preds = predict_map(

            test_ids, labels_scores)



    return post_verify_preds





def save(predictions):

    """Save prediction result to submission.csv."""

    with open('submission.csv', 'w') as f:

        csv_writer = csv.DictWriter(f, fieldnames=['id', 'landmarks'])

        csv_writer.writeheader()

        for image_id, prediction in predictions.items():

            label = prediction['class']

            score = prediction['score']

            csv_writer.writerow({'id': image_id, 'landmarks': '{} {:.8f}'.format(label, score)})
def run():

    start = time.time()

    training_size = len(labelmap.keys())

    print('Found {} training images.'.format(training_size))



    if QUICK_SUBMIT and training_size == PUBLIC_TRAIN_SIZE:

        # dummy submission

        print('Copying sample submission...')

        shutil.copyfile(DATASET_DIR + 'sample_submission.csv', 'submission.csv')

        return

    

    print('Extracting global features on testing set...')

    test_ids, test_embeddings = global_features(TEST_DIR)

    

    print('\nExtracting global features on training set...')

    train_ids, train_embeddings = global_features(TRAIN_DIR)

    

    print('\nPredicting...')

    preds = predict(test_ids, test_embeddings,

                    train_ids, train_embeddings)

    

    print('Saving result to csv...')

    save(preds)

    

    end = time.time()

    print('All done!({:.2f}mins used)'.format((end-start)/60))
run()