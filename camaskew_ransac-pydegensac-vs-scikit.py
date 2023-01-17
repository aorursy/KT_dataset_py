!pip install pydegensac



import cv2

import matplotlib.pyplot as plt

import numpy as np

import PIL

from scipy import spatial

import tensorflow as tf

from copy import deepcopy

import pydegensac

from skimage.measure import ransac as scikit_ransac

from skimage.transform import AffineTransform as scikit_AffineTransform

from time import time
# Both are images with label=9

# IMAGE_PATHS = [

#     '../input/recognition/recognition/images/train/2/8/2/28267d88d4d9ea30.jpg',

#     '../input/recognition/recognition/images/train/e/a/2/ea2537ff6259b15b.jpg'

# ]

#  IMAGE_PATHS = [

#     '../input/recognition/recognition/images/train/e/a/2/ea2537ff6259b15b.jpg',

#     '../input/recognition/recognition/images/train/e/a/2/ea2537ff6259b15b.jpg'

# ]

IMAGE_PATHS = [

    '../input/recognition/recognition/images/train/4/3/0/4305abb116733aa7.jpg',

    '../input/recognition/recognition/images/train/b/a/b/bab8b418edb91523.jpg',

    '../input/recognition/recognition/images/train/7/9/9/7992a31d0d0df612.jpg'

]



LOCAL_FEATURE_EXTRACTOR_FN = tf.saved_model.load('../input/delg-models/saved_models/delg_gld_20200520').prune(

    ['input_image:0', 'input_scales:0', 'input_abs_thres:0', 'input_max_feature_num:0'],

    ['boxes:0', 'features:0'])





def extract_local_features(image_path):

    image_tensor = tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert("RGB")))

    image_scales_tensor = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])

    score_threshold_tensor = tf.constant(175.)

    max_feature_num_tensor = tf.constant(1000)

    features = LOCAL_FEATURE_EXTRACTOR_FN(image_tensor, image_scales_tensor, score_threshold_tensor, max_feature_num_tensor)



    # Shape: (N, 2)

    keypoints = tf.divide(tf.add(tf.gather(features[0], [0, 1], axis=1), tf.gather(features[0], [2, 3], axis=1)),

                          2.0).numpy()



    # Shape: (N, 128)

    descriptors = tf.nn.l2_normalize(features[1], axis=1, name='l2_normalization').numpy()



    return keypoints, descriptors





def get_putative_matching_keypoints(test_keypoints, test_descriptors, train_keypoints, train_descriptors,

                                    max_distance=0.8):

    test_descriptor_tree = spatial.cKDTree(test_descriptors)

    _, matches = test_descriptor_tree.query(train_descriptors, distance_upper_bound=max_distance)



    test_kp_count = test_keypoints.shape[0]

    train_kp_count = train_keypoints.shape[0]



    test_matching_keypoints = np.array(

        [test_keypoints[matches[i],] for i in range(train_kp_count) if matches[i] != test_kp_count])

    train_matching_keypoints = np.array(

        [train_keypoints[i,] for i in range(train_kp_count) if matches[i] != test_kp_count])



    return test_matching_keypoints, train_matching_keypoints, matches



kp_0, desc_0 = extract_local_features(IMAGE_PATHS[0])

kp_1, desc_1 = extract_local_features(IMAGE_PATHS[1])



bf = cv2.BFMatcher()

matches = bf.knnMatch(desc_0, desc_1, k=2)

# Need to draw only good matches, so create a mask

matchesMask = [False for i in range(len(matches))]

# SNN ratio test

for i,(m,n) in enumerate(matches):

    if m.distance < 0.9*n.distance:

        matchesMask[i]=True

tentative_matches = [m[0] for i, m in enumerate(matches) if matchesMask[i] ]



src_pts = np.float32([ kp_0[m.queryIdx] for m in tentative_matches ]).reshape(-1,2)

dst_pts = np.float32([ kp_1[m.trainIdx] for m in tentative_matches ]).reshape(-1,2)
NUM_RUNS = 100

MAX_RANSAC_ITERATIONS = 1000

MAX_REPROJECTION_ERROR = 4.0

HOMOGRAPHY_CONFIDENCE = 0.995



scikit_inliers = 0

pydegensac_inliers = 0



# DEGENSAC

start_time = time() 

for i in range(NUM_RUNS): 

  _, inliers = pydegensac.findHomography(src_pts, dst_pts,

                                         MAX_REPROJECTION_ERROR,

                                         HOMOGRAPHY_CONFIDENCE,

                                         MAX_RANSAC_ITERATIONS)

  pydegensac_inliers += int(deepcopy(inliers).astype(np.float32).sum())

pydegensac_time = time() - start_time





# SCIKIT

start_time = time() 

for i in range(NUM_RUNS):

  _, inliers = scikit_ransac((src_pts, dst_pts),

                             scikit_AffineTransform,

                             min_samples=3,

                             stop_probability=HOMOGRAPHY_CONFIDENCE,

                             residual_threshold=MAX_REPROJECTION_ERROR,

                             max_trials=MAX_RANSAC_ITERATIONS)

  scikit_inliers += sum(inliers)

scikit_time = time() - start_time
fastest = 'pydegensac' if pydegensac_time < scikit_time else 'scikit'



print(f'Over the course of {NUM_RUNS} runs:')

print(f'pydegensac found {pydegensac_inliers} inliers in {pydegensac_time}s.')

print(f'scikit found {scikit_inliers} inliers in {scikit_time}s.')

print(f'{fastest} was ~{int(max(pydegensac_time, scikit_time) / min(pydegensac_time, scikit_time))} times faster.')
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

#We will draw correspondences found and the geometric transformation between the images.





def decolorize(img):

    return cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)





def draw_matches(kps1, kps2, tentatives, img1, img2, mask):

    matchesMask = mask.ravel().tolist()

    h,w,ch = img1.shape

    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    img_out = cv2.drawMatches(img1,kps1,img2,kps2,tentatives, None)

    plt.figure()

    fig, ax = plt.subplots(figsize=(15, 15))

    ax.imshow(img_out, interpolation='nearest')

    return





images = [decolorize(cv2.imread(image_path)) for image_path in IMAGE_PATHS]



_, pydegensac_mask = pydegensac.findHomography(src_pts, dst_pts,

                                       MAX_REPROJECTION_ERROR,

                                       HOMOGRAPHY_CONFIDENCE,

                                       MAX_RANSAC_ITERATIONS)



_, scikit_mask = scikit_ransac((src_pts, dst_pts),

                                          scikit_AffineTransform,

                                          min_samples=3,

                                          residual_threshold=20,

                                          max_trials=MAX_RANSAC_ITERATIONS)



cv2_kp0 = [cv2.KeyPoint(kp[1], kp[0], 1) for kp in kp_0]

cv2_kp1 = [cv2.KeyPoint(kp[1], kp[0], 1) for kp in kp_1]



draw_matches(cv2_kp0, cv2_kp1, tentative_matches, images[0], images[1], scikit_mask)

draw_matches(cv2_kp0, cv2_kp1, tentative_matches, images[0], images[1], pydegensac_mask)
