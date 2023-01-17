# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import imageio
import torch

import sys

import scipy
import scipy.io
import scipy.misc

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/landmark-recognition-2020/train.csv')
# Support functions
def get_path_to_image(id_, base_folder='../input/landmark-recognition-2020/train'):
    return os.path.join(base_folder, id_[0], id_[1], id_[2], id_ + '.jpg')
    
def get_list_of_id_by_landmark_id(df, landmark_id):
    return df.loc[df['landmark_id'] == landmark_id]['id'].values

def get_first_id_by_landmark_id(df, landmark_id):
    return df.loc[df['landmark_id'] == landmark_id]['id'].values[0]
landmark_id_set = list(set(train_df['landmark_id'].values))
print('lenght of training images set (unique landmark id)', len(landmark_id_set))
f = open('images_train.txt', 'w')
for i in range(len(landmark_id_set) // 100):
    image_path = get_path_to_image(get_first_id_by_landmark_id(train_df, landmark_id_set[i]))
    f.write(image_path + '\n')
#     print('{}/{}'.format(i, len(landmark_id_set)), end='\r')
f.close()

path_to_test = '../input/landmark-recognition-2020/test'
f = open('images_test.txt', 'w')
for root, dirs, files in os.walk(path_to_test, topdown=False):
   for name in files:
    image_path = get_path_to_image(name.split('.')[0], base_folder='../input/landmark-recognition-2020/test')
    f.write(image_path + '\n')
f.close()

sys.path.append('../input/googlelandmarkd2netmodel/d2_net')

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# # Process the file
preprocessing = 'caffe'
model_file = '../input/googlelandmarkd2netmodel/d2_net/models/d2_tf.pth'
max_edge = 1600
max_sum_edges = 2800
multiscale = False
use_relu = True

# Creating CNN model
model = D2Net(
    model_file=model_file,
    use_relu=use_relu,
    use_cuda=use_cuda
)

def predict(path):
    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=preprocessing
    )
    with torch.no_grad():
        if multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]
    return keypoints, scores, descriptors
pred = predict('../input/landmark-recognition-2020/train/0/0/0/0000059611c7d079.jpg')
pred
# !python3 d2-net/extract_features.py --image_list_file images_train.txt --model_file d2-net/models/d2_tf.pth  --output_path='train_data'
# !python3 d2-net/extract_features.py --image_list_file images_test.txt --model_file d2-net/models/d2_tf.pth  --output_path='test_data'
# import cv2
# import csv

# def load_labelmap(path='../input/landmark-recognition-2020/train.csv'):
#   with open(path, mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     labelmap = {row['id']: row['landmark_id'] for row in csv_reader}

#   return labelmap

# def to_hex(image_id) -> str:
#   return '{0:0{1}x}'.format(image_id, 16)

# def save_submission_csv(predictions=None):
#   """Saves optional `predictions` as submission.csv.

#   The csv has columns {id, landmarks}. The landmarks column is a string
#   containing the label and score for the id, separated by a ws delimeter.

#   If `predictions` is `None` (default), submission.csv is copied from
#   sample_submission.csv in `IMAGE_DIR`.

#   Args:
#     predictions: Optional dict of image ids to dicts with keys {class, score}.
#   """

#   if predictions is None:
#     # Dummy submission!
#     shutil.copyfile(
#         os.path.join(DATASET_DIR, 'sample_submission.csv'), 'submission.csv')
#     return

#   with open('submission.csv', 'w') as submission_csv:
#     csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])
#     csv_writer.writeheader()
#     for image_id, prediction in predictions.items():
#       label = prediction['class']
#       score = prediction['score']
#       csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})

# def match_descriptors(kp1, desc1, kp2, desc2):
#     # Match the keypoints with the warped_keypoints with nearest neighbor search
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(desc1, desc2)
#     matches_idx1 = np.array([m.queryIdx for m in matches])
#     m_kp1 = [kp1[idx] for idx in matches_idx1]
#     matches_idx2 = np.array([m.trainIdx for m in matches])
#     m_kp2 = [kp2[idx] for idx in matches_idx2]


#     return m_kp1, m_kp2, matches

# def compute_homography(matched_kp1, matched_kp2):
# #     matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
# #     matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
#     matched_kp1 = np.array(matched_kp1)
#     matched_kp2 = np.array(matched_kp2)
#     # Estimate the homography between the matches using RANSAC

#     H, inliers = cv2.findHomography(matched_kp1,
#                                     matched_kp2,
#                                     cv2.RANSAC)
#     inliers = inliers.flatten()

#     return H, inliers

# def get_scores(kp1, kp2, m_kp1, m_kp2, desc1, desc2, matches):
#     x, y = 0, 0
    
#     m_desc1 = np.zeros((len(matches), 512), dtype='float')
#     m_desc2 = np.zeros((len(matches), 512), dtype='float')
#     data1 = np.zeros((len(matches), 2), dtype='int')
#     data2 = np.zeros((len(matches), 2), dtype='int')
#     for idx, mat in enumerate(matches):
#         img1_idx = mat.queryIdx
#         img2_idx = mat.trainIdx
#         m_desc1[idx,:] = desc1[img1_idx]
#         m_desc2[idx,:] = desc2[img2_idx]
    
#         (x1, y1, _) = kp1[img1_idx]
#         (x2, y2, _) = kp2[img2_idx]
#         dx = x2
#         dy = y2
#         x += x1 - dx
#         y += y1 - dy
#         data1[idx, :] = [x1, y1]
#         data2[idx, :] = [x2, y2]

# #     data3 = np.zeros((len(kp2), 2), dtype='int')
# #     for idx, kp in enumerate(kp2):
# #         (x3, y3, _) = kp
# #         data3[idx, :] = [x3, y3]

#     sigma1 = np.var(data1, axis=0)
#     sigma1 = sigma1[0] / sigma1[1]
#     sigma2 = np.var(data2, axis=0)
#     sigma2 = sigma2[0] / sigma2[1]


#     p = (len(m_kp2)) / (len(kp2) + 0.0001)
#     score = p*np.sqrt(sigma1 / sigma2) if sigma1 < sigma2 else p*np.sqrt(sigma2 / sigma1)
#     score2 = (2*sigma1*sigma2) / (sigma1**2 + sigma2**2)
#     score3 = 100*2*len(matches) / (len(kp2) + len(kp1))
#     cosine = 1 - (np.dot(m_desc1.flatten(), m_desc2.flatten())/(np.linalg.norm(m_desc1.flatten())*np.linalg.norm(m_desc2.flatten())))
#     print('p, res_c: ', p)
#     print('score: ', score, score2, np.linalg.norm(data1-data2), len(matches), score3, cosine)
# labelmap = load_labelmap()
# for name in os.listdir('0')[0]:
#     npz_file = np.load(os.path.join('0', name))
#     desc2 = npz_file['descriptors']
#     kp2 = npz_file['keypoints']
#     scores2 = npz_file['scores']

#     for name2 in os.listdir('0')[:2]:
#         npz_file = np.load(os.path.join('0', name2))
#         desc1 = npz_file['descriptors']
#         kp1 = npz_file['keypoints']
#         scores1 = npz_file['scores']
        
#         train_id = name2.split('.')[0]
#         label = labelmap[train_id]
                         
#         m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
#         H, inliers = compute_homography(m_kp1, m_kp2)
#         matches = np.array(matches)[inliers.astype(bool)].tolist()

#         print('0' + '0', label)
#         get_scores(kp1, kp2, m_kp1, m_kp2, m_desc1, m_desc2, matches)

#     for name2 in os.listdir('1'):
#         npz_file = np.load(os.path.join('1', name2))
#         desc1 = npz_file['descriptors']
#         kp1 = npz_file['keypoints']
#         scores1 = npz_file['scores']
        
#         train_id = name2.split('.')[0]
#         label = labelmap[train_id]
    
#         m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
#         H, inliers = compute_homography(m_kp1, m_kp2)
#         matches = np.array(matches)[inliers.astype(bool)].tolist()
#         print('0' + '1', label)
#         get_scores(kp1, kp2, m_kp1, m_kp2, desc1, desc2, matches)
#     print('#'*20)

