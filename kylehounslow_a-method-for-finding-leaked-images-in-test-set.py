from __future__ import print_function

import os

import glob

import cv2

from matplotlib import pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

# print some info about server

print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["df", "-h"]).decode("utf8"))



test_root = '../input/test'

train_root = '../input/train'

additional_train_root = '../input/additional'



test_paths = glob.glob(test_root + '/*.jpg')

#train_paths = glob.glob(train_root + '/**/*.jpg') + glob.glob(additional_train_root + '/**/*.jpg')

train_paths = glob.glob(additional_train_root + '/Type_2/*.jpg') # only use type 2 additional for this demo

# list some files as sanity check

print (len(test_paths))

print (test_paths[:10])

print (len(train_paths))

print (train_paths[:10])
def get_histogram_features(img):

    """

    Get 1D feature vector of RGB histogram

    """

    # here we get 12x12x12 histogram

    hist = cv2.calcHist([img], [0, 1, 2], 

                        None, 

                        [12, 12, 12], 

                        [0, 256, 0, 256, 0, 256])

    features = np.array(hist).astype(np.float32).flatten() # flatten to 1D

    features /= 255. # normalize between 0.0 and 1.0

    return features
test_vectors = []

print('begin extracting histogram features from test images...')

for i, pth in enumerate(test_paths):

    img = cv2.imread(pth)

    features = get_histogram_features(img)

    #print(features.shape)

    test_vectors.append(features)

    if i%50 == 0 or i == len(test_paths) - 1:

        print ('{} of {} test vectors loaded'.format(i + 1, len(test_paths)))

test_vectors = np.array(test_vectors)

print ('done.')

print('test_vectors.shape={}'.format(test_vectors.shape))
train_vectors = []

trim_to_amount = 1000 # only small chunk of images for brevity

train_paths = train_paths[:trim_to_amount]

print('begin extracting histogram features from train images...')

for i, pth in enumerate(train_paths):

    img = cv2.imread(pth)

    if img is None or img.shape[0] == 0 or img.shape[1] == 0:

        raise Exception('corrupt image {}'.format(pth)) # TODO: handle corrupt images

    features = get_histogram_features(img)

    #print(features.shape)

    train_vectors.append(features)

    if i%50 == 0 or i == len(train_paths) - 1:

        print ('{} of {} train vectors loaded'.format(i + 1, len(train_paths)))

train_vectors = np.array(train_vectors)

print ('done loading features.')

print('train_vectors.shape={}'.format(train_vectors.shape))
from sklearn.neighbors import KDTree

def find_duplicates(min_match_dist=2200.0):

    # load train vectors into KDTree

    kd = KDTree(train_vectors, leaf_size=40, metric='euclidean')

    # find K closest vectors to each test vector

    k = 1

    # compare test/train vectors to find duplicates

    # Note: we could also find all distances and indices in one shot with kd.query(X=test_vectors)

    print ('beginning KNN search...')

    for i, test_vector in enumerate(test_vectors):

        dists, indices = kd.query(X=test_vector.reshape(1, -1), 

                                  k=k, 

                                  return_distance=True)

        dists = dists[0]

        indices = indices[0]



        skip = False

        for j, ind in enumerate(indices):

            distance = dists[j]

            if distance > min_match_dist:

                skip = True

                continue

            train_img_path = train_paths[ind]

            train_img = cv2.imread(train_img_path)

            train_img = cv2.resize(train_img, (256, 256))  # resize for display

            train_filename = os.path.basename(train_img_path)

            train_class_type = os.path.basename(os.path.dirname(train_img_path))

            # write some useful text on each image (filename, distance, class type)

            cv2.putText(train_img, train_filename, (0, 25), 1, 1.75, (0, 255, 0), 2)

            cv2.putText(train_img, 'dist={}'.format(distance), (0, 50), 1, 1.75, (0, 255, 0), 2)

            cv2.putText(train_img, train_class_type, (0, 75), 1, 1.75, (0, 255, 0), 2)

            plt.subplot(1, k + 1, j + 2)  # plot result image

            plt.title('{}'.format(j + 1))

            plt.axis('off')

            plt.imshow(cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB))

        if skip is not True:

            test_img_path = test_paths[i]

            test_img = cv2.imread(test_img_path)

            test_img = cv2.resize(test_img, (256, 256))  # resize for display

            plt.subplot(1, k + 1, 1)  # plot query image

            plt.title('query image')

            plt.axis('off')

            test_filename = os.path.basename(test_img_path)

            cv2.putText(test_img, test_filename, (0, 25), 1, 1.75, (0, 0, 255), 2)

            plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

            plt.show()

    print('done.')
min_match_dist = 2200.0 # found experimentally, feel free to increase/decrease

find_duplicates(min_match_dist=min_match_dist)