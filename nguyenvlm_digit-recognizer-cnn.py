# !pip install hyperas

!pip install opencv-contrib-python==3.4.2.16
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import cv2

print(cv2.__version__)

print(os.listdir("../input"))
def plot_eval_result(train_hist, figsize=(20,8), title='Evaluation Result'):

    plt.figure(figsize=figsize)

    for i in train_hist.history.keys():

        plt.plot(train_hist.history[i])

    plt.xlabel('Epoch')

    plt.ylabel('Metric')

    plt.show()
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X_train = np.uint8(train.iloc[:, 1:].values).reshape(-1,28,28,1)

Y_train = train.iloc[:, 0]

x_test = np.uint8(test.values).reshape(-1,28,28,1)
print(X_train.shape, Y_train.shape, x_test.shape)
# Seed value

seed_value= 1998



# Set `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED']=str(seed_value)



# Set `python` built-in pseudo-random generator at a fixed value

import random

random.seed(seed_value)



# Set `numpy` pseudo-random generator at a fixed value

np.random.seed(seed_value)



# Set `tensorflow` pseudo-random generator at a fixed value

import tensorflow as tf

tf.set_random_seed(seed_value)



# Set GPU Availability and force TF to use only 1 thread:

from keras.backend import set_session, tensorflow_backend

config = tf.ConfigProto(

    device_count = {'GPU': 1, 'CPU': 100}, 

#     intra_op_parallelism_threads=1,

#     inter_op_parallelism_threads=1

)

set_session(tf.Session(config=config))

print(tensorflow_backend._get_available_gpus())
# X_train = X_train/255

# x_test = x_test/255
# from sklearn.model_selection import StratifiedKFold



# KFold = 5

# Folds = StratifiedKFold(n_splits=KFold, shuffle=True).split(X_train, Y_train)
sift = cv2.xfeatures2d.SIFT_create(5)



def fd_sift(image):

    kps, des = sift.detectAndCompute(image, None)

    return des if des is not None else np.array([]).reshape(0, 128)
X_sift = [fd_sift(img).reshape(10, 128) for img in X_train]

print(len(X_sift))

print(len(X_sift[0]))
def my_dist(des1, des2):

    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)   # or pass empty dictionary

    return len(cv2.FlannBasedMatcher(index_params, search_params))
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=10, p=2, metric=my_dist)
clf.fit(X_sift, Y_train)
y_pred = pd.DataFrame(data = final_result, index=list(range(1, test.shape[0]+1) ), columns = ['Label'])

y_pred.to_csv("output.csv", index=True, index_label='ImageId')