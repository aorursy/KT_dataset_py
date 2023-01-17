import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visulization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

import gc # garbage collection

import glob # extract path via pattern matching

from tqdm.notebook import tqdm # progressbar

import random

import math

import cv2 # read image

# store to disk



from sklearn.model_selection import train_test_split
ROOT_DIR = '../input/state-farm-distracted-driver-detection/'

TRAIN_DIR = ROOT_DIR + 'imgs/train/'

TEST_DIR = ROOT_DIR + 'imgs/test/'

driver_imgs_list = pd.read_csv(ROOT_DIR + "driver_imgs_list.csv")

sample_submission = pd.read_csv(ROOT_DIR + "sample_submission.csv")
random_list = np.random.permutation(len(driver_imgs_list))[:50]

df_copy = driver_imgs_list.iloc[random_list]

image_paths = [TRAIN_DIR+row.classname+'/'+row.img 

                   for (index, row) in df_copy.iterrows()]
img_path_list = []

label_list = []

for index, row in driver_imgs_list.iterrows():

    img_path_list.append('{0}{1}/{2}'.format(TRAIN_DIR, row.classname, row.img))

    label_list.append(int(row.classname[1]))

# One hot vector representation of labels

y_labels = np.array(label_list, dtype=np.int8)

x_img_path = np.array(img_path_list)
np.save('x_img_path.npy', x_img_path)

np.save('y_labels.npy', y_labels)
from sklearn.utils import shuffle



x_img_path_shuffled, y_labels_shuffled = shuffle(x_img_path, y_labels)



# saving the shuffled file.

# you can load them later using np.load().

np.save('y_labels_shuffled.npy', y_labels_shuffled)

np.save('x_img_path_shuffled.npy', x_img_path_shuffled)
# Used this line as our filename array is not a numpy array.

x_img_path_shuffled_numpy = np.array(x_img_path_shuffled)



X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(

    x_img_path_shuffled_numpy, y_labels_shuffled, test_size=0.2, random_state=1)



print(X_train_filenames.shape) # (3800,)

print(y_train.shape)           # (3800, 12)



print(X_val_filenames.shape)   # (950,)

print(y_val.shape)             # (950, 12)



# You can save these files as well. As you will be using them later for training and validation of your model.

np.save('X_train_filenames.npy', X_train_filenames)

np.save('y_train.npy', y_train)



np.save('X_val_filenames.npy', X_val_filenames)

np.save('y_val.npy', y_val)
ORB_extractor = cv2.ORB_create(nfeatures=200)

all_descriptors = []

for filepath in X_train_filenames:

    img = cv2.imread(filepath, 0)

    points, desc = ORB_extractor.detectAndCompute(img, None)

    all_descriptors.append(desc)
kmeans_features = np.vstack(tuple(all_descriptors[:2000]))

all_features = np.vstack(tuple(all_descriptors))

np.save('kmeans_features.npy', kmeans_features)

np.save('all_features.npy', all_features)
from sklearn.cluster import KMeans

kmeans_model = KMeans(n_clusters = 50).fit(kmeans_features)
def extractFeatures(kmeans, descriptor_list, no_clusters):

    image_count = len(descriptor_list)

    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])

    for i in range(image_count):

        for j in range(len(descriptor_list[i])):

            feature = descriptor_list[i][j]

            feature = feature.reshape(1, 32)

            idx = kmeans.predict(feature)

            im_features[i][idx] += 1



    return im_features
im_features = extractFeatures(kmeans_model, all_descriptors, 50)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler().fit(im_features)        

im_features_normed = scale.transform(im_features)
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

def svcParamSelection(X, y, kernel, nfolds):

    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]

    gammas = [0.1, 0.11, 0.095, 0.105]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search.best_params_



def findSVM(im_features, train_labels, kernel):

    features = im_features   

    params = svcParamSelection(features, train_labels, kernel, 5)

    C_param, gamma_param = params.get("C"), params.get("gamma")

    print(C_param, gamma_param)

    svm = SVC(kernel = kernel, C =  C_param, gamma = gamma_param)

    svm.fit(features, train_labels)

    return svm
svm_model = findSVM(im_features_normed, y_train, "linear")
def extractFeatures(kmeans_model, descriptor_list, image_count, no_clusters=50):

    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])

    for i in range(image_count):

        for j in range(len(descriptor_list[i])):

            feature = descriptor_list[i][j]

            feature = feature.reshape(1, 32)

            idx = kmeans_model.predict(feature)

            im_features[i][idx] += 1

    return im_features



def test_model(kmeans_model, svm_model, test_x, test_y):

    ORB_extractor_test = cv2.ORB_create(nfeatures=200)

    all_descriptors = []

    count = 0

    for filepath in test_x:

        img = cv2.imread(filepath, 0)

        _, desc = ORB_extractor_test.detectAndCompute(img, None)

        if desc is not None:

            all_descriptors.append(desc)

            count += 1

    test_features = extractFeatures(kmeans_model, all_descriptors, count, 50)

    test_features = scale.transform(test_features)

  

    predictions = svm_model.predict(test_features)

    return predictions

    print("Test images classified.")



    #plotConfusions(true, predictions)

    print("Confusion matrixes plotted.")



    #findAccuracy(true, predictions)

    print("Accuracy calculated.")

    print("Execution done.")
result = test_model(kmeans_model, svm_model, X_val_filenames, y_val)
correct = sum(y_val == result)
correct/y_val.shape[0]