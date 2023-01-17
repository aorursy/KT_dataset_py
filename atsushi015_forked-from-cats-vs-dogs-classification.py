# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# You may need to install a custom package named *imutils* to run this notebook. 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/train"))



# Any results you write to the current directory are saved as output.
# Adopted (and modified) from https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.feature_selection import RFE, SelectFromModel

from imutils import paths

import numpy as np

import imutils # a simple image utility library

import cv2 #opencv library

import os
def image_to_feature_vector(image, size=(32, 32)):

# resize the image to a fixed size, then flatten the image into

# a list of raw pixel intensitieb

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    return cv2.resize(ycrcb, size, interpolation=cv2.INTER_AREA).flatten()

dataset = "../input/train/train/"
# grab the list of images that we'll be describing

print("[INFO] describing images...")

imagePaths = list(paths.list_images(dataset))

print(len(imagePaths))

print(imagePaths[0])

# initialize the raw pixel intensities matrix, the features matrix,

# and labels list

rawImages = []

labels = []
# loop over the input images

for (i, imagePath) in enumerate(imagePaths):

	# load the image and extract the class label (assuming that our

	# path as the format: /path/to/dataset/{class}.{image_num}.jpg

	image = cv2.imread(imagePath)

	label = 1 if imagePath.split(os.path.sep)[-1].split(".")[0] == "dog" else 0

 

	# extract raw pixel intensity "features", followed by a color

	# histogram to characterize the color distribution of the pixels

	# in the image

	pixels = image_to_feature_vector(image)

	

 

	# update the raw images, features, and labels matricies,

	# respectively

	rawImages.append(pixels)

	labels.append(label)

 

	# show an update every 1,000 images

	if i > 0 and i % 1000 == 0:

		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
# show some information on the memory consumed by the raw images

# matrix and features matrix

rawImages = np.array(rawImages)

labels = np.array(labels)

print("[INFO] pixels matrix: {:.2f}MB".format(

	rawImages.nbytes / (1024 * 1000.0)))
# partition the data into training and testing splits, using 75%

# of the data for training and the remaining 25% for testing

(trainRI, trainRL) = (rawImages, labels)

# Select a subset of the entire dataset 

rawImages_subset = rawImages[:2000]

labels_subset= labels[:2000]

(trainRI, trainRL) = (rawImages_subset, labels_subset)

# train and evaluate a AdaBoost classifer on the raw pixel intensities

print("[INFO] evaluating raw pixel accuracy...")

#neighbors = [1, 3, 5, 7, 13]

#for k in neighbors:

model = AdaBoostClassifier(n_estimators=300)

acc = cross_val_score(model, trainRI, trainRL, cv=3)

print("[INFO] raw pixel accuracy: {}".format(acc))
# train and evaluate a XGBoost classifer on the raw pixel intensities

print("[INFO] evaluating raw pixel accuracy...")

model = XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=6, colsample_bytree=0.2, subsample=0.9, objective="binary:logistic")

acc = cross_val_score(model, trainRI, trainRL, cv=3)

print("[INFO] raw pixel accuracy: {}".format(acc))
# train and evaluate a Logistic Regression classifer on the raw pixel intensities

print("[INFO] evaluating raw pixel accuracy...")

model = LogisticRegression(solver='liblinear', max_iter=300)

acc = cross_val_score(model, trainRI, trainRL, cv=3)

print("[INFO] raw pixel accuracy: {}".format(acc))
(trainRI, trainRL) = (rawImages, labels)



print("[INFO] evaluating raw pixel accuracy...")

model = XGBClassifier(learning_rate=0.1, n_estimators=300, max_depth=6, colsample_bytree=0.2, subsample=0.9, objective="binary:logistic")

acc = cross_val_score(model, trainRI, trainRL, cv=3)

print("[INFO] raw pixel accuracy: {}".format(acc))