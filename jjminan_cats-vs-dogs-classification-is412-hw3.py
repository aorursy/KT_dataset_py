# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# You may need to install a custom package named *imutils* to run this notebook. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/train"))

# Any results you write to the current directory are saved as output.
# Adopted (and modified) from https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imutils import paths
import numpy as np
import imutils # a simple image utility library
import cv2 #opencv library
import os
def image_to_feature_vector(image, size=(32, 32)):
# resize the image to a fixed size, then flatten the image into
# a list of raw pixel intensities
    return cv2.resize(image, size).flatten()

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
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
 
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
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
# Select a subset of the entire dataset 
rawImages_subset = rawImages[:2000]
labels_subset= labels[:2000]
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages_subset, labels_subset, test_size=0.25, random_state=42)
# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
neighbors = [1, 3, 5, 7, 13]
for k in neighbors:
    model = KNeighborsClassifier(n_neighbors= k)
    model.fit(trainRI, trainRL)
    acc = model.score(testRI, testRL)
    print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))
#nn_hidden_layers_sizes = [150, (150,100), 100, (100,50), 75, (75,75)]
nn_hidden_layers_sizes = [150, 100, 75, (75,75)]
nn_activation = ['identity','logistic','tanh','relu']
nn_learning_rate = ['constant','invscaling','adaptive']
print("[INFO] Raw pixel accuracy (idden layers size, activation, learning rate): accuracy")
for size in nn_hidden_layers_sizes:
    print("\n-------------------------------------------------")
    for act in nn_activation:
        for rate in nn_learning_rate:
            neural_network = MLPClassifier(hidden_layer_sizes = size, activation = act, learning_rate = rate, random_state = 1)
            neural_network.fit(trainRI, trainRL)
            accuracy = neural_network.score(testRI, testRL)
            print("[INFO] Raw pixel accuracy ({}, {}, {}): {:.2f}%".format(size, act, rate, accuracy * 100)) 
#svm_c = [1, 3, 5, 7, 13]
svm_c = [1, 7]
svm_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
#kernel precomputed gives error
# gamma = 'scale' to get rid of warning
print("[INFO] Raw pixel accuracy (C, kernel): accuracy")
for c in svm_c:
    print("\n-------------------------------------------------\n")
    for k in svm_kernel:
        svc_model = SVC(C = c, kernel = k, random_state = 1, gamma = 'scale')
        svc_model.fit(trainRI, trainRL)
        accuracy = svc_model.score(testRI, testRL)
        print("[INFO] Raw pixel accuracy ({}, {}): {:.2f}%".format(c, k, accuracy * 100)) 