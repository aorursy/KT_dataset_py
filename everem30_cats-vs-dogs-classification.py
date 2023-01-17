# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/train"))

# Any results you write to the current directory are saved as output.
# Adopted (and modified) from https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import imutils # A simple image utility library. https://github.com/jrosebr1/imutils
import cv2 # opencv library
import os
print("Imports done")
def image_to_feature_vector(image, size=(32, 32)):
# resize the image to a fixed size, then flatten the image into
# a list of raw pixel intensities
    return cv2.resize(image, size).flatten()
print("Function defined")
dataset = "../input/train/train/"
print("Dataset variable created")
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
		print("[INFO] Processed {}/{}".format(i, len(imagePaths)))
        
print("All images processed")
# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
labels = np.array(labels)
print("[INFO] Pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
bUseCompleteDataset = False

# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
if bUseCompleteDataset:
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages, labels, test_size=0.25, random_state=42)
    print("Complete train and test sets created")
else:
    # Select a subset of the entire dataset 
    rawImages_subset = rawImages[:2000]
    labels_subset= labels[:2000]
    (trainRI, testRI, trainRL, testRL) = train_test_split(
        rawImages_subset, labels_subset, test_size=0.25, random_state=42)
    print("Small train and test sets created")
# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] Evaluating raw pixel accuracy...")
neighbors = [1, 3, 5, 7, 13]

print("Evaluation in progress")
for k in neighbors:
    knn_model = KNeighborsClassifier(n_neighbors= k)
    knn_model.fit(trainRI, trainRL)
    acc = knn_model.score(testRI, testRL)
    print("[INFO] Raw pixel accuracy ({} neighbors): {:.2f}%".format(k, acc * 100))
print("Evaluation completed")
bClassifierTesting_neural = False
if bClassifierTesting_neural:
    layer_sizes = [100, 300, 500]
    activation_types = ['relu', 'identity', 'logistic', 'tanh']
    rates = ['constant', 'invscaling', 'adaptive']
else:
    layer_sizes = [300]
    activation_types = ['relu']
    rates = ['adaptive']

print("Evaluation in progress")
for size in layer_sizes:
    for act in activation_types:
        for rate in rates:
            mlp_model = MLPClassifier(hidden_layer_sizes=size, activation=act, learning_rate=rate)
            mlp_model.fit(trainRI, trainRL)
            acc = mlp_model.score(testRI, testRL)
            print("[INFO] Raw pixel accuracy ({}, {}, {}): {:.2f}%".format(size, act, rate, acc * 100))     
print("Evaluation completed")
bClassifierTesting_support = False
if bClassifierTesting_support:
    c_values = [1.0, 5.0, 10.0]
    kernel_types = [ 'rbf', 'linear', 'poly','sigmoid']
else:
    c_values = [1.0]
    kernel_types = ['linear']

print("Evaluation in progress")
for c_val in c_values:
        for k in kernel_types:
            svc_model = SVC(C=c_val, kernel=k, gamma='scale')
            svc_model.fit(trainRI, trainRL)
            acc = svc_model.score(testRI, testRL)
            print("[INFO] Raw pixel accuracy ({}, {}): {:.2f}%".format(c_val, k, acc * 100))     
print("Evaluation completed")