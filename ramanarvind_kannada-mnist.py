# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 as opencv

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import svm

from sklearn.decomposition import PCA

#import mahotas



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

print('Train shape = ', train.shape)

splitTrain, splitTest, splitTrainLabels, splitTestLabels = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2)

print('splitTrain size = ', splitTrain.shape)

print('splitTest size = ', splitTest.shape)
# Prepare the training dataset for the KNN classifier. The KNN classifier requires the data and the labels

# to be a float32 format

knnTrain = np.array(splitTrain).astype(np.float32)

knnTrainLabels = np.array(splitTrainLabels).astype(np.float32)



# Train the KNN classifier

knn = opencv.ml.KNearest_create()

knn.train(knnTrain, opencv.ml.ROW_SAMPLE, knnTrainLabels)



# Prepare the training dataset for the KNN classifier. The KNN classifier requires the  data and the labels

# to be a float32 format

knnTest = np.array(splitTest).astype(np.float32)

knnTestLabels = np.array(splitTestLabels).astype(np.float32)



# Run the KNN classifier

ret, result, neighbours, dist = knn.findNearest(knnTest, k=5)



# Measure the accuracy

knnTestLabels = knnTestLabels.reshape(knnTestLabels.size, 1)

matches  = result == knnTestLabels

correct  = np.count_nonzero(matches) 

accuracy = correct * 100.0 / result.size

knnCM = confusion_matrix(result, knnTestLabels)

print('KNN Accuracy = ', accuracy)

print(knnCM)
# Prepare the training dataset for the KNN classifier. The KNN classifier requires the data and the labels

# to be a float32 format

knnTrain = np.array(splitTrain).astype(np.float32)

knnTrainLabels = np.array(splitTrainLabels).astype(np.float32)



# Train the KNN classifier

knn = opencv.ml.KNearest_create()

knn.train(knnTrain, opencv.ml.ROW_SAMPLE, knnTrainLabels)



# Prepare the training dataset for the KNN classifier. The KNN classifier requires the  data and the labels

# to be a float32 format

knnTest = np.array(splitTest).astype(np.float32)

knnTestLabels = np.array(splitTestLabels).astype(np.float32)



# Run the KNN classifier

ret, result, neighbours, dist = knn.findNearest(knnTest, k=5)



# Measure the accuracy

knnTestLabels = knnTestLabels.reshape(knnTestLabels.size, 1)

matches  = result == knnTestLabels

correct  = np.count_nonzero(matches) 

accuracy = correct * 100.0 / result.size

knnCM = confusion_matrix(result, knnTestLabels)

print('KNN Accuracy = ', accuracy)

print(knnCM)
def fd_hu_moments(rasterImg):

    moments = opencv.HuMoments(opencv.moments(rasterImg, binaryImage=True)).flatten()

    return moments



"""

def fd_haralick(rasterImg):

    haralick = mahotas.features.haralick(image).mean(axis=0)

    return haralick

"""

image = splitTrain.to_numpy()



rasterImg = image[0].reshape(28, 28)

fdTrain = fd_hu_moments(rasterImg)

for i in range (1, image.shape[0]):

    rasterImg = image[i].reshape(28, 28)

    moments = fd_hu_moments(rasterImg)

    #haralick = fd_haralick(rasterImg)

    fdTrain = np.vstack((fdTrain, moments))



print('Starting to fit the SVM classifier')

sv = svm.SVC(kernel='rbf', C=9, gamma='scale', decision_function_shape='ovo')

sv.fit(fdTrain, splitTrainLabels)

print('Finished fitting the SVM classifier')



# Measure the accuracy against the training dataset

image = splitTest.to_numpy()

rasterImg = image[0].reshape(28, 28)

moments = fd_hu_moments(rasterImg)

fdTest = moments

for i in range (1, image.shape[0]):

    rasterImg = image[i].reshape(28, 28)

    moments = fd_hu_moments(rasterImg)

    #haralick = fd_haralick(rasterImg)

    fdTest = np.vstack((fdTest, moments))

result = sv.predict(fdTest)



# Measure accuracy

svmTestLabels = np.array(splitTestLabels).astype(np.float32)

matches  = result == svmTestLabels

correct  = np.count_nonzero(matches) 

accuracy = correct * 100.0 / result.size

svmCM = confusion_matrix(result, svmTestLabels)

print('SVM Accuracy = ', accuracy)

print(svmCM)
def fd_hu_moments(rasterImg):

    moments = opencv.HuMoments(opencv.moments(rasterImg, binaryImage=True)).flatten()

    return moments



def fd_haralick(rasterImg):

    haralick = mahotas.features.haralick(image).mean(axis=0)

    return haralick



image = splitTrain.to_numpy()



rasterImg = image[0].reshape(28, 28)

fdTrain = fd_hu_moments(rasterImg)

for i in range (1, image.shape[0]):

    rasterImg = image[i].reshape(28, 28)

    moments = fd_hu_moments(rasterImg)

    #haralick = fd_haralick(rasterImg)

    fdTrain = np.vstack((fdTrain, moments))



print('Starting to fit the SVM classifier')

sv = svm.SVC(kernel='rbf', C=9, gamma='scale', decision_function_shape='ovo')

sv.fit(fdTrain, splitTrainLabels)

print('Finished fitting the SVM classifier')



# Measure the accuracy against the training dataset

image = splitTest.to_numpy()

rasterImg = image[0].reshape(28, 28)

moments = fd_hu_moments(rasterImg)

fdTest = moments

for i in range (1, image.shape[0]):

    rasterImg = image[i].reshape(28, 28)

    moments = fd_hu_moments(rasterImg)

    #haralick = fd_haralick(rasterImg)

    fdTest = np.vstack((fdTest, moments))

result = sv.predict(fdTest)



# Measure accuracy

svmTestLabels = np.array(splitTestLabels).astype(np.float32)

matches  = result == svmTestLabels

correct  = np.count_nonzero(matches) 

accuracy = correct * 100.0 / result.size

svmCM = confusion_matrix(result, svmTestLabels)

print('SVM Accuracy = ', accuracy)

print(svmCM)
# Train the model against the training dataset

pca = PCA(n_components=0.7, svd_solver='full', whiten=True)

trainPCA = pca.fit_transform(splitTrain)

sv = svm.SVC(kernel='rbf', C=9, gamma='scale', decision_function_shape='ovo')

sv.fit(trainPCA, splitTrainLabels)



# Measure the accuracy against the cross validation dataset

testPCA = pca.transform(splitTest)

result = sv.predict(testPCA)



# Measure accuracy

matches  = result == svmTestLabels

correct  = np.count_nonzero(matches) 

accuracy = correct * 100.0 / result.size

svmCM = confusion_matrix(result, svmTestLabels)

print('SVM Accuracy = ', accuracy)

print(svmCM)
test = pd.read_csv('../input/Kannada-MNIST/test.csv')

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

print(test.shape)

testPCA = pca.transform(test.iloc[:, 1:])

result = sv.predict(testPCA)

submission['label'] = result

submission.to_csv('submission.csv', index=False)

submission.head()