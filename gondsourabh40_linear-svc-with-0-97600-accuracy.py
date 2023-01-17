import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import os

df = pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

df.head()
df.label.value_counts().plot.bar()
#X_train is Input Vector for Training and y_train is corresponding output of X_train

X_train = df.drop(labels =["label"],axis = 1)

y_train = df.label

X_train
X_train = X_train.apply(lambda x:x.values.reshape(28,28),axis=1)

X_test = X_test.apply(lambda x:x.values.reshape(28,28),axis=1)
from skimage import feature,img_as_ubyte

for img in X_train.head(5):

    img = img.astype('uint8')

    img = img_as_ubyte(img)

    plt.figure()

    plt.imshow(img,cmap='gray')
def calculateHog(img):

    winSize = (10,10)

    blockSize = (5,5)

    blockStride = (5,5)

    cellSize = (5,5)

    nbins = 9

    derivAperture = 1

    winSigma = -1.

    histogramNormType = 0

    L2HysThreshold = 0.2

    gammaCorrection = 1

    nlevels = 64

    signedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)

    descriptor = hog.compute(img)

    return descriptor
X_train = X_train.apply(lambda x:calculateHog(x.astype('uint8')).T[0])

X_test = X_test.apply(lambda x:calculateHog(x.astype('uint8')).T[0])
X_train = pd.DataFrame.from_records(X_train.values)

X_test = pd.DataFrame.from_records(X_test.values)

print(X_train.shape)
from sklearn.svm import LinearSVC

classifier = LinearSVC(max_iter=10000)

classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_train)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_train,y_pred))
y_test = classifier.predict(X_test)
df = pd.DataFrame(y_test,columns=['Label'])

df['ImageId'] = df.index+1

df = df[['ImageId','Label']]

df.to_csv('final_submission.csv',index=False)