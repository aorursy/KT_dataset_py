#version8.0

#BumBelBee

#!pip install imutils
import numpy as np

import cv2

#import imutils

import csv

import argparse

import glob

import math

import pandas as pd

import matplotlib.pyplot as plt

import sys

#Variables

#ColorDiscripteur

bins = (8, 12, 3)

#Sercher

indexPath = "index001.csv"

indexTestPath = "index_test.csv"



databasePath = "DB2C"



#colorDiscripteur

def describe(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    features = []

    (h, w) = image.shape[:2]

    (cX, cY) = (int(w * 0.5), int(h * 0.5))

    segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),

        (0, cX, cY, h)]

    (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)

    ellipMask = np.zeros(image.shape[:2], dtype = "uint8")

    cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)



    for (startX, endX, startY, endY) in segments:

        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")

        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)

        cornerMask = cv2.subtract(cornerMask, ellipMask)

        hist = histogram(image, cornerMask)

        features.extend(hist)

    hist = histogram(image, ellipMask)

    features.extend(hist)



    return features



def histogram(image, mask):

    hist = cv2.calcHist([image], [0, 1, 2], mask, bins,

        [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():

        hist = cv2.normalize(hist).flatten()

    else:

        hist = cv2.normalize(hist, hist).flatten()



    return hist



#Sercher

def search(queryFeatures, limit = 3):

    results = {}

    with open(indexPath) as f:

        reader = csv.reader(f)



        for row in reader:

            features = [float(x) for x in row[1:]]

            d = chi2_distance(features, queryFeatures)

            results[row[0]] = d

        f.close()

    results = sorted([(v, k) for (k, v) in results.items()])



    return results[:limit]



def chi2_distance(histA, histB, eps = 1e-10):

    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)

        for (a, b) in zip(histA, histB)])



    return d
#data_Train_Path = "../input/asl-alphabet/asl_alphabet_train"



#output = open(indexPath, "w")



#for imagePath in glob.glob(data_Train_Path + "/*/*/*.jpg"):

    #imageID = imagePath[imagePath.rfind("\\") + 1:]

    #target = imagePath[imagePath.rfind("train")+6 :imagePath.rfind("train")+7]

    #image = cv2.imread(imagePath)

    #features = describe(image)

    #features = [str(f) for f in features]

    #output.write("%s,%s\n" % (imageID+","+target, ",".join(features)))

#output.close()

data = pd.read_csv('../input/alphabet-asl-indexs/index001.csv', header=None)

col_list =['name','target'] + ['s' + str(x) for x in range(0,1440)]

data.columns = col_list

y = data.target

data = data.drop('name',1)

X_train = data.drop('target',1)

del data



## transform categorical target to numbers (encoding the target columns)

y[y=='A']=1

y[y=='B']=2

y[y=='C']=3

y[y=='D']=4

y[y=='d']=5

y[y=='E']=6

y[y=='F']=7

y[y=='G']=8

y[y=='H']=9

y[y=='I']=10

y[y=='J']=11

y[y=='K']=12

y[y=='L']=13

y[y=='M']=14

y[y=='N']=15

y[y=='n']=16

y[y=='O']=17

y[y=='P']=18

y[y=='Q']=19

y[y=='R']=20

y[y=='S']=21

y[y=='s']=22

y[y=='T']=23

y[y=='U']=24

y[y=='V']=25

y[y=='W']=26

y[y=='X']=27

y[y=='Y']=28

y[y=='Z']=29





y=y.astype('int')

y.unique()
#data_Test_Path = "../input/asl-alphabet/asl_alphabet_test"



#output = open(indexTestPath, "w")

#X_test = []

#for imagePath in glob.glob(data_Test_Path + "/*/*.jpg"):

    #imageID = imagePath[imagePath.rfind("\\") + 1:imagePath.rfind(".")-5 ]

    #image = cv2.imread(imagePath)

    #features = describe(image)

    #features = [str(f) for f in features]

    #output.write("%s,%s\n" % (str(imageID), ",".join(features))) 



#output.close()
col_list =['imagePath'] + ['s' + str(x) for x in range(0,1440)]

X_test = pd.read_csv('../input/alphabet-asl-indexs/index_test.csv', header=None)

X_test.columns = col_list

Y_test = X_test['imagePath']

for i in range(len(Y_test)):

    Y_test[i] = Y_test[i][Y_test[i].rfind("/") + 1:]

X_test = X_test.drop('imagePath',1)



Y_test[Y_test=='A']=1

Y_test[Y_test=='B']=2

Y_test[Y_test=='C']=3

Y_test[Y_test=='D']=4

Y_test[Y_test=='del']=5

Y_test[Y_test=='E']=6

Y_test[Y_test=='F']=7

Y_test[Y_test=='G']=8

Y_test[Y_test=='H']=9

Y_test[Y_test=='I']=10

Y_test[Y_test=='J']=11

Y_test[Y_test=='K']=12

Y_test[Y_test=='L']=13

Y_test[Y_test=='M']=14

Y_test[Y_test=='N']=15

Y_test[Y_test=='nothing']=16

Y_test[Y_test=='O']=17

Y_test[Y_test=='P']=18

Y_test[Y_test=='Q']=19

Y_test[Y_test=='R']=20

Y_test[Y_test=='S']=21

Y_test[Y_test=='space']=22

Y_test[Y_test=='T']=23

Y_test[Y_test=='U']=24

Y_test[Y_test=='V']=25

Y_test[Y_test=='W']=26

Y_test[Y_test=='X']=27

Y_test[Y_test=='Y']=28

Y_test[Y_test=='Z']=29



Y_test = Y_test.astype('int')
print(X_train.shape,y.shape)

print(X_test.shape,Y_test.shape)

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



scaler = StandardScaler()

pca = PCA(n_components=13)



X_train_transformed = pca.fit_transform(X_train)

X_test_transformed = pca.transform(X_test)



X_train_transformed = scaler.fit_transform(X_train_transformed)

X_test_transformed = scaler.transform(X_test_transformed)
from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.externals import joblib



#model = svm.SVC(kernel='linear')

#model = svm.SVC(gamma='scale',verbose=True)

model = RandomForestClassifier()



model.fit(X_train_transformed,y)

#joblib_file = "ASL_Alphabet_Model.pkl"

#joblib.dump(model, joblib_file)



Y_pred = model.predict(X_test_transformed)





#X_train,X_test,Y_train,Y_test=train_test_split(D['data'],D['target'],test_size=0.3,random_state=random.seed())
print('score of model :',model.score(X_test_transformed,Y_test),'\n\n')

for i in range(len(Y_test)):

    print(Y_pred[i],Y_test[i])