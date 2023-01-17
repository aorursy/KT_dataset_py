#Import Packages

import numpy as np

import pandas as pd

import dicom

import os

import matplotlib.pyplot as plt

import cv2

import math



#defining the pixel size and the slice count

IMG_SIZE_PX = 50

SLICE_COUNT = 20
stage1_data = np.load('stage1data-50-50-20.npy')



train_data = stage1_data[:]

print(train_data.shape)



train_features = []

train_labels = []





for i in range(0,train_data.shape[0]):

    train_features = train_features + [(train_data[i,0])[0]]

    train_labels = train_labels + [(train_data[i,1])[1]]

    

for i in range(0,len(train_features)):

    for j in range(0,len(train_features[i])):

        for k in range(0,len((train_features[i])[j])):

                       if ((train_features[i])[j])[k] == -2000:

                           ((train_features[i])[j])[k] = 0



train_features = np.array(train_features)

train_labels = np.array(train_labels)



nsamples, nx, ny = train_features.shape

d2_train_features = train_features.reshape((nsamples,nx*ny))

'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(d2_train_features)

d2_train_features = scaler.transform(d2_train_features)

'''

print(train_features.shape)

print(d2_train_features.shape)

print(train_labels.shape)

   
stage2_data = np.load('stage2data-50-50-20.npy')



print('Test Data')

test_data = stage2_data[:]

print(test_data.shape)



test_features = []



for i in range(0,test_data.shape[0]):

    test_features = test_features + [(test_data[i,0])[0]]



for i in range(0,len(test_features)):

    for j in range(0,len(test_features[i])):

        for k in range(0,len((test_features[i])[j])):

                       if ((test_features[i])[j])[k] == -2000:

                           ((test_features[i])[j])[k] = 0    

    

test_features = np.array(test_features)

nsamples, nx, ny = test_features.shape

d2_test_features = test_features.reshape((nsamples,nx*ny))

'''

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(d2_test_features)

d2_test_features = scaler.transform(d2_test_features)

'''

print('Test Data')

print(test_features.shape)

print(d2_test_features.shape)
from sklearn.linear_model import LogisticRegression



from sklearn.grid_search import GridSearchCV

#Set the tuning parameter

params = {'C':[1000,1200,1400], 'tol': [0.00001]}



##Create the Logistic Regression model

logreg = LogisticRegression(solver='newton-cg', multi_class='multinomial')

##Use the Grid Search with the params

clf = GridSearchCV(logreg,params, scoring='log_loss', refit='True', n_jobs=-1, cv=5)

##Fit the models

clf_fit = clf.fit(d2_train_features,train_labels)



test_pred_labels= clf.predict(d2_test_features)



test_pred_labProb = clf.predict_proba(d2_test_features)

pred = []

for i in range(0,test_pred_labProb.shape[0]):

    pred = pred + [test_pred_labProb[i,1]]

#print(pred)

'''

test_pred_logProb = clf.predict_log_proba(d2_test_features)

pred = []

for i in range(0,test_pred_logProb.shape[0]):

    pred = pred + [test_pred_logProb[i,1]]

'''

logisticPatients_df = pd.read_csv('.../Data/stage2_sample_submission.csv')

logisticPatients_df['cancer'] = pd.Series(pred)

logisticPatients_df.to_csv(".../Data/stage2_pred_log_3.csv",index=False)