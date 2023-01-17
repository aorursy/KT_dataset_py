#load with pandas, manipulate with numpy, plot with matplotlib

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



#ML - we will classify using a naive xgb with stratified cross validation

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss







#filenames

inputFolder = "../input/"

trainSet = 'train.json'

testSet = 'test.json'

subName = 'iceberg-svd-xgb-3fold.csv'

#load data

trainDF = pd.read_json(inputFolder+trainSet)

testDF = pd.read_json(inputFolder+testSet)
#get numpy arrays for train/test data, prob there is a more pythonic approach

band1 = trainDF['band_1'].values

im1 = np.zeros((len(band1),len(band1[0])))

for j in range(len(band1)):

    im1[j,:]=np.asarray(band1[j])

    

band2 = trainDF['band_2'].values

im2 = np.zeros((len(band2),len(band2[0])))

for j in range(len(band2)):

    im2[j,:]=np.asarray(band2[j])

    

#get numpy array for test data

band1test = testDF['band_1'].values

im1test = np.zeros((len(band1test),len(band1test[0])))

for j in range(len(band1test)):

    im1test[j,:]=np.asarray(band1test[j])

    

band2test = testDF['band_2'].values

im2test = np.zeros((len(band2test),len(band2test[0])))

for j in range(len(band2test)):

    im2test[j,:]=np.asarray(band2test[j])
import cv2

from skimage import filters

from skimage import data, exposure



U1,s1,V1 = np.linalg.svd(np.vstack((im1,im1test)),full_matrices = 0)

U2,s2,V2 = np.linalg.svd(np.vstack((im2,im2test)),full_matrices = 0)

#svd of the two bands

Uh1,sh1,Vh1 = np.linalg.svd(exposure.equalize_hist(np.vstack((im1,im1test))),full_matrices = 0)

Uh2,sh2,Vh2 = np.linalg.svd(exposure.equalize_hist(np.vstack((im2,im2test))),full_matrices = 0)

print(Uh2.shape,Vh2.shape)
#original 

nmodes=20



im1p=np.dot(U1[:,:nmodes],V1[:nmodes,])

im2p=np.dot(U2[:,:nmodes],V2[:nmodes,])

im1ph=np.dot(Uh1[:,:nmodes],Vh1[:nmodes,])

im2ph=np.dot(Uh2[:,:nmodes],Vh2[:nmodes,])



nmodes = 20



X = np.hstack((U1[:len(trainDF),:nmodes],U2[:len(trainDF),:nmodes]))

X = np.hstack((X,Uh1[:len(trainDF),:nmodes]))

X = np.hstack((X,Uh2[:len(trainDF),:nmodes]))

X_test = np.hstack((U1[len(trainDF):,:nmodes],U2[len(trainDF):,:nmodes]))

X_test = np.hstack((X_test,Uh1[len(trainDF):,:nmodes]))

X_test = np.hstack((X_test,Uh2[len(trainDF):,:nmodes]))

y = trainDF['is_iceberg'].values
#is there a native xgb way of doing it?

def logloss_xgb(preds, dtrain):

    labels = dtrain.get_label()

    score = log_loss(labels, preds)

    return 'logloss', score
nfolds = 3;

xgb_mdl=[None]*nfolds





xgb_params = {

        'objective': 'binary:logistic',

        'n_estimators':1000,

        'max_depth': 8,

        'subsample': 0.9,

        'colsample_bytree': 0.9 ,

     #   'max_delta_step': 1,

     #   'min_child_weight': 10,

        'eta': 0.01,

      #  'gamma': 0.5

        }





folds = list(StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=2016).split(X, y))



d_test = xgb.DMatrix(X_test)



preds = np.zeros((X_test.shape[0],nfolds))



for j, (train_idx, valid_idx) in enumerate(folds):

    X_train = X[train_idx]

    y_train = y[train_idx]

    

    X_valid = X[valid_idx]

    y_valid = y[valid_idx]

    

    d_train =  xgb.DMatrix(X_train,label=y_train)

    d_valid =  xgb.DMatrix(X_valid,label=y_valid)

    

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    

    xgb_mdl[j]=xgb.train(

            xgb_params, 

            d_train, 

            1600, watchlist, 

            early_stopping_rounds=70, 

            feval=logloss_xgb, 

            maximize=False, 

            verbose_eval=100)

    preds[:,j] = xgb_mdl[j].predict(d_test)
import matplotlib.pyplot as plt

y = trainDF['is_iceberg'].values

pre = xgb_mdl[j].predict(xgb.DMatrix(X))

plt.scatter(pre, y)

plt.show()
y_pred = np.mean(preds,axis=1)

sub = pd.DataFrame()

sub['id'] = testDF['id']

sub['is_iceberg'] = y_pred

sub.to_csv(subName, index=False)
