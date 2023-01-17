# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import gc

import cv2

from PIL import Image

import matplotlib.pyplot as plt



###########################################################################



from tqdm import tqdm



#########################################################################



import lightgbm  as lgb

import catboost as cbt

import xgboost as xgb

import sklearn.ensemble as ensem

import sklearn.linear_model as lm

import sklearn.svm as svm 

import sklearn.neighbors as neibs

import sklearn.naive_bayes  as nb

import sklearn.discriminant_analysis as danl 



##########################################################



import sklearn.metrics as metrics

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator



##########################################################



import sklearn.decomposition as decomp

import sklearn.manifold as mnfld
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

imgsize = 14

imgsize2 = 196

images = []

paths = train.id_code

for path in paths :

    img = cv2.imread(f'../input/train_images/{path}.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img,(imgsize,imgsize),interpolation=cv2.INTER_CUBIC)

    img = (img.reshape((imgsize2,))/255).tolist()

    images.append(img)
trainxdf =  pd.DataFrame(images)
#kernel : "linear" | "poly" | "rbf" | "sigmoid"

#kernel  | "cosine" | "precomputed"

kpcatrain = decomp.KernelPCA(n_components=8,

                            kernel="poly",

                            gamma=None, 

                            degree=3, 

                            coef0=1,

                            kernel_params=None,

                            alpha=0.1, 

                            fit_inverse_transform=False, 

                            eigen_solver='auto',

                            tol=0,

                            max_iter=None,

                            remove_zero_eig=False,

                            random_state=0

                          )



colskpca = ["kpca1","kpca2","kpca3","kpca4","kpca5",

           "kpca6","kpca7","kpca8"]

kpcatraindf =  pd.DataFrame(kpcatrain.fit_transform(trainxdf),

                            columns=colskpca)



trainxdf =  pd.concat([trainxdf,kpcatraindf], axis=1)
rf = ensem.RandomForestClassifier(n_estimators=2000,

                                  max_depth=200,

                                  random_state=100)

rf.fit(trainxdf,train.diagnosis)
imgsize = 14

imgsize2 = 196

imagest = []

paths = test.id_code

for path in paths :

    img = cv2.imread(f'../input/test_images/{path}.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img,(imgsize,imgsize),interpolation=cv2.INTER_CUBIC)

    img = (img.reshape((imgsize2,))/255).tolist()

    imagest.append(img)
testxdf =  pd.DataFrame(imagest)
########################################################################



kpcatest = decomp.KernelPCA(n_components=8,

                            kernel="poly",

                            gamma=None, 

                            degree=3, 

                            coef0=1,

                            kernel_params=None,

                            alpha=0.1, 

                            fit_inverse_transform=False, 

                            eigen_solver='auto',

                            tol=0,

                            max_iter=None,

                            remove_zero_eig=False,

                            random_state=0

                          )



colskpca = ["kpca1","kpca2","kpca3","kpca4","kpca5",

           "kpca6","kpca7","kpca8"]

kpcatestdf =  pd.DataFrame(kpcatest.fit_transform(testxdf),

                           columns=colskpca)

testxdf =  pd.concat([testxdf,kpcatestdf], axis=1)
predVal = rf.predict(testxdf)
test['diagnosis'] = predVal

test.to_csv("submission.csv",index=False, header=True)