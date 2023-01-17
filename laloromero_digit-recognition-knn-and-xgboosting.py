import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Read training CSV 

df = pd.read_csv('../input/train.csv')

# Read testing CSV

df_test = pd.read_csv('../input/test.csv')

print(df.shape)

print(df_test.shape)
import matplotlib.pyplot as plt

# Show first image

samp_im1 = df.iloc[[0],1:].values.reshape((28,28))

plt.imshow(samp_im1)
# Convert data into two arrays, one representing input and other for output

from sklearn.model_selection import train_test_split

X = np.array([df.iloc[[i],1:].values.flatten() for i in range(df.shape[0])])

y = np.array([df.iloc[[i],0].values for i in range(df.shape[0])])

# Then split training data into trian set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=28000, random_state=42)
# Fit KNN model

from sklearn.neighbors import KNeighborsClassifier as knn

# By default knn uses Euclidean distance. 

# Also it's recommended to set an odd number for n_neighbors to avoid tie during classification

neigh = knn(n_neighbors=9)

neigh.fit(X_train, y_train)

# Get number of classes

neigh.classes_
from sklearn.metrics import confusion_matrix

# Check training with confusion matrix

y_pred = neigh.predict(X_test)

confusion_matrix(y_test, y_pred)

# Has good predictions, however still misscalssifies
# knn algorithm allows generating probabilities of class memebership

# this probabilities would be new features

y_pred_train = neigh.predict_proba(X_train)

y_pred_train
# Create XGBoost training set

# X_xgb_train = X_train

from sklearn import preprocessing

X_xgb_train = preprocessing.StandardScaler().fit_transform(X_train)

y_pred_train = preprocessing.StandardScaler().fit_transform(y_pred_train)

X_xgb_train = np.append(X_xgb_train, y_pred_train, axis=1)
# Create DMatrix to call XGBoost

import xgboost as xgb

dtrain = xgb.DMatrix(X_xgb_train, label=y_train)

num_round=350

# Small eta makes slower training, but better results

param = {'max_depth': 11, 'eta': 0.1, 'verbosity': 1, 'subsamples': 0.63, 

         'objective': 'multi:softmax', 'num_class': 10

        }

param['nthread'] = 4

param['eval_metric'] = ['cox-nloglik','rmse']

bst = xgb.train(param, dtrain, num_round)
# Generate new features from the probability of being in a class and use XGBoost to classify

y_pred_test = neigh.predict_proba(X_test)

# Standarize data

X_xgb_test = preprocessing.StandardScaler().fit_transform(X_test)

y_pred_test = preprocessing.StandardScaler().fit_transform(y_pred_test)

X_xgb_test = np.append(X_xgb_test, y_pred_test, axis=1)

# Create DMatrix for XGBoost

dtest = xgb.DMatrix(X_xgb_test)

y_xgb_pred = bst.predict(dtest, ntree_limit=num_round)
# Validate classification with confusion matrix

y_xgb_pred = np.round(np.abs(y_xgb_pred))

confusion_matrix(y_test, y_xgb_pred)
# Generate CSV file

X_test = np.array([df_test.iloc[[i]].values.flatten() for i in range(df_test.shape[0])])



y_pred_test = neigh.predict_proba(X_test)

X_xgb_test = preprocessing.StandardScaler().fit_transform(X_test)

y_pred_test = preprocessing.StandardScaler().fit_transform(y_pred_test)

X_xgb_test = np.append(X_test, y_pred_test, axis=1)



dtest = xgb.DMatrix(X_xgb_test)



y_xgb_pred = bst.predict(dtest)



y_xgb_pred = np.round(np.abs(y_xgb_pred))

y_xgb_pred.astype(np.int64)



Label = pd.Series(y_xgb_pred,name = 'Label')

ImageId = pd.Series(range(1,y_xgb_pred.shape[0]),name = 'ImageId')

submission = pd.concat([ImageId,Label],axis = 1)

submission.to_csv('submission.csv',index = False)