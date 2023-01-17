# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## import packages and modules
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
import math
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import time
# show plots inline
%matplotlib inline
## dataset path
filename = "../input/kobe-neural-network/processed.csv"
## set default figure size
figure_size = (15,10)
# set max display row number
pd.set_option('max_rows',5)
## load data
df = pd.read_csv(filename, index_col = 'shot_id')
print(df.head(3))
# split data
mask = df['shot_made_flag'].isna()
predict_df = df[mask]
train_df = df[~mask]
# matrix
Y = train_df['shot_made_flag'].as_matrix()
Y = Y.reshape(Y.shape[0],1) 
X = train_df.drop(['shot_made_flag'], axis=1).as_matrix()
predict_Y = predict_df['shot_made_flag'].as_matrix()
predict_Y = predict_Y.reshape(predict_Y.shape[0],1)
predict_X = predict_df.drop(['shot_made_flag'], axis=1).as_matrix()
# k-fold cross validation
k = 5
kfold = KFold(k, True, 1)
def buildmodel(train_X, train_Y, hidden_list, epoch_num = 150, batch_size = 32):
    N,D = train_X.shape
    d = train_Y.shape[1]
    # create model
    model = Sequential()
    # add layers
    model.add(Dense(hidden_list[0], input_dim=D, activation='relu'))
    for i in hidden_list[1:]:
        model.add(Dense(i, activation='relu'))
    model.add(Dense(d, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(train_X, train_Y, epochs=epoch_num, batch_size=batch_size)
    return model
models = []
hidden_list = [25,12]
for train_index, test_index in kfold.split(X):
    train_X = X[train_index]
    train_Y = Y[train_index]
    test_X = X[test_index]
    test_Y = Y[test_index]
    # fit model
    model = buildmodel(train_X, train_Y, hidden_list)
    models.append(model)
    # evaluate model
    prediction = model.predict(test_X)
    rounded = [round(x[0]) for x in prediction]
    accuracy = sum([int(rounded[i])==int(test_Y[i][0]) for i in range(len(prediction))])/len(prediction)
    print("Test accuracy:%.2f%%" % (accuracy*100))
# predict
predictions = []
for model in models:
    predictions.append(model.predict(predict_X))
predictions = np.array(predictions)
predictions = np.mean(predictions,axis=0)
# submit
submission = pd.DataFrame({"shot_id":df[mask].index, "shot_made_flag":predictions.reshape(predictions.shape[0]).tolist()})
submission.sort_values('shot_id',inplace=True)
submission.to_csv("submission.csv",index=False)