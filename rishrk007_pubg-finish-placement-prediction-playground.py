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
train=pd.read_csv("../input/train_V2.csv")
test=pd.read_csv("../input/test_V2.csv")
train.shape
train.isnull().count()
y_train=train["winPlacePerc"]
y_train.fillna(0,inplace=True)
train=train.drop(columns=['Id','groupId','matchId','numGroups','winPlacePerc'])
test=test.drop(columns=['Id','groupId','matchId','numGroups'])
train.head(10)
import seaborn as sns

#sns.heatmap(train.corr())

corr=train.corr()

corr.style.background_gradient()
train.shape
test.shape
y_train.shape
train=pd.get_dummies(train) #dummies encoding for dtype objects in training data
test=pd.get_dummies(test) #dummies encoding for dtype objects in testing data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train=sc.fit_transform(train) #to bring every feature value in same range
test=sc.fit_transform(test)
train.shape
test.shape
from keras import models

from keras import layers

from keras import optimizers

from keras.layers import Dropout
model=models.Sequential()
model.add(layers.Dense(128,activation='relu',input_shape=(train.shape[1],)))

model.add(Dropout(0.2))
model.add(layers.Dense(256,activation='relu'))

model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))

model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))

model.add(Dropout(0.2))
model.add(layers.Dense(128,activation='relu'))

model.add(Dropout(0.2))
model.add(layers.Dense(128,activation='relu'))

model.add(Dropout(0.2))
model.add(layers.Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(np.array(train),np.array(y_train),epochs=16,batch_size=512)
pred=model.predict(test)
sub=pd.read_csv("../input/sample_submission_V2.csv")
sub["winPlacePerc"]=pred
sub.head(10)
sub.to_csv("submission.csv",index=False)