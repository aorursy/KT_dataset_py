# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/hepatitis-1/hepatitis_1.csv", header=None)

train = train[train != "?"]

train.dropna(inplace=True)

#train = pd.get_dummies(train)

train = train.astype("float32")

train
import matplotlib.pyplot as plt

import seaborn as sns



matrix = train.corr()

sns.heatmap(matrix, annot=False)
survived = train.loc[:, 0] # 1=die 2=survived

age = train.loc[:, 1]

sex = train.loc[:, 2] # 1=male 2=female

anti = train.loc[:, 4] # 1=no 2=yes

bilirubin = train.loc[:, 14] # decimals

bilirubin = bilirubin.astype("float64")

hist = train.loc[:, 19] # 1=no 2=yes

ascites = train.loc[:, 12]

albumin = train.loc[:, 17]

#data = [survived, sex, steroid]

#df = pd.concat(data, axis=1)

#df

bilirubin.dtypes
data_train = [ascites, albumin]



X = pd.concat(data_train, axis=1)

X = pd.get_dummies(X)

y = train.loc[:, 0]

y = y.to_frame()

X
#from sklearn.model_selection import train_test_split

#from sklearn.naive_bayes import GaussianNB



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#gnb = GaussianNB()

#y_pred = gnb.fit(X_train, y_train).predict(X_test)

#y_pred
from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



model = Sequential()

model.add(Dense(12, input_dim=2, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)



predictions = model.predict_classes(X)

predictions
predictions = model.predict(X_test)

predictions