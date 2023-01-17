# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

train_df.head(5)
train_df.isnull().sum()
train_df.isna().sum()
corr = train_df.corr()

corr.values
plt.figure(figsize=(17,7))

sns.heatmap(corr)
plt.figure(figsize=(17,7))

plt.subplot(2,2,1)

sns.boxplot(train_df['Outcome'],train_df['BloodPressure'])

plt.subplot(2,2,2)

sns.boxplot(train_df['Outcome'],train_df['Glucose'])
plt.figure(figsize=(17,7))

sns.boxplot(train_df['Outcome'],train_df['Age'])
X = train_df.drop(['Outcome'], axis =1).values

y = train_df['Outcome'].values
X[0:5]
y[0:5]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)

print('training data : ',X_train.shape, y_train.shape)

print('testing data : ',X_test.shape, y_test.shape)
dtc = DecisionTreeClassifier()

dtc = dtc.fit(X_train, y_train)

dtc
y_pred = dtc.predict(X_test)
acc = accuracy_score(y_pred, y_test)

print('Accuracy is : ',acc)
from sklearn.decomposition import PCA



pca = PCA(3)

X_pca = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)
y_preds = rfc.predict(X_test)

acc_rfc = accuracy_score(y_preds,y_test)

acc_rfc
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler().fit(X)
X_transform = scaler.transform(X)
classifier =Sequential()



# create input layer

classifier.add(Dense(units=6,kernel_initializer='uniform' , activation='tanh' , input_dim=8))



# create hidden layer

classifier.add(Dense(units=6,kernel_initializer='uniform' , activation='tanh'))

# create hidden layer



# create output layer

classifier.add(Dense(units=1 , kernel_initializer='uniform' , activation='sigmoid'))



#compiling the ANN

classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
y_pred = classifier.predict(X_test)
print(accuracy_score(y_preds,y_test))