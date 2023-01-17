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
import tensorflow.keras

import tensorflow as tf

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/heart.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
conditions=[(df.chol<200),

            (df.chol<239),

            (df.chol<1000)]

choices=[0,1,2]



df.chol=np.select(conditions, choices)

df.chol.tail(5)
df.dtypes
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
sns.countplot(df.target,hue=df.sex)

plt.title("Count of Diseases and not")

plt.xlabel('target')

plt.ylabel('count')
cols=['age','ca','cp','exang','fbs','oldpeak','restecg','sex','slope','thal','thalach','trestbps','chol']

x=df[cols]

y=df['target']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(units=8,activation='relu',input_shape=(13,)))
classifier.add(Dense(units=4,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=20,nb_epoch=20,validation_split=0.2)
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test, y_pred))

print('\n')

print(classification_report(y_test, y_pred))