# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

df[df.target_class == 0].count()
y = df.corr()

sns.heatmap(y,annot=True,cmap="bone",linewidths=1,fmt=".2f",linecolor="indigo")
plt.figure(figsize=(16,10))

plt.subplot(2,2,1)

sns.barplot(data=df,y=" Mean of the integrated profile",x="target_class")

plt.subplot(2,2,2)

sns.barplot(data=df,y=" Mean of the DM-SNR curve",x="target_class")

plt.subplot(2,2,3)

sns.barplot(data=df,y=" Standard deviation of the integrated profile",x="target_class")

plt.subplot(2,2,4)

sns.barplot(data=df,y=" Standard deviation of the DM-SNR curve",x="target_class")

plt.show()
labels = df.target_class.values



df.drop(["target_class"],axis=1,inplace=True)



features = df.values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))



features_scaled = scaler.fit_transform(features)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(features_scaled,labels,test_size=0.2)
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=42,solver="liblinear",C=2,penalty="l1")



lr_model.fit(x_train,y_train)



y_head_lr = lr_model.predict(x_test)



lr_score = lr_model.score(x_test,y_test)



lr_score
from sklearn.tree import DecisionTreeClassifier

dc_model = DecisionTreeClassifier(random_state=20)



dc_model.fit(x_train,y_train)



y_head_dc = dc_model.predict(x_test)



dc_score = dc_model.score(x_test,y_test)



dc_score
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=7,weights="distance")



knn_model.fit(x_train,y_train)



y_head_knn = knn_model.predict(x_test)



knn_score = knn_model.score(x_test,y_test)



knn_score