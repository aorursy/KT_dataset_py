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
train = pd.read_csv('../input/heart.csv')

train.head()
train.describe()
train.columns
train.info()
df_female = [rows for _, rows in train.groupby('sex')][0]

df_male= [rows for _, rows in train.groupby('sex')][1]
df_female=df_female.drop(['sex'], axis=1)

df_male=df_male.drop(['sex'], axis=1)
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.ticker as ticker

import matplotlib.cm as cm

import matplotlib as mpl

from matplotlib.gridspec import GridSpec

plt.figure(1, figsize=(20,10))

the_grid = GridSpec(2, 3)

plt.subplot(the_grid[0, 0], aspect=1, title='Female v/s Chest Pain Type')

df_female.age.groupby(df_female.cp).sum().plot(kind='pie',autopct='%.2f')



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s Chest Pain Type')

df_male.age.groupby(df_male.cp).sum().plot(kind='pie',autopct='%.2f')



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details Chest Pain Type')

train.age.groupby(train.cp).sum().plot(kind='pie',autopct='%.2f')



plt.suptitle('Pie Chart for Chest Pain Type', fontsize=16)

g=sns.jointplot(x="age", y="trestbps", data=train, color="r")

plt.subplots_adjust(top=.9)

g.fig.suptitle('Age v/s resting blood pressure (in mm Hg on admission to the hospital)') 

plt.show()
g=sns.jointplot(x="age", y="chol", data=train, color="g")

plt.subplots_adjust(top=.9)

g.fig.suptitle('Age v/s Serum cholestoral in mg/dl') 

plt.show()
plt.figure(1, figsize=(20,10))



plt.subplot(the_grid[0, 0], aspect=1 ,title='Female v/s fasting blood sugar')

df_female.age.groupby(df_female.fbs).sum().plot(kind='pie',autopct='%.2f',labels=['<120','>120'],textprops={'fontsize': 15})



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s fasting blood sugar')

df_male.age.groupby(df_male.fbs).sum().plot(kind='pie',autopct='%.2f',labels=['<120','>120'],textprops={'fontsize': 15})



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details (fasting blood sugar)')

train.age.groupby(train.fbs).sum().plot(kind='pie',autopct='%.2f',labels=['<120','>120'],textprops={'fontsize': 15})



plt.suptitle('Pie Chart for Fasting Blood Sugar', fontsize=16)
plt.figure(1, figsize=(20,10))



plt.subplot(the_grid[0, 0], aspect=1 ,title='Female v/s resting electrocardiographic results')

df_female.age.groupby(df_female.restecg).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s resting electrocardiographic results')

df_male.age.groupby(df_male.restecg).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details (resting electrocardiographic results)')

train.age.groupby(train.restecg).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.suptitle('Pie Chart for Resting Electrocardiographic Results', fontsize=16)
g=sns.jointplot(x="age", y="thalach", data=train, color="b")

plt.subplots_adjust(top=.9)

g.fig.suptitle('Age v/s Maximum Heart Rate Achieved') 

plt.show()
plt.figure(1, figsize=(20,10))



plt.subplot(the_grid[0, 0], aspect=1 ,title='Female v/s exercise induced angina')

df_female.age.groupby(df_female.exang).sum().plot(kind='pie',autopct='%.2f',labels=['No','Male'],textprops={'fontsize': 15})



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s exercise induced angina')

df_male.age.groupby(df_male.exang).sum().plot(kind='pie',autopct='%.2f',labels=['No','Male'],textprops={'fontsize': 15})



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details (exercise induced angina)')

train.age.groupby(train.exang).sum().plot(kind='pie',autopct='%.2f',labels=['No','Male'],textprops={'fontsize': 15})



plt.suptitle('Pie Chart for Exercise Induced Angina', fontsize=16)
plt.figure(1, figsize=(20,20))

g=sns.jointplot(x="age", y="oldpeak", data=train, color="b")

plt.subplots_adjust(top=.9)

g.fig.suptitle('Age v/s ST Depression Induced by Exercise Relative to Rest') 

plt.show()
plt.figure(1, figsize=(20,10))



plt.subplot(the_grid[0, 0], aspect=1 ,title='Female v/s Slope of the peak exercise ST segment')

df_female.age.groupby(df_female.slope).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s Slope of the peak exercise ST segment')

df_male.age.groupby(df_male.slope).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details (the slope of the peak exercise ST segment)')

train.age.groupby(train.slope).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.suptitle('Pie Chart for The Slope of the Peak Exercise ST Segment', fontsize=16)
plt.figure(1, figsize=(20,10))



plt.subplot(the_grid[0, 0], aspect=1 ,title='Female v/s Number of Major Vessels (0-3) Colored by Flourosopy')

df_female.age.groupby(df_female.ca).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s Number of Major Vessels (0-3) Colored by Flourosopy')

df_male.age.groupby(df_male.ca).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details (Number of Major Vessels (0-3) Colored by Flourosopy)')

train.age.groupby(train.ca).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.suptitle('Pie Chart for Number of Major Vessels (0-3) Colored by Flourosopy', fontsize=16)
plt.figure(1, figsize=(20,10))



plt.subplot(the_grid[0, 0], aspect=1 ,title='Female v/s Thal')

df_female.age.groupby(df_female.thal).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s Thal')

df_male.age.groupby(df_male.thal).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details (Thal)')

train.age.groupby(train.thal).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.suptitle('Pie Chart for Thal', fontsize=16)
plt.figure(1, figsize=(20,10))



plt.subplot(the_grid[0, 0], aspect=1 ,title='Female v/s Target')

df_female.age.groupby(df_female.target).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 1], aspect=1, title='Male v/s Target')

df_male.age.groupby(df_male.target).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.subplot(the_grid[0, 2], aspect=1, title='Overall details (Target)')

train.age.groupby(train.target).sum().plot(kind='pie',autopct='%.2f',textprops={'fontsize': 15})



plt.suptitle('Pie Chart for Target', fontsize=16)
%matplotlib inline

plt.rcParams['figure.figsize']=20,20

sns.pairplot(train)
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
X=train.iloc[:,:-1]

y=train.iloc[:,-1]
from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test) 

from sklearn import metrics 

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100,"%")
from sklearn.svm import SVC #

svcclf = SVC(kernel='linear')

svcclf.fit(X, y)

y_pred1 = svcclf.predict(X_test)

print("Support Vector Classifier model accuracy(in %):", metrics.accuracy_score(y_test, y_pred1)*100,"%")