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


import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import sklearn

from keras.models import Sequential

from keras.layers import Dense, Activation

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

%matplotlib inline

from sklearn.datasets import *

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras import optimizers 





#필요한 라이브러리들을 import 하겠습니다.
#Loading dataset

wine = pd.read_csv('../input/winequality-red.csv')
#Let's check the form of data

wine.head()
#Information about the data columns

wine.info()


fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'fixed acidity', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'volatile acidity', data = wine)





fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'citric acid', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'residual sugar', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'chlorides', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'free sulfur dioxide', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'total sulfur dioxide', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'sulphates', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'alcohol', data = wine)



fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'pH', data = wine)





fig = plt.figure(figsize = (15,9))

sns.boxplot(x = 'quality', y = 'density', data = wine)

# Quality 와 요소별 그래프를 시각화함으로써, 어떤 요소가 상관관계가 있는지 파악해보았습니다.

# 그 결과, 모든 요소가 quality에 영향을 주는 것은 아니라는 걸 확인할 수 있었습니다.
#1 - Bad / 2 - Average / 3 - Excellent 으로 등급을 셋으로 나누어 모델을 만들어보겠습니다.

#quality = 2,3 --> Bad

#quality = 4,5,6 --> Average

#quality = 7,8 --> Excellent

#Create an empty list called Reviews



wine_reviews = []

for i in wine['quality']:

    if i >= 2 and i <= 4:

        wine_reviews.append('1')

    elif i >= 5 and i <= 6:

        wine_reviews.append('2')

    elif i >= 7 and i <= 8:

        wine_reviews.append('3')

wine['Grade'] = wine_reviews



wine['Grade'].value_counts()



sns.countplot(wine['Grade'])
"""

#Now seperate the dataset as response variable and feature variabes

X = wine.drop('quality', axis = 1)

X = X.drop('Grade', axis=1)

X = X.drop('fixed acidity', axis=1)

X = X.drop('residual sugar', axis=1)

X = X.drop('density', axis=1)

X = X.drop('pH', axis=1)

X = X.drop('alcohol', axis=1)

y = wine['Grade']



print(X.head)

wine=X

"""




X = wine.drop('quality',axis = 1)

X = X.drop('Grade',axis=1)

y = wine['Grade']

print(X.head)
#view final data

print(X.columns)

print(X.head(10))

print(y.head(10))

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

#view the scaled features

print(X)
from sklearn.decomposition import PCA

pca = PCA()

X_pca = pca.fit_transform(X)
#plot the graph to find the principal components

plt.figure(figsize=(10,10))

plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')

plt.grid()
#AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 

#we shall pick the first 8 components for our prediction.

pca_new = PCA(n_components=8)

X_new = pca_new.fit_transform(X)
print(X_new)

X_new = pd.DataFrame(X_new)

print(type(X_new))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2,random_state = 42)
print(x_train)

print(y_train)

print(x_test)

print(y_test)

print(type(x_train))
rfc = RandomForestClassifier(n_estimators=220)

rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)

#Let's see how our model performed

print(classification_report(y_test, pred_rfc))



#Confusion matrix for the random forest classification

print(confusion_matrix(y_test, pred_rfc))
print(x_train)

print(y_train)

print(x_test)

print(y_test)

print(type(x_train))

print(type(y_train))

print(type(x_test))

print(type(y_test))
from sklearn.utils import resample



# concatenate our training data back together

X = pd.concat([x_train, y_train], axis=1)

print(X.head(10))

print(X.Grade.head(10))



#X_new = pd.DataFrame(X_new)

#mask = df['A'] == 'foo'

#df.query('A == "foo"')

# separate minority and majority classes

Bad = X.query('Grade == "1"')



Average = X.query('Grade == "2"')

Excellent = X.query('Grade == "3"')



print(Bad.head(20))

print(Average.head(20))

print(Excellent.head(20))

# upsample minority

Bad_upsampled = resample(Bad,

                          replace=True, # sample with replacement

                          n_samples=int(len(Average)), # match number in majority class

                          random_state=42) # reproducible results



Excellent_upsampled = resample(Excellent,

                          replace=True, # sample with replacement

                          n_samples=int(len(Average)), # match number in majority class

                          random_state=42) # reproducible results





# combine majority and upsampled minority

upsampled = pd.concat([Average, Bad_upsampled,Excellent_upsampled])

upsampled = upsampled.sample(frac=1)

# check new class counts



print(type(upsampled))

upsampled.Grade.value_counts()

print(upsampled)

x_train = upsampled.drop('Grade',axis=1)

y_train = upsampled['Grade']
rfc = RandomForestClassifier(n_estimators=220)

rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)

#Let's see how our model performed

print(classification_report(y_test, pred_rfc))



#Confusion matrix for the random forest classification

print(confusion_matrix(y_test, pred_rfc))
Bad = X.query('Grade == "1"')



Average = X.query('Grade == "2"')

Excellent = X.query('Grade == "3"')

# downsample majority

Average_downsampled = resample(Average,

                          replace=False, # sample with replacement

                          n_samples=len(Bad), # match number in majority class

                          random_state=42) # reproducible results





# combine majority and upsampled minority

downsampled = pd.concat([Average_downsampled, Bad ,Excellent])

downsampled = downsampled.sample(frac=1)

# check new class counts



print(type(downsampled))

downsampled.Grade.value_counts()

print(downsampled)
x_train = downsampled.drop('Grade',axis=1)

y_train = downsampled['Grade']
rfc = RandomForestClassifier(n_estimators=220)

rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)

#Let's see how our model performed

print(classification_report(y_test, pred_rfc))



#Confusion matrix for the random forest classification

print(confusion_matrix(y_test, pred_rfc))
Bad = X.query('Grade == "1"')



Average = X.query('Grade == "2"')

Excellent = X.query('Grade == "3"')

# upsample minority

Bad_upsampled = resample(Bad,

                          replace=True, # sample with replacement

                          n_samples=(int(len(Average)*(1/1.65))), # match number in majority class

                          random_state=42) # reproducible results



Excellent_upsampled = resample(Excellent,

                          replace=True, # sample with replacement

                          n_samples=(int(len(Average)*(1/1.65))), # match number in majority class

                          random_state=42) # reproducible results

# downsample majority

Average_downsampled = resample(Average,

                          replace=False, # sample with replacement

                          n_samples=(int(len(Average)*(1/1.65))), # match number in majority class

                          random_state=42) # reproducible results

mixsampled = pd.concat([Bad_upsampled,Average_downsampled,Excellent_upsampled])

mixsampled = mixsampled.sample(frac=1)
x_train = mixsampled.drop('Grade',axis=1)

y_train = mixsampled['Grade']
rfc = RandomForestClassifier(n_estimators=220)

rfc.fit(x_train, y_train)

pred_rfc = rfc.predict(x_test)



#Let's see how our model performed

print(classification_report(y_test, pred_rfc))



#Confusion matrix for the random forest classification

print(confusion_matrix(y_test, pred_rfc))