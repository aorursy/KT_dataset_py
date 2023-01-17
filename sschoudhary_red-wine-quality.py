# importing libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

wine = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

wine.head()
wine.info()

# viewing information
wine.describe()

# hence no negative values
wine.isnull().sum()

# to check the null values, hence no null values
# to check the correlation

plt.figure(figsize=(8,6))

cor = wine.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
cor.nlargest(10,'quality')['quality']

# as shown below--> alcohol, sulphates, citric acid are highly correlated with target variable
wine.quality.value_counts().sort_index()
# Analysis of alcohol with wine quality:

bx = sns.boxplot(x="quality", y='alcohol', data = wine)
# Analysis of sulphates with wine quality:

bx = sns.boxplot(x="quality", y='sulphates', data = wine)
# Analysis of citric acid with wine quality:

bx = sns.boxplot(x="quality", y='citric acid', data = wine)
# creating 3 groups for quality 

#1,2,3 --> Bad

#4,5,6,7 --> Average

#8,9,10 --> Good



# another way to create 

# rating_bin=[0,4,6,10]

# group_labels=['Bad','Average','good']

# wine['group']=pd.cut(wine.quality,group_bin,3, labels=group_labels)

rating=[]

for i in wine['quality']:

    if i>=1 and i<=3:

        rating.append('1')

    if i>=4 and i<=7:

        rating.append('2')

    if i>=8 and i<=10:

        rating.append('3')

wine['rating']=rating
wine.head()

# as we created quality column above, hence added
sns.countplot(x='rating', data=wine)
from collections import Counter

Counter(wine['quality'])

# checking the total number for each quality
Counter(wine['rating'])

# checking the total number for rating, 2 has highest rating which means 'Average'
# y = wine['quality']

# x = wine.drop('quality',axis = 1)

# dropping 

x = wine.iloc[:,:11]

y = wine['rating']
x.head()
y.head()
# splitting the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# checking the accuracy by importing algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score

import warnings

warnings.filterwarnings("ignore")

lr = LogisticRegression()

lr.fit(x_train, y_train)

lr_predict = lr.predict(x_test)

lr_conf_matrix = confusion_matrix(y_test, lr_predict)

lr_acc_score = accuracy_score(y_test, lr_predict)

print(lr_conf_matrix)

print(lr_acc_score)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

knnpred = knn.predict(x_test)

print(confusion_matrix(y_test,knnpred))

print(classification_report(y_test,knnpred))

print(metrics.accuracy_score(y_test,knnpred))
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=250)

rf.fit(x_train,y_train)

Rfpred=rf.predict(x_test)

print(classification_report(y_test,Rfpred,digits=4))

print(confusion_matrix(y_test, Rfpred))

print(metrics.accuracy_score(y_test,Rfpred))

print(rf.feature_importances_)
#SVM

from sklearn.svm import SVC

model_SVC=SVC()

model_SVC.fit(x_train, y_train)

predictions=model_SVC.predict(x_test)

#Classification_report & Confusion_matrix to test the model is good or not

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test,predictions,digits=4))

print(metrics.accuracy_score(y_test,predictions))