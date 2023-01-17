# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.shape, test.shape
train.head()
test.head()
# Checking null values in training dataset

train.isnull().sum().sum()
# Checking null values in test dataset

test.isnull().sum().sum()
train.describe()
test.describe()
# Count the target

sns.countplot(train['target'])
# Check how many positive and negative number are there in training data

train_pos_count=[]

train_neg_count=[]

for i in range(200):

    pos=0

    neg=0

    for j in train[str(i)]:

        if j>0:

            pos+=1

        else:

            neg+=1

    train_pos_count.append(pos)

    train_neg_count.append(neg)
# Plotting the distplot of positive train count

myarray = np.asarray(train_pos_count)

sns.distplot(myarray)
# Plotting the distplot of negative test count

myarray = np.asarray(train_neg_count)

sns.distplot(myarray)
# Check how many positive and negative number are there in test data



test_pos_count=[]

test_neg_count=[]

for i in range(200):

    pos=0

    neg=0

    for j in test[str(i)]:

        if j>0:

            pos+=1

        else:

            neg+=1

    test_pos_count.append(pos)

    test_neg_count.append(neg)
# Plotting the distplot of positive test count

myarray = np.asarray(test_pos_count)

sns.distplot(myarray)
# Plotting the distplot of negative test count

myarray = np.asarray(test_neg_count)

sns.distplot(myarray)
# Preparing the data

X=train.drop(['id','target'],axis=1)

y=train['target']
# KMeans Clustering 

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

train_labels=kmeans.labels_
# Labels Count in cluster of training data 

one=0

zero=0

for i in train_labels:

    if i==1:

        one+=1

    else:

        zero+=1

print("1: ",one)

print("0: ",zero)
y_train.value_counts()
X_test=test.drop(['id'],axis=1)
kmeans_test = KMeans(n_clusters=2, random_state=0).fit(X_test)

test_labels=kmeans_test.labels_
# Labels Count in cluster of test data

test_one=0

test_zero=0

for i in test_labels:

    if i==1:

        test_one+=1

    else:

        test_zero+=1

print("1: ",test_one)

print("0: ",test_zero)
# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

train_pca_result = pca.fit_transform(X)

test_pca_result = pca.fit_transform(X_test)
%matplotlib notebook

fig = plt.figure()

#plt.figure(figsize=(20,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_pca_result[:,0], train_pca_result[:,1], train_pca_result[:,2], marker='o')

plt.show()
%matplotlib notebook

fig = plt.figure()

# plt.figure(figsize=(20,10))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(test_pca_result[:,0], test_pca_result[:,1], test_pca_result[:,2], marker='o')

plt.show()
# Splitting the dataset in training and validation

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_ = scaler.fit_transform(X_train)

val_= scaler.fit_transform(X_val)

test_ = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

Model_rf=RandomForestClassifier(max_depth=2)

Model_rf.fit(train_,y_train)

y_pred=Model_rf.predict(val_)

#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,y_val))
from sklearn.ensemble import AdaBoostClassifier

Model_ada=AdaBoostClassifier()

Model_ada.fit(train_,y_train)

y_pred=Model_ada.predict(val_)

#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,y_val))
from sklearn.ensemble import GradientBoostingClassifier

Model_gb=GradientBoostingClassifier()

Model_gb.fit(train_,y_train)

y_pred=Model_gb.predict(val_)

#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,y_val))


from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

best_score = 0

for penalty in ['l1', 'l2']:

    for C in [0.001, 0.01, 0.1, 1, 10, 100]:       

        logreg = LogisticRegression(class_weight='balanced',  penalty=penalty, C=C, solver='liblinear')

        logreg.fit(train_, y_train)

        score = logreg.score(val_, y_val)

        if score > best_score:

            best_score = score

            best_parameters = {'C': C, 'penalty': penalty}       
logreg = LogisticRegression(**best_parameters)

logreg.fit(train_, y_train)

test_score = logreg.score(val_, y_val)

test_score
# from sklearn.linear_model import LogisticRegression

# logreg = LogisticRegression(class_weight='balanced', solver='liblinear', penalty ='l1', C= 0.1, max_iter=10000)

# logreg.fit(train_, y_train)

# test_score = logreg.score(val_, y_val)

# test_score
# Logistic Regression model

y_pred_final=logreg.predict_proba(test_)[:,1]
submission = pd.DataFrame({"id": test["id"],"target": y_pred_final})

submission.to_csv('submission.csv', index=False)