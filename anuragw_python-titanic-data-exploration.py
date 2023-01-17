from sklearn.model_selection import train_test_split

features_train,features_test=train_test_split(features_train)

feature_train
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv ("../input/test.csv")

full = train.append(test)
# Mapping sex to 1s and 0s

sex = pd.Series(np.where(full.Sex =="male",1,0),name= 'Sex')
# cleaning age col

age = full.Age.fillna(full.Age.mean())
#Extract Cabin information

cabin = full.Cabin.fillna('U')

cabin = cabin.map(lambda x:x[0])

cabin = pd.get_dummies(cabin,prefix='Cabin')
#Extract Embarked information

embarked= pd.get_dummies(full.Embarked,prefix='Embarked')
# Clean Fare information

fare= full.Fare.fillna(full.Fare.mean())
#Combine preprocessed features

features = pd.DataFrame()

labels = pd.DataFrame()

features['Age']=age

features['Sex']=sex

features = pd.concat([features,cabin,embarked],axis=1)

labels['labels']= full.Survived.fillna(0)

# Train on SVM on complete data



features_train = features[:891].as_matrix()

features_test = features[891:].as_matrix()

labels_train = labels[:891].astype(int).as_matrix()

features_train.shape

from sklearn.model_selection import train_test_split

features_train_X, features_test_X, labels_train_X, labels_test_X= train_test_split(features_train,labels_train)
from sklearn.decomposition import PCA

pca = PCA(n_components=10,svd_solver="randomized")

pca.fit(features_train_X,labels_train_X)

features_train_transformed = pca.transform(features_train_X)

pca.fit(features_test_X,labels_test_X)

features_test_transformed = pca.transform(features_test_X)
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

scalar.fit(features_train_transformed,labels_train_X)

features_train_scaled = scalar.transform(features_train_transformed)

features_test_scaled = scalar.fit_transform(features_test_transformed,labels_test_X)
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()

clf.fit(features_train_scaled,labels_train_X.ravel())

accuracy = clf.score(features_test_scaled,labels_test_X.ravel())

predicted = clf.predict(features_test_scaled)

from sklearn.metrics import *

print ("Classification report:\n ",classification_report(predicted,labels_test_X.ravel()))

print ("Precision Score :", precision_score(predicted,labels_test_X.ravel()))

print ("Recall Score :",recall_score(predicted,labels_test_X.ravel()))

print ("Accuracy :",accuracy )
# TODO 

# Write better explanation

# Add more todo 