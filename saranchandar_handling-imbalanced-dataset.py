import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



file = pd.read_csv("../input/ecoli.csv")

print(file.head())
file.isnull().values.any()
file['Class'].describe()
f = file.groupby("Class")

f.count()
file['Class'] = file['Class'].map({'positive': 1, 'negative': 0})

print(file['Class'].head())
sns.pairplot(file,hue='Class')
file['Class'].hist()
from sklearn.cross_validation import train_test_split

train, test = train_test_split(file,test_size=0.2)

features_train=train[['Mcg','Gvh','Lip','Chg','Aac','Alm1','Alm2']]

features_test = test[['Mcg','Gvh','Lip','Chg','Aac','Alm1','Alm2']]

labels_train = train.Class

labels_test = test.Class

print(train.shape)

print(test.shape)

print(features_train.shape)

print(features_test.shape)

print(labels_train.shape)

print(labels_test.shape)

print(labels_test.head())
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

model = clf.fit(features_train, labels_train)

feature_labels = ['Mcg','Gvh','Lip','Chg','Aac','Alm1','Alm2']

for feature in zip(feature_labels,model.feature_importances_):

    print(feature)
new_file = file[['Mcg','Gvh','Aac','Alm1','Alm2','Class']]

new_file.head()
new_train, new_test = train_test_split(new_file,test_size=0.2)

new_features_train=new_train[['Mcg','Gvh','Aac','Alm1','Alm2']]

new_features_test = new_test[['Mcg','Gvh','Aac','Alm1','Alm2']]

labels_train = new_train.Class

labels_test = new_test.Class

print(train.shape)

print(test.shape)

print(new_features_train.shape)

print(new_features_test.shape)

print(labels_train.shape)

print(labels_test.shape)

print(labels_test.head())
clf = RandomForestClassifier()

model = clf.fit(new_features_train, labels_train)

print("Accuracy of Randomforest Classifier:",clf.score(new_features_test,labels_test))
from collections import Counter

from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(new_features_train, labels_train)

print("before sampling:",format(Counter(labels_train)))

print("after sampling:",format(Counter(y_resampled)))
from sklearn.linear_model import LogisticRegression

clf1 = LogisticRegression()

clf1.fit(X_resampled, y_resampled)

print('Accuracy:',clf1.score(new_features_test,labels_test))
