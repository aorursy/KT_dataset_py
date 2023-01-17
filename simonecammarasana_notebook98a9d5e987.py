import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import tensorflow as tf
import os
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
train = pd.read_csv('../input/gene-expression/data_set_ALL_AML_train.csv')
test = pd.read_csv('../input/gene-expression/data_set_ALL_AML_independent.csv')
labels = pd.read_csv('../input/gene-expression/actual.csv',index_col='patient')
print(train)
print(labels)
#Clean the "call" column
train_keepers = [col for col in train.columns if "call" not in col]
test_keepers = [col for col in test.columns if "call" not in col]

train = train[train_keepers]
test = test[test_keepers]

#Transpose the data
train = train.T
test = test.T

# Clean up the column names for training data
train.columns = train.iloc[1]
train = train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)

# Clean up the column names for training data
test.columns = test.iloc[1]
test = test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)
print(train)
labels = labels.replace({'ALL':0,'AML':1})
labels_train = labels[labels.index <= 38]
labels_test = labels[labels.index > 38]
print(labels_train)
df_all = train.append(test, ignore_index=True)
X_all = preprocessing.StandardScaler().fit_transform(df_all)
X_train = X_all[:38,:] 
X_test = X_all[38:,:]
model = SVC()
model.fit(X_train, labels_train.values.ravel())
pred = model.predict(X_all[38:,:])
print('Accuracy: ', accuracy_score(labels_test, pred))