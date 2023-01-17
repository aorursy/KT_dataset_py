# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# basic function of python

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sb

%matplotlib inline

import numpy as np

from pandas import ExcelWriter

from pandas import ExcelFile

import xlrd

from scipy import stats

from datetime import datetime



# feature hashing

from sklearn.feature_extraction import FeatureHasher



# target encoder

import category_encoders as ce



# feature selection

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import feature_selection



# oversampling

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



# building the models

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import tensorflow

# from tensorflow.contrib.keras import models, layers

# from tensorflow.contrib.keras import activations, optimizers, losses



# standardize the vaiable

from sklearn.preprocessing import StandardScaler



# from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split



# validation

from sklearn.metrics import confusion_matrix,classification_report
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')

submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')
train.head()
test.head()
submission.head()
train.drop(['id'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)
train['target'].value_counts()
train.dtypes
train.shape
test.shape
train.head()
test.head()
train.isnull() # Checking missing values
train.isnull().sum() # check the missing values
sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
missing_data = train.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")    
test.isnull() # Checking missing values
test.isnull().sum() # check the missing values
sb.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
missing_data = test.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")    
train.head()
train.dtypes
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=8, input_type='string')

sp = fh.fit_transform(train['ord_5'])

df = pd.DataFrame(sp.toarray(), columns=['fh1', 'fh2', 'fh3', 'fh4', 'fh5', 'fh6', 'fh7', 'fh8'])

pd.concat([train, df], axis=1)

train.drop('ord_5',axis=1,inplace=True)

train
from sklearn.feature_extraction import FeatureHasher

fh = FeatureHasher(n_features=8, input_type='string')

sp = fh.fit_transform(test['ord_5'])

df = pd.DataFrame(sp.toarray(), columns=['fh1', 'fh2', 'fh3', 'fh4', 'fh5', 'fh6', 'fh7', 'fh8'])

pd.concat([test, df], axis=1)

test.drop('ord_5',axis=1,inplace=True)

test
train = pd.get_dummies(train, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4','ord_3', 'ord_4'],drop_first=True, sparse=True)
train.shape
test = pd.get_dummies(test, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4','ord_3', 'ord_4'],drop_first=True, sparse=True)
test.shape
cols_ = ['nom_5','nom_6','nom_7','nom_8','nom_9']

ce_target_encoder = ce.TargetEncoder(cols = cols_, smoothing=0.50)

ce_target_encoder.fit(train[cols_], train['target'])

train_nom = ce_target_encoder.transform(train[cols_])

train.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)

train = pd.concat([train, train_nom], axis=1)

train
ce_target_encoder = ce.TargetEncoder(cols = ['nom_5','nom_6','nom_7','nom_8','nom_9'], smoothing=0.50)

cols = ['nom_5','nom_6','nom_7','nom_8','nom_9']

ce_target_encoder.fit(train[cols], train['target'])

#train = oof.sort_index() 

test_nom = ce_target_encoder.transform(test[cols])

test_nom
test.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)

test = pd.concat([test, test_nom], axis=1)

test
train.head()
# Category variables -> Numerical variables

list_feat=['bin_3','bin_4','ord_1','ord_2']
for feature in list_feat:

    labels = train[feature].astype('category').cat.categories.tolist()

    replace_map_comp = {feature : {k: v for k,v in zip(labels,list(range(0,len(labels)+1)))}}



    train.replace(replace_map_comp, inplace=True)
list_feat=['bin_3','bin_4','ord_1','ord_2']
for feature in list_feat:

    labels = test[feature].astype('category').cat.categories.tolist()

    replace_map_comp = {feature : {k: v for k,v in zip(labels,list(range(0,len(labels)+1)))}}



    test.replace(replace_map_comp, inplace=True)
# Day

train['day_sin'] = np.sin(2 * np.pi * train['day']/7)

train['day_cos'] = np.cos(2 * np.pi * train['day']/7)

# Month

train['month_sin'] = np.sin(2 * np.pi * train['month']/12)

train['month_cos'] = np.cos(2 * np.pi * train['month']/12)
# Day

test['day_sin'] = np.sin(2 * np.pi * test['day']/7)

test['day_cos'] = np.cos(2 * np.pi * test['day']/7)

# Month

test['month_sin'] = np.sin(2 * np.pi * test['month']/12)

test['month_cos'] = np.cos(2 * np.pi * test['month']/12)
train.head()
test.head()
train.drop(['day','month'],axis=1,inplace=True)
test.drop(['day','month'],axis=1,inplace=True)
train.head()
# train_target = train['target']

# train_target

# train.drop('target',axis=1,inplace=True)

# train = pd.concat([train, train_target], axis=1)

# train
test.head()
# checking the imbalance

sb.countplot(x='target',data=train,palette='RdBu_r') # Barplot for the dependent variable
train['target'].value_counts()
# # Separate the majority of data and the minority of data

df_majority = train[train['target']==0]

df_minority = train[train['target']==1]
# oversampling minority data

df_minority_upsampled = resample(df_minority, 

                                 replace=True,     # replace the original data

                                 n_samples=208236,    # the number of data to match with majority

                                 random_state=123) # reproducible results
# Combine majority class with upsampled minority class

df_upsampled = pd.concat([df_majority, df_minority_upsampled])
sb.countplot(x='target',data=df_upsampled,palette='RdBu_r')
# Display new class counts

df_upsampled['target'].value_counts()
# dataset=df_upsampled._get_values
# Separate input features (X) and target variable (y)

y = df_upsampled.target

X = df_upsampled.drop('target', axis=1)
train.shape
# # Build a forest and compute the feature importances

# model1 = ExtraTreesClassifier(n_estimators=250,

#                               random_state=0)



# model1.fit(dataset_train,dataset_label)

# importances = model1.feature_importances_

# std = np.std([tree.feature_importances_ for tree in model1.estimators_],

#              axis=0)

# indices = np.argsort(importances)[::-1]
# Unbalanced dataset

# X = train.iloc[:,np.r_[:,0:8,9:76]]  #independent columns

# y = train.iloc[:,np.r_[:,8]]    #target column

# Balanced dataset

y = df_upsampled.target

X = df_upsampled.drop('target', axis=1)

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model1 = ExtraTreesClassifier()

model1.fit(X,y)

print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model1.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
# X = dataset[:,np.r_[:,0:8,9:76]]   #independent columns

# y = dataset[:,np.r_[:,8]]    #target column

y = df_upsampled.target

X = df_upsampled.drop('target', axis=1)

#get correlations of each features in dataset

corrmat = train.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sb.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# from xgboost import XGBClassifier

# from xgboost import plot_importance



# # X = dataset[:,np.r_[:,0:8,9:76]]   #independent columns

# # y = dataset[:,np.r_[:,8]]    #target column

# y = df_upsampled.target

# X = df_upsampled.drop('target', axis=1)

# # fit model no training data

# model2 = XGBClassifier()

# model2.fit(X,y)

# # feature importance

# print(model2.feature_importances_)

# # plot feature importance



# plt.figure(figsize=(3,6))

# plot_importance(model2,max_num_features=20)

# plt.show()
# from numpy import sort

# from xgboost import XGBClassifier

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score

# from sklearn.feature_selection import SelectFromModel



# # X = train.iloc[:,np.r_[:,0:8,9:76]]  #independent columns

# # Y = train.iloc[:,np.r_[:,8]]    #target column

# y = df_upsampled.target

# X = df_upsampled.drop('target', axis=1)



# # split data into train and test sets

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

# # fit model on all training data

# model = XGBClassifier()

# model.fit(X_train, y_train)

# # make predictions for test data and evaluate

# y_pred = model.predict(X_test)

# predictions = [round(value) for value in y_pred]

# accuracy = accuracy_score(y_test, predictions)

# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# # Fit model using each importance as a threshold

# thresholds = sort(model.feature_importances_)

# for thresh in thresholds:

#     # select features using threshold

#     selection = SelectFromModel(model, threshold=thresh, prefit=True)

#     select_X_train = selection.transform(X_train)

#     # train model

#     selection_model = XGBClassifier()

#     selection_model.fit(select_X_train, y_train)

#     # eval model

#     select_X_test = selection.transform(X_test)

#     y_pred = selection_model.predict(select_X_test)

#     predictions = [round(value) for value in y_pred]

#     accuracy = accuracy_score(y_test, predictions)

#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
## Import the random forest model.

from sklearn.ensemble import RandomForestClassifier 

## This line instantiates the model. 

model3 = RandomForestClassifier() 

## Fit the model on your training data.

model3.fit(X, y)
feature_importances = pd.DataFrame(model3.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances
(pd.Series(model3.feature_importances_, index=X.columns).nlargest(20).plot(kind='barh'))
# display the relative importance of each attribute

output1=model1.feature_importances_
# output2=model2.feature_importances_
output3=model3.feature_importances_
output = output1 + output3 #  + output2
n=18

important_features=np.argsort(output)[::-1][:n]
important_features
training_data = X.iloc[:,important_features]

training_label = y
testing_data=test.iloc[:,important_features]
# train.shape
# training_data = train.iloc[:,np.r_[:,0:8,9:76]]  #independent columns

# training_label = train.iloc[:,np.r_[:,8]]   #target column

training_label = df_upsampled.target

training_data = df_upsampled.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(training_data,training_label,test_size=0.33,random_state=101)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
logmodel = LogisticRegression(C=0.01, solver='liblinear')

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print("Accuracy is", accuracy_score(y_test,predictions)*100)
cm1 = confusion_matrix(y_test,predictions)
print(cm1)
print(classification_report(y_test,predictions))
plt.clf()

plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.Wistia)

classNames = ['Negative','Positive']

plt.title('Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')

tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm1[i][j]))

plt.show()
# scaler = StandardScaler()
# scaler.fit(training_data)
# scaled_features = scaler.transform(training_data)
# scaled_features
# X_train, X_test, y_train, y_test = train_test_split(scaled_features,training_label,test_size=0.30)

# print ('Train set:', X_train.shape,  y_train.shape)

# print ('Test set:', X_test.shape,  y_test.shape)
# error_rate = []



# # Will take some time

# for i in range(1,20):

    

#     knn = KNeighborsClassifier(n_neighbors=i)

#     knn.fit(X_train,y_train)

#     pred_i = knn.predict(X_test)

#     error_rate.append(np.mean(pred_i != y_test))
# knn = KNeighborsClassifier(n_neighbors=1) # n_neighbors = k
# knn.fit(X_train,y_train)
# predictions = knn.predict(X_test)
# from sklearn import metrics

# print("Train set Accuracy: ", metrics.accuracy_score(y_train, knn.predict(X_train)))

# print("Test set Accuracy: ", metrics.accuracy_score(y_test, predictions))
# print("Accuracy is", accuracy_score(y_test,predictions)*100)
# cm2 = confusion_matrix(y_test,predictions)
# print(cm2)
# print(classification_report(y_test,predictions))
# plt.clf()

# plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.Wistia)

# classNames = ['Negative','Positive']

# plt.title('Confusion Matrix')

# plt.ylabel('True label')

# plt.xlabel('Predicted label')

# tick_marks = np.arange(len(classNames))

# plt.xticks(tick_marks, classNames, rotation=45)

# plt.yticks(tick_marks, classNames)

# s = [['TN','FP'], ['FN', 'TP']]

# for i in range(2):

#     for j in range(2):

#         plt.text(j,i, str(s[i][j])+" = "+str(cm2[i][j]))

# plt.show()
X_train, X_test, y_train, y_test = train_test_split(training_data,training_label,test_size=0.3,random_state=101)
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print("Accuracy is", accuracy_score(y_test,predictions)*100)
print(classification_report(y_test,predictions))
cm3 = confusion_matrix(y_test,predictions)
print(cm3)
plt.clf()

plt.imshow(cm3, interpolation='nearest', cmap=plt.cm.Wistia)

classNames = ['Negative','Positive']

plt.title('Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')

tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm3[i][j]))

plt.show()
rfc = RandomForestClassifier(n_estimators=170)

rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print("Accuracy is", accuracy_score(y_test,rfc_pred)*100)
print(classification_report(y_test,rfc_pred))
cm4 = confusion_matrix(y_test,rfc_pred)
print(cm4)
plt.clf()

plt.imshow(cm4, interpolation='nearest', cmap=plt.cm.Wistia)

classNames = ['Negative','Positive']

plt.title('Confusion Matrix')

plt.ylabel('True label')

plt.xlabel('Predicted label')

tick_marks = np.arange(len(classNames))

plt.xticks(tick_marks, classNames, rotation=45)

plt.yticks(tick_marks, classNames)

s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):

    for j in range(2):

        plt.text(j,i, str(s[i][j])+" = "+str(cm4[i][j]))

plt.show()
# X_train, X_test, y_train, y_test = train_test_split(training_data, training_label, test_size=0.30, random_state=101)

# print ('Train set:', X_train.shape,  y_train.shape)

# print ('Test set:', X_test.shape,  y_test.shape)
# model = SVC()
# model.fit(X_train,y_train) # If C is 0, we can have no margin kernel ='Radial Basis Functions'(Big cone located in all points of data set)
# predictions = model.predict(X_test)
# cm5 = confusion_matrix(y_test,predictions)
# print("Accuracy is", accuracy_score(y_test,predictions)*100)
# print(cm5)
# print(classification_report(y_test,predictions))
# plt.clf()

# plt.imshow(cm5, interpolation='nearest', cmap=plt.cm.Wistia)

# classNames = ['Negative','Positive']

# plt.title('Confusion Matrix')

# plt.ylabel('True label')

# plt.xlabel('Predicted label')

# tick_marks = np.arange(len(classNames))

# plt.xticks(tick_marks, classNames, rotation=45)

# plt.yticks(tick_marks, classNames)

# s = [['TN','FP'], ['FN', 'TP']]

# for i in range(2):

#     for j in range(2):

#         plt.text(j,i, str(s[i][j])+" = "+str(cm5[i][j]))

# plt.show()
submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')
final_prediction=rfc.predict(test)
submission["target"] = rfc.predict_proba(test)[:, 1]
# submission["target"] =final_prediction
submission.head()
submission.to_csv('submission.csv', index=False)