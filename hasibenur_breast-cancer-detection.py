# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import svm

from sklearn.ensemble import RandomForestRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc

import random

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/GSE58606_data.csv")
data.head()
data=data.dropna(axis=0, how='any')

data.shape
assert data.target.notnull().all()

#returns nothing it means we don't have any nan values.
data.groupby("target_actual").count()
data.groupby("target").count()
correlations= data.corr()

correlations = correlations["target"].sort_values(ascending=False)
corr_many= correlations[correlations >0.5]

corr_many
corr_few= correlations[(correlations >0.10) & (correlations < 0.11)]

corr_few
features= correlations.index[0:10]

f,ax= plt.subplots(figsize=(10,10))

sns.heatmap(data.loc[:,features].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
#Data for Analysis

feature =data[data.columns[0:1926]] #independent columns

target=data.iloc[:,1926] #target column i.e success



model = ExtraTreesClassifier()

model.fit(feature,target)

print(model.feature_importances_) #use inbuilt class 



feat_importances = pd.Series(model.feature_importances_, index=feature.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.title("top 10 most important features in data")

plt.show()
## Normal Breast Tissue

normal= data[data.target==0]

normal
normal.describe()
### the highest correlation

normal["46361 : hsa-miR-1278"].hist()
## the lowest correlation

normal["168626 : hsa-miR-4662a-5p"].hist()
## Cancer data

cancer= data[data.target==1]

cancer.describe()
cancer["46361 : hsa-miR-1278"].hist()
cancer["168626 : hsa-miR-4662a-5p"].hist()
### Normalization

X =data[data.columns[0:1926]] #independent columns

Y=data.iloc[:,1926] #target column i.e target

X= (X - np.min(X))/(np.max(X) - np.min(X))
#Train and Test Splitting

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)



#Model and Training

clf = svm.SVC()

y_pred=clf.fit(X_train,Y_train).predict(X_test)



print("SVM score:", clf.score(X_test,Y_test))
X_test.shape
#Model Evaluation

conf_mat = confusion_matrix(Y_test,y_pred)

acc = accuracy_score(Y_test,y_pred)

precision = precision_score(Y_test,y_pred)

recall = recall_score(Y_test,y_pred)

f1= f1_score(Y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Print Results

print('Confusion Matrix is :')

print(conf_mat)

print('\nAccuracy is :')

print(acc)

print('\nPrecision is :')

print(precision)

print('\nRecall is: ')

print(recall)

print('\nF-score is: ')

print(f1)
#Train and Test Splitting

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)



#Model and Training

gnb = GaussianNB()

y_pred = gnb.fit(X_train,Y_train).predict(X_test)



#Model Evaluation

conf_mat = confusion_matrix(Y_test,y_pred)

acc = accuracy_score(Y_test,y_pred)

precision = precision_score(Y_test,y_pred)

recall = recall_score(Y_test,y_pred)

f1= f1_score(Y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Print Results

print('Confusion Matrix is :')

print(conf_mat)

print('\nAccuracy is :')

print(acc)

print('\nPrecision is :')

print(precision)

print('\nRecall is: ')

print(recall)

print('\nF-score is: ')

print(f1)
print("Naive Bayes score:", gnb.score(X_test,Y_test))
#Train and Test Splitting

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)



#Model and Training

knn = KNeighborsClassifier(n_neighbors=5)

y_pred = knn.fit(X_train, Y_train).predict(X_test)



#Model Evaluation

conf_mat = confusion_matrix(Y_test,y_pred)

acc = accuracy_score(Y_test,y_pred)

precision = precision_score(Y_test,y_pred)

recall = recall_score(Y_test,y_pred)

f1= f1_score(Y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Print Results

print('Confusion Matrix is :')

print(conf_mat)

print('\nAccuracy is :')

print(acc)

print('\nPrecision is :')

print(precision)

print('\nRecall is: ')

print(recall)

print('\nF-score is: ')

print(f1)
print("KNN score:", knn.score(X_test,Y_test))
#Train and Test Splitting

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)



#Model and Training

knn = KNeighborsClassifier(n_neighbors=3)

y_pred = knn.fit(X_train, Y_train).predict(X_test)



#Model Evaluation

conf_mat = confusion_matrix(Y_test,y_pred)

acc = accuracy_score(Y_test,y_pred)

precision = precision_score(Y_test,y_pred)

recall = recall_score(Y_test,y_pred)

f1= f1_score(Y_test,y_pred)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Print Results

print('Confusion Matrix is :')

print(conf_mat)

print('\nAccuracy is :')

print(acc)

print('\nPrecision is :')

print(precision)

print('\nRecall is: ')

print(recall)

print('\nF-score is: ')

print(f1)