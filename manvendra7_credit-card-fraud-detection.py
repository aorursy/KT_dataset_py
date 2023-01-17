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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

style.use('ggplot')

%matplotlib inline

plt.rcParams['figure.figsize'] = (15, 8)



seed = 999
train = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#check shape 

print(f'data has {train.shape[0]} number of rows and {train.shape[1]} number of columns')
train.head(10)
train.info()
train.isnull().sum()
train.describe().T
sns.distplot(train['Amount'])

plt.title('Amount Distribution',fontsize=20)

plt.show()
plt.hist(train['Time'])

plt.title('Distribution of Time',fontsize=20)

plt.xlabel('Time')

plt.show()
sns.countplot(train['Class'])

plt.title('Class Counts',fontsize=20)

plt.show()
train['Class'].value_counts()/train.shape[0]  #check percentage of positive and negative class
fraud_cases = train[train['Class'] == 1]  #keep all the fraud cases 

non_fraud_cases = train[train['Class'] == 0]  #keep all the non-fraud cases
#Distribution of amount in fraud cases

plt.hist(fraud_cases['Amount'])

plt.title('Distribution of Amount for fraudulent cases',fontsize=20)

plt.show()
plt.figure(figsize=(12,6))

plt.hist(non_fraud_cases['Amount'])

plt.title('Distribution of Amount for non-fraudulent cases')

plt.show()
plt.figure(figsize=(12,6))

plt.hist(fraud_cases['Time'])

plt.title('Distribution of Time for fraudulent cases')

plt.show()
plt.figure(figsize=(12,6))

plt.hist(non_fraud_cases['Time'])

plt.title('Distribution of Time for non-fraudulent cases')

plt.show()
plt.figure(figsize=(17,8))

cor = train.corr()

cor

#sns.heatmap(cor,annot=True,fmt='.2g')
plt.figure(figsize=(17,8))

cor = train.corr()

sns.heatmap(cor,annot=True,fmt='.2g')
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train['Amount'] = sc.fit_transform(train[['Amount']])

train['Time'] = sc.fit_transform(train[['Time']])
# keep all the predictor variables in X and Classes in Y

X = train.drop('Class', axis=1)

Y = train['Class']
#Check the shape of X and Y

print('Shape of X and Y')

print(X.shape)

print(Y.shape)
#split the data into 67:33 ratio for training and testing

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression



X_train,X_test,Y_train,Y_test = train_test_split(X,Y, stratify = Y, test_size = .33, random_state = 42)



#check proportion of fraud and non-fraud cases in tarining and testing sets



print('Proportion of classes in training set')

print(Y_train.value_counts() / len(Y_train))



print('Proportion of classes in test set')

print(Y_test.value_counts() / len(Y_test))
from sklearn.model_selection import GridSearchCV #import GridSearchCV to find best parameters

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix #import metrices



clf = LogisticRegression(verbose=3,warm_start=True) #create instance of LogisticRegression

params = {'C' : np.power(10.0, np.arange(-3,3))} #set c parameter



#find best parameter value for c

logit_grid = GridSearchCV(clf, param_grid = params, scoring='roc_auc', n_jobs=70)

logit_grid.fit(X_train,Y_train) #fit model on training set

predict = logit_grid.predict(X_test) #predict for test set
#check the roc_auc_score

print('Training score ',roc_auc_score(Y_train,logit_grid.predict(X_train)))

print('Testing score ',roc_auc_score(Y_test,predict))
#Plot confusion matric

from sklearn.metrics import plot_confusion_matrix

conf = confusion_matrix(Y_train,logit_grid.predict(X_train))

sns.heatmap(conf,annot=True,fmt='d')

plt.ylabel('ACtual Class')

plt.xlabel('Predicted Class')

plt.show()
pred_probas = logit_grid.predict_proba(X_train)[:, 1]

fpr,tpr, thresholds = roc_curve(Y_train,pred_probas)

plt.plot(fpr,tpr)

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.axvline(0.1,linestyle='--')

plt.show()
# we will set the threshold as 0.1

predicted_class = []

for i in pred_probas:

    if i > 0.1:

        predicted_class.append(1)

    else:

        predicted_class.append(0)
plt.figure(figsize=(8,5))

from sklearn.metrics import plot_confusion_matrix

conf = confusion_matrix(Y_train,predicted_class)

sns.heatmap(conf,annot=True,fmt='d')

plt.title('Confusion Matrix')

plt.ylabel('ACtual Class')

plt.xlabel('Predicted Class')

plt.show()
roc_auc_score(Y_train,predicted_class)