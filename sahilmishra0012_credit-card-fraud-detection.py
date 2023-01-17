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
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.dtypes
data.isna().any()
data.shape
data.columns
data.hist(figsize=(20,20))

plt.show()
correlation_matrix = data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,vmax=0.8,square = True)

plt.show()
print("Percentage of fraud transactions = ",data['Class'].value_counts()[1]/(data['Class'].value_counts()[1]+data['Class'].value_counts()[0])*100)
y=data['Class']
data=data.drop(['Class'],axis=1)
train_data,test_data,y_train,y_test=train_test_split(data,y,stratify=y,test_size=0.3)

train_data,cv_data,y_train,y_cv=train_test_split(train_data,y_train,stratify=y_train,test_size=0.3)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from tqdm import tqdm

from sklearn.metrics import roc_curve, auc
train_auc = []

cv_auc = []

log_alphas=[]



parameters = {'C':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2.5,5]}



for i in tqdm(parameters['C']):

    lr = LogisticRegression(C=i)

    lr.fit(train_data, y_train)



    y_train_pred = lr.predict(train_data)    

    y_cv_pred = lr.predict(cv_data)

       

    train_auc.append(roc_auc_score(y_train,y_train_pred))

    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))
import math 

for a in tqdm(parameters['C']):

    b = math.log10(a)

    log_alphas.append(b)

print(log_alphas)
plt.figure(figsize=(20,15))

plt.plot(log_alphas, train_auc, label='Train AUC')

plt.plot(log_alphas, cv_auc, label='CV AUC')



plt.scatter(log_alphas, train_auc, label='Train AUC points')

plt.scatter(log_alphas, cv_auc, label='CV AUC points')



plt.legend()

plt.xlabel("C: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
best_alpha=10**-1

lr = LogisticRegression(C=best_alpha)

lr.fit(train_data, y_train)

y_train_pred = lr.predict(train_data)    

y_test_pred = lr.predict(test_data)
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)

test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
x=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.plot(x,x)

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
from imblearn.over_sampling import SMOTE
smt = SMOTE()

train_data, y_train = smt.fit_sample(train_data, y_train)
train_auc = []

cv_auc = []

log_alphas=[]



parameters = {'C':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2.5,5]}



for i in tqdm(parameters['C']):

    lr = LogisticRegression(C=i)

    lr.fit(train_data, y_train)



    y_train_pred = lr.predict(train_data)    

    y_cv_pred = lr.predict(cv_data)

       

    train_auc.append(roc_auc_score(y_train,y_train_pred))

    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))
import math 

for a in tqdm(parameters['C']):

    b = math.log10(a)

    log_alphas.append(b)

print(log_alphas)
plt.figure(figsize=(20,15))

plt.plot(log_alphas, train_auc, label='Train AUC')

plt.plot(log_alphas, cv_auc, label='CV AUC')



plt.scatter(log_alphas, train_auc, label='Train AUC points')

plt.scatter(log_alphas, cv_auc, label='CV AUC points')



plt.legend()

plt.xlabel("C: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
best_alpha=10**-4

lr = LogisticRegression(C=best_alpha)

lr.fit(train_data, y_train)

y_train_pred = lr.predict(train_data)    

y_test_pred = lr.predict(test_data)
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)

test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
x=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.plot(x,x)

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
from imblearn.under_sampling import NearMiss
nms = NearMiss()

train_data, y_train = nms.fit_sample(train_data, y_train)
train_auc = []

cv_auc = []

log_alphas=[]



parameters = {'C':[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2.5,5]}



for i in tqdm(parameters['C']):

    lr = LogisticRegression(C=i)

    lr.fit(train_data, y_train)



    y_train_pred = lr.predict(train_data)    

    y_cv_pred = lr.predict(cv_data)

       

    train_auc.append(roc_auc_score(y_train,y_train_pred))

    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))
import math 

for a in tqdm(parameters['C']):

    b = math.log10(a)

    log_alphas.append(b)

print(log_alphas)
plt.figure(figsize=(20,15))

plt.plot(log_alphas, train_auc, label='Train AUC')

plt.plot(log_alphas, cv_auc, label='CV AUC')



plt.scatter(log_alphas, train_auc, label='Train AUC points')

plt.scatter(log_alphas, cv_auc, label='CV AUC points')



plt.legend()

plt.xlabel("C: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()
best_alpha=10**-4

lr = LogisticRegression(C=best_alpha)

lr.fit(train_data, y_train)

y_train_pred = lr.predict(train_data)    

y_test_pred = lr.predict(test_data)
train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)

test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
x=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

plt.plot(x,x)

plt.legend()

plt.xlabel("K: hyperparameter")

plt.ylabel("AUC")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()