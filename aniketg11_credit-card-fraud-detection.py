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
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('../input/creditcard.csv')
data.head()
class_count = pd.value_counts(data['Class']).sort_index()
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Fraud class histogram')
class_count.plot(kind="bar")

from sklearn.preprocessing import StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data['normAmount'].head()
data = data.drop(['Time', 'Amount'], axis=1)
data.columns.values
#No of data points in minority class
no_of_frauds = len(data[data['Class']==1])
print(no_of_frauds)
# Picking the indices of the fraud classes
fraud_indices = np.array(data[data['Class']==1].index)
# Picking the indices of the normal classes
normal_indices = np.array(data[data['Class']==0].index)
# Out of the indices we picked, randomly select "x" number (no_of_frauds)
normal_random_indices = np.random.choice(normal_indices, no_of_frauds, replace=False)
normal_random_indices = np.array(normal_random_indices)

#Append 2 indices
under_sample_indices  = np.concatenate([fraud_indices, normal_random_indices])
print('Appended 2 indices')
#Select rows with indices present in under_sample_indices
under_sample_data = data.iloc[under_sample_indices,:]
print('Select rows with indices present in under_sample_indices')
under_sample_data.head()
#Percentage of normal transactions
normal_trans_per = len(under_sample_data[under_sample_data['Class']==0])/ len(under_sample_data)*100
print('Normal Transaction Percentage', normal_trans_per)

#Percentage of fraud transactions
fraud_trans_per =  len(under_sample_data[under_sample_data['Class']==0])/ len(under_sample_data)*100
print('Normal Transaction Percentage', fraud_trans_per)

x_under_sample_data = under_sample_data
x_under_sample_data = x_under_sample_data.drop('Class', axis=1)
y_under_sample_data = under_sample_data['Class']
print(x_under_sample_data.sample())
y_under_sample_data.head()

x = data
x = x.drop('Class', axis=1)
y=data['Class']
print(len(x))
print(len(y))
from sklearn.model_selection import train_test_split
#whole dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print('Train data length',len(x_train))
print('Test data length', len(x_test))
print('Total Length', len(x))
#Undersampled dataset

under_sampled_x_train, under_sampled_x_test, under_sampled_y_train, under_sampled_y_test = train_test_split(x_under_sample_data, y_under_sample_data, test_size= 0.3, random_state = 0)
print('Under_sampled Train data length', len(under_sampled_x_train))
print('Under_sampled Test data length', len(under_sampled_x_test))
print('Under_sampled Train data length', len(under_sampled_y_train))
print('Under_sampled Test data length', len(under_sampled_y_test))
print('Total Length', len(x_under_sample_data))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc,roc_auc_score, roc_curve, recall_score,classification_report 

c_param_range = [0.01, 0.1, 1, 10, 100]

for i in c_param_range:
    lr = LogisticRegression(C=i, penalty ='l1')
    lr.fit(x_train, y_train)
    pred = lr.predict(x_test)
    recall_acc = recall_score(y_test, pred)
    print('Recall Score of ', recall_acc, ' for C = ', i)

print(under_sampled_x_train.shape)
print(under_sampled_y_train.shape)
print(under_sampled_x_test.shape)
print(under_sampled_y_test.shape)

print(type(under_sampled_x_train))
print(type(under_sampled_y_train))
print(type(under_sampled_x_test))
print(type(under_sampled_y_test))
under_sampled_y_train = under_sampled_y_train.values.reshape(-1, 1)
under_sampled_y_test = under_sampled_y_test.values.reshape(-1, 1)

c_param_range = [0.01, 0.1, 1, 10, 100]

for i in c_param_range:
    lr = LogisticRegression(C=i, penalty ='l1')
    lr.fit(under_sampled_x_train, under_sampled_y_train)
    under_sampled_pred = lr.predict(under_sampled_x_test)
    under_sampled_recall_acc = recall_score(under_sampled_y_test, under_sampled_pred)
    print('Shape' ,under_sampled_recall_acc.shape)
    print('Recall Score of ', under_sampled_recall_acc, ' for C = ', i)
