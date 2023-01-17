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
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from collections import Counter
data = pd.read_csv("../input/creditcard.csv")
data.head()
#31 columns of data. As we can see,all the columns starting wiith 'v' seem to be data extracted from the Principal Components of the original data.
#let's try to learn more about the data.
#Target variable is the 'Class' variable, where 0 means 'fraudulent' and 1 means 'genuine'
data.describe()
len(data)
#A total of 284,807 rows
#let's check the distribution of the target variable 'Class'
Counter(data['Class'])
#plot a barplot to see how unbalanced the data looks graphically
import seaborn as sns
count_class = pd.value_counts(data['Class'])
print(count_class)
plt.figure(figsize=(16,10))
count_class.plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Class')
data.columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['s_Amount']=sc.fit_transform(data['Amount'].reshape(-1,1))
data = data.drop(['Time', 'Amount'], axis=1)
data.columns
x = data.loc[:, data.columns!= 'Class']
y = data.loc[:, data.columns == 'Class']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state = 62)
#1. Find the number of the minority class
number_fraud = len(data[data['Class']==1])
number_non_fraud = len(data[data['Class']==0])
print(number_fraud)
print(number_non_fraud)
#2. Find the indices of the majority class
index_non_fraud = data[data['Class']==0].index
#.3 Find the indices of the minority class
index_fraud = data[data['Class']==1].index
#4. Randomly sample the majority indices with respect to the number of minority classes
random_indices = np.random.choice(index_non_fraud, number_fraud,replace='False')
len(random_indices)
#5. Concat the minority indices with the indices from step 4
under_sample_indices = np.concatenate([index_fraud,random_indices])
#Get the balanced dataframe - This is the final undersampled data
under_sample_df = data.iloc[under_sample_indices]
under_sample_df
Counter(under_sample_df['Class'])
under_sample_class_counts = pd.value_counts(under_sample_df['Class'])
under_sample_class_counts.plot(kind='bar')
x_under = under_sample_df.loc[:, under_sample_df.columns!='Class']
y_under = under_sample_df.loc[:, under_sample_df.columns=='Class']
x_under.columns
y_under.columns
x_under_train, x_under_test, y_under_train, y_under_test = train_test_split(x_under, y_under, test_size=0.25, random_state=100)
x_under_train.head()
y_under_train.head()
from sklearn.linear_model import LogisticRegression
lr_under = LogisticRegression()
lr_under.fit(x_under_train, y_under_train)
from sklearn.metrics import accuracy_score, recall_score
lr_under_predict = lr_under.predict(x_under_test)
lr_under_accuracy = accuracy_score(lr_under_predict, y_under_test)
lr_recall = recall_score(lr_under_predict, y_under_test)
print(lr_under_accuracy)
print(lr_recall)
fraud_sample = data[data['Class']==1].sample(number_non_fraud, replace=True)
#create a new dataframe containing only non-fraud data
df_fraud = data[data['Class']==0]
over_sample_df = pd.concat([fraud_sample,df_fraud], axis=0)
over_sample_class_counts=pd.value_counts(over_sample_df['Class'])
over_sample_class_counts.plot(kind='bar')
plt.xlabel = 'Class'
plt.ylabel = 'Frequency'
x_over = data.loc[:,over_sample_df.columns!='Class']
y_over = data.loc[:,over_sample_df.columns=='Class']
x_over_train, x_over_test, y_over_train, y_over_test = train_test_split(x_over, y_over, test_size = 0.25)
lr_over = LogisticRegression()
lr_over.fit(x_over_train,y_over_train)
lr_over_predict=lr_over.predict(x_over_test)
lr_over_accuracy = accuracy_score(lr_over_predict, y_over_test)
lr_over_recall = recall_score(lr_over_predict, y_over_test)
print(lr_over_accuracy)
print(lr_over_recall)
import imblearn
from imblearn.over_sampling import SMOTE
x_val, x_train_new, y_val,y_train_new = train_test_split(x_train, y_train, test_size = 0.25, random_state=12)
sm = SMOTE()
x_train_res, y_train_res = sm.fit_sample(x_val, y_val)
x_train_res, y_train_res = sm.fit_sample(x_val, y_val)
Counter(y_train_res)
lr_smote = LogisticRegression()
lr_smote.fit(x_train_res, y_train_res)
#predict on the train data
lr_smote_predict = lr_smote.predict(x_train_new)
#print accuracy and recall on train data
print(accuracy_score(lr_smote_predict,y_train_new))
print(recall_score(lr_smote_predict,y_train_new))
#predict on the test data
lr_smote_predict_test = lr_smote.predict(x_test)
print(accuracy_score(lr_smote_predict_test,y_test))
print(recall_score(lr_smote_predict_test,y_test))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train_res, y_train_res)
rf_smote_predict = rf.predict(x_train_new)
rf_smote_predict_test = rf.predict(x_test)
print(accuracy_score(rf_smote_predict,y_train_new))
print(recall_score(rf_smote_predict,y_train_new))
print(accuracy_score(rf_smote_predict_test,y_test))
print(recall_score(rf_smote_predict_test,y_test))
