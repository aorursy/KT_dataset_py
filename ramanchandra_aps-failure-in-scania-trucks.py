#Import required libraries

import numpy as np 

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import scikitplot as skplot

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings('ignore')

df_train=pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set_processed_8bit.csv")

df_test=pd.read_csv("../input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set_processed_8bit.csv")

print("Shape of the datasets...")

print("Shape of train dataset:",df_train.shape)

print("Shape of the test dataset:",df_test.shape)
df_train.head()
df_test.head()
#Concanate train & test dataset

dataset=pd.concat(objs=[df_train.drop(columns=["class"]),df_test.drop(columns=["class"])],axis=0)

dataset.shape
dataset.head()
dataset.info()
dataset.describe()
total_miss_values=dataset.isna().sum().sort_values(ascending=False)

total_miss_values
#Encode labels to 0 & 1

le=LabelEncoder()

df_train["class"]=le.fit_transform(df_train["class"])

df_test["class"]=le.transform(df_test["class"])

print("Target labels are :",le.classes_);
#Correlation matrix

df_train.corr()
#plot correlation matrix

f=plt.figure(figsize=(15,15))

ax=f.add_subplot(111)

cax=ax.matshow(df_train.corr(),interpolation='nearest')

f.colorbar(cax)

plt.title('Correlation matrix',fontsize=15)

plt.show();
#Train dataset target labels distribution

plt.figure(figsize=(15,8))

sns.distplot(df_train["class"]);
# Train dataset

X_train=df_train.drop(columns=["class"])

y_train=df_train["class"]



#Test dataset

X_test=df_test.drop(columns=["class"])

y_test=df_test["class"]
sm=SMOTE(random_state=42)

#Resample the train dataset

X_train,y_train=sm.fit_sample(X_train,y_train)

print("Resampled train dataset shape :",X_train.shape,y_train.shape);
# distribution of classes in train dataset

plt.figure(figsize=(15,8))

sns.distplot(y_train);
# distribution of classes in test dataset

plt.figure(figsize=(15,8))

sns.distplot(y_test);
df_train.hist(figsize=(16,35),bins=10,xlabelsize=8,ylabelsize=8);
lr=LogisticRegression()

lr.fit(X_train,y_train);
#predict on test data

y_pred_lr=lr.predict(X_test)
#confusion matrix

cm=confusion_matrix(y_test,y_pred_lr,labels=[0,1])

cm
#Plot confusion matrix

skplot.metrics.plot_confusion_matrix(y_test,y_pred_lr,figsize=(15,8),title='Confusion matrix for Logistic Regression model')
#load datasets in lgb formate

train_data=lgb.Dataset(X_train,label=y_train)
#Create the validation dataset

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=42)

print("The shape of X_train is : {} & the shape of y_train is : {}".format(X_train.shape,y_train.shape))

print("The shape of X_val is : {} & the shape of y_val is : {}".format(X_val.shape,y_val.shape))

validation_data=lgb.Dataset(X_val,label=y_val)
#set parameters for training

params={ 'num_leaves':145,

        'object':'binary',

        'metric':['auc','binary_logloss']

       }
#Train the model

num_round=20

lgb_model=lgb.train(params,train_data,num_round,valid_sets=validation_data,early_stopping_rounds=5)

#Prediction on unseen dataset

y_pred=lgb_model.predict(X_test,num_iteration=lgb_model.best_iteration)>0.5
#Confusion matrix

cm=confusion_matrix(y_test,y_pred,labels=[0,1])

cm
#Plot confusion matrix

skplot.metrics.plot_confusion_matrix(y_test,y_pred,figsize=(15,8),title='Confusion matrix for LGBM model')

plt.show()