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
#Get tools

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import time

from sklearn.pipeline import make_pipeline

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_auc_score

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import classification_report, confusion_matrix
df=pd.read_csv('../input/creditcard.csv')

print(df.head())
#From previous exercise, unsupporting variables and variables with collinearity were identified

#Get rid of those for a simpler set of predictors

df.drop(['Amount','Time'],axis=1,inplace=True)

df.drop(['V8','V13','V23','V26','V27','V28'],axis=1,inplace=True)

df.drop(['V2','V3','V5','V7','V9','V11','V15','V19','V20','V21','V22','V24','V25'],axis=1,inplace=True)

print(df.head())
#As in previous exercise, imbalanced class ratio warrants

#(1)training set with balanced class representation; and 

#(2)an unseen test set with a similar distribution of minority to majority class (fraud ~ 0.17%)



#Instead of undersampling the majority class, here I use Synthetic Minority Over-sampling Technique (SMOTE)



#First, split into train and test sets to ensure no information leaked to test set



X,y=df.iloc[:,:-1],df.iloc[:,-1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)



#Check majority vs. minority class distribution in train and test sets



print('Fraudulent share, train set (before SMOTE): {0:.2%}'.format(sum(y_train==1)/len(y_train)))

print('Fraudulent share, test set: {0:.2%}'.format(sum(y_test==1)/len(y_test))) 
#Apply SMOTE to train set

sm=SMOTE(random_state=22)

X_resampled, y_resampled=sm.fit_sample(X_train,y_train)



#Check majority vs. minority class distribution in train set after resampling



print('Fraudulent share, train set (after SMOTE): {0:.2%}'.format(sum(y_resampled==1)/len(y_resampled)))
#Fit Random Forest Classifier used previously to data

#Predict and evaluate effect of SMOTE



RFC_mod=RandomForestClassifier(max_depth=4,random_state=22)



RFC_mod.fit(X_resampled,y_resampled)
#Predict and check AUPRC as well as time the prediction



t0=time.time()

y_pred_RFC=RFC_mod.predict(X_test)

t1=time.time()



print('Predicting with Random Forest Classifier model took: {0:.4f} seconds'.format(t1-t0))



labels=['No Fraud','Fraud']

#Calculate precision recall curve and area under curve

precision_RFC,recall_RFC,threshold_RFC=precision_recall_curve(y_test,y_pred_RFC)

auprc_RFC=auc(recall_RFC,precision_RFC)

print()

print('Area under precision recall curve, Random Forest Classifier model: {0:.4f}'.format(auprc_RFC))

print()

print(classification_report(y_test,y_pred_RFC,target_names=labels))

print()

print(confusion_matrix(y_test,y_pred_RFC))
#Using SMOTE instead of undersampling the majority the class improved model performance

#Area under precision recall curve increased to 0.5445 from 0.4824

#Prediction still took under one second - YAY!



#Let's see if changing hyperparameters of Random Forest Classifier can improve model performance further

#Here, I choose to tweak the 

#(1)max_depth (maximum depth of each tree), and 

#(2)n_estimators (number of trees).



# Maximum depth of each tree

max_depth=[None,5,10]



# Number of trees

n_estimators=[10,12,14,16]



#create grid

grid={

    'max_depth': max_depth,

    'n_estimators': n_estimators

}



#Random search of parameters

RFC_search=GridSearchCV(estimator=RandomForestClassifier(random_state=22), param_grid=grid,cv=3,scoring='f1',verbose=True)



#Fit model

RFC_search.fit(X_resampled, y_resampled)



#print results

print(RFC_search.best_score_)

print(RFC_search.best_params_)

#Wow! That took a long time for hardly an exhaustive search!



#The best max_depth and n_estimators identified are:

#'None' and '16', respectively.

#Refit the random forest classifier with these hyperparameters and check performance



RFC_mod=RandomForestClassifier(random_state=22,max_depth=None,n_estimators=16)

RFC_mod.fit(X_resampled,y_resampled)



t0=time.time()

y_pred_RFC=RFC_mod.predict(X_test)

t1=time.time()



print('Predicting with Random Forest Classifier model took: {0:.4f} seconds'.format(t1-t0))



#Calculate precision recall curve and area under curve

precision_RFC,recall_RFC,threshold_RFC=precision_recall_curve(y_test,y_pred_RFC)

auprc_RFC=auc(recall_RFC,precision_RFC)

print()

print('Area under precision recall curve, Random Forest Classifier model: {0:.4f}'.format(auprc_RFC))

print()

print(classification_report(y_test,y_pred_RFC,target_names=labels))

print()

print(confusion_matrix(y_test,y_pred_RFC))
#Great! The area under the precision recall curve rose to 0.8163

#and the number of false positives and false negatives limited.



#Since the Random Forest Classifier is a 'bagging' algorithm, 

#wouldn't increasing the number of trees improve performance?

#Right now, the default number of trees is 10; but this will be changed to 100 soon.

#Let's fit this soon-to-be 'default' model to our training set and compare the performance.



RFC_mod=RandomForestClassifier(random_state=22,max_depth=None,n_estimators=100)

RFC_mod.fit(X_resampled,y_resampled)



t0=time.time()

y_pred_RFC=RFC_mod.predict(X_test)

t1=time.time()



print('Predicting with Random Forest Classifier model took: {0:.4f} seconds'.format(t1-t0))



#Calculate precision recall curve and area under curve

precision_RFC,recall_RFC,threshold_RFC=precision_recall_curve(y_test,y_pred_RFC)

auprc_RFC=auc(recall_RFC,precision_RFC)

print()

print('Area under precision recall curve, Random Forest Classifier model: {0:.4f}'.format(auprc_RFC))

print()

print(classification_report(y_test,y_pred_RFC,target_names=labels))

print()

print(confusion_matrix(y_test,y_pred_RFC))