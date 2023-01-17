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
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report

creditcard_data=pd.read_csv("../input/creditcard.csv")
creditcard_data['Amount']=StandardScaler().fit_transform(creditcard_data['Amount'].values.reshape(-1, 1))
creditcard_data.drop(['Time'], axis=1, inplace=True)

def generatePerformanceReport(clf,X_train,y_train,X_test,y_test,bool_):
    if bool_==True:
        clf.fit(X_train,y_train.values.ravel())
    pred=clf.predict(X_test)
    cnf_matrix=confusion_matrix(y_test,pred)
    tn, fp, fn, tp=cnf_matrix.ravel()
    print('---------------------------------')
    print('Length of training data:',len(X_train))
    print('Length of test data:', len(X_test))
    print('---------------------------------')
    print('True positives:',tp)
    print('True negatives:',tn)
    print('False positives:',fp)
    print('False negatives:',fn)
    #sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    print('----------------------Classification report--------------------------')
    print(classification_report(y_test,pred))
    
#generate 50%, 66%, 75% proportions of normal indices to be combined with fraud indices
#undersampled data
normal_indices=creditcard_data[creditcard_data['Class']==0].index
fraud_indices=creditcard_data[creditcard_data['Class']==1].index
for i in range(1,4):
    normal_sampled_data=np.array(np.random.choice(normal_indices, i*len(fraud_indices),replace=False))
    undersampled_data=np.concatenate([fraud_indices, normal_sampled_data])
    undersampled_data=creditcard_data.iloc[undersampled_data]
    print('length of undersampled data ', len(undersampled_data))
    print('% of fraud transactions in undersampled data ',len(undersampled_data.loc[undersampled_data['Class']==1])/len(undersampled_data))
    #get feature and label data
    feature_data=undersampled_data.loc[:,undersampled_data.columns!='Class']
    label_data=undersampled_data.loc[:,undersampled_data.columns=='Class']
    X_train, X_test, y_train, y_test=train_test_split(feature_data,label_data,test_size=0.30)
    for j in [LogisticRegression(),SVC(),RandomForestClassifier(n_estimators=100)]:
        clf=j
        print(j)
        generatePerformanceReport(clf,X_train,y_train,X_test,y_test,True)
        #the above code classifies X_test which is part of undersampled data
        #now, let us consider the remaining rows of dataset and use that as test set
        remaining_indices=[i for i in creditcard_data.index  if i not in undersampled_data.index]
        testdf=creditcard_data.iloc[remaining_indices]
        testdf_label=creditcard_data.loc[:,testdf.columns=='Class']
        testdf_feature=creditcard_data.loc[:,testdf.columns!='Class']
        generatePerformanceReport(clf,X_train,y_train,testdf_feature,testdf_label,False)
#oversampled_data data
normal_sampled_indices=creditcard_data.loc[creditcard_data['Class']==0].index
oversampled_data=creditcard_data.iloc[normal_sampled_indices]
fraud_data=creditcard_data.loc[creditcard_data['Class']==1]
oversampled_data=oversampled_data.append([fraud_data]*300, ignore_index=True)
print('length of oversampled_data data ', len(oversampled_data))
print('% of fraud transactions in oversampled_data data ',len(oversampled_data.loc[oversampled_data['Class']==1])/len(oversampled_data))
#get feature and label data
feature_data=oversampled_data.loc[:,oversampled_data.columns!='Class']
label_data=oversampled_data.loc[:,oversampled_data.columns=='Class']
X_train, X_test, y_train, y_test=train_test_split(feature_data,label_data,test_size=0.30)
for j in [LogisticRegression(),RandomForestClassifier(n_estimators=100)]:
    clf=j
    print(j)
    generatePerformanceReport(clf,X_train,y_train,X_test,y_test,True)
    #the above code classifies X_test which is part of undersampled data
    #now, let us consider the remaining rows of dataset and use that as test set
    remaining_indices=[i for i in creditcard_data.index  if i not in oversampled_data.index]
    testdf=creditcard_data.iloc[remaining_indices]
    testdf_label=creditcard_data.loc[:,testdf.columns=='Class']
    testdf_feature=creditcard_data.loc[:,testdf.columns!='Class']
    generatePerformanceReport(clf,X_train,y_train,testdf_feature,testdf_label,False)