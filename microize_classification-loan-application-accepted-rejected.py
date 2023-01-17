#import Libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_column',None)

pd.set_option('display.max_row',None)
audit_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/credit_risk/training_set_labels.csv" )

test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/credit_risk/testing_set_labels.csv')
audit_data.head()
# checking Unique Values for each Column

for col in audit_data.columns:

    print(col,':',audit_data[col].nunique())

    print(audit_data[col].unique())

    print('---'*3)
# Label Encoding

from sklearn.preprocessing import LabelEncoder



label_encod_list=['checking_status','credit_history','purpose','savings_status','employment','personal_status','other_parties','property_magnitude','other_payment_plans','housing','job','own_telephone','foreign_worker']



for col in label_encod_list:

    lenc=LabelEncoder()

    audit_data[col]=lenc.fit_transform(audit_data[col])

    test_data[col]=lenc.transform(test_data[col])

    

audit_data['class']=lenc.fit_transform(audit_data['class'])
# checking for null values

audit_data.isna().sum()
# checking for Duplicates

audit_data.duplicated().sum()
# checking whether Dataset is balanced or not

audit_data['class'].value_counts(normalize=True)
drop_col=['class']

col_need=['checking_status', 'duration','savings_status','other_payment_plans']
# assigning X and y

X=audit_data.drop(drop_col,axis=1)

#X=audit_data[col_need]

y=audit_data['class']



# train_test_split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, stratify=y,random_state=36)



# XGB classifier

import xgboost as xgb

clf= xgb.XGBClassifier(objective='reg:logistic',seed=36)



# GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid={'n_estimators':[100,200,500],'max_depth':[3,5,7] ,'eta':[0.1,0.01,0.001]}

clf_cv=GridSearchCV(clf,cv=3,param_grid=param_grid,verbose=True,n_jobs=-1,scoring='accuracy')



clf_cv.fit(X_train,y_train)

y_predict=clf_cv.predict(X_test)



# confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_predict)



print(cm)

print(clf_cv.best_params_)

print(clf_cv.best_score_)
clf1= xgb.XGBClassifier(objective='reg:logistic',seed=36,max_depth=5, n_estimators=200)

clf1.fit(X_train,y_train)

xgb.plot_importance(clf1)
"""from sklearn.feature_selection import RFECV

selector=RFECV(clf1,cv=5)

selector.fit(X_train,y_train)

print(selector.support_)

print(X_train.columns)"""
#drop_col=['foreign_worker','num_dependents','housing','residence_since','other_parties','job','installment_commitment','credit_history']

test_data=test_data

#test_data=test_data[col_need]

test_predict=clf1.predict(test_data)

test_data['prediction']=test_predict



test_data['prediction']=lenc.inverse_transform(test_data['prediction'])





test_data['prediction'].to_csv('loanapplication.csv',index=False)