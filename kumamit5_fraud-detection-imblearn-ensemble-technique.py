#Import the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import f1_score, classification_report
#Read the data

data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#Checking the head

data.head()
#Getting the data description

data.info()
#Checking the missing value if any

data.isnull().mean()*100
#Checking the target counts

target_counts = data['Class'].value_counts()
target_counts
#Plot the target count

plt.figure(figsize=(12,6))

sns.barplot(x = target_counts.index, y = target_counts.values)
#Seggregate the data into the dependent and target feature

X = data[[col for col in data.columns if col not in 'Class']]

y = data['Class']
#Getting shape before oversampling

X.shape, y.shape
#Balanced the data set

smk = SMOTETomek()

X_res, y_res = smk.fit_sample(X,y) 
#Getting the shape after the oversampling

X_res.shape, y_res.shape
#Split the data to training and testing set

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.25)
#Fit the Decision tree model



for d in range(2,8):

    dtree = DecisionTreeClassifier(criterion='gini',max_depth=d, class_weight='balanced')

    dtree.fit(X_train, y_train)

    y_pred_t = dtree.predict(X_test)

    print(f'{d} ====  {f1_score(y_train, dtree.predict(X_train))} for trainig')

    print(f'{d} ====  {f1_score(y_test, y_pred_t)} for testing')
#Print classification report of the Decision Tree

print(classification_report(y_test, y_pred_t))
#Fit the Random Forest model

for d in range(2,8):

    rforest = RandomForestClassifier(n_estimators=100,criterion='gini', class_weight='balanced',bootstrap= True,max_depth=d)

    rforest.fit(X_train, y_train)

    y_pred_r = rforest.predict(X_test)

    print(f'{d} ====  {f1_score(y_train, rforest.predict(X_train))} for trainig')

    print(f'{d} ====  {f1_score(y_test, y_pred_r)} for testing')
#Print classification report of the Decision Tree

print(classification_report(y_test, y_pred_r))
#Fit the Adaboost model

for l in range(1,4):

    adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7,class_weight='balanced'),

                              n_estimators=50,learning_rate=l)

    adaboost.fit(X_train, y_train)

    y_pred_a = adaboost.predict(X_test)

    print(f'{l} ====  {f1_score(y_train, adaboost.predict(X_train))} for trainig')

    print(f'{l} ====  {f1_score(y_test, y_pred_a)} for testing')
#Creating Adaboost model with learning_rate = 1

adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=7,class_weight='balanced'),

                              n_estimators=50,learning_rate=1)

adaboost.fit(X_train, y_train)
#Predict the test set result

y_pred_ad = adaboost.predict(X_test)
#Print the classsification report

print(classification_report(y_test,y_pred_ad))
#Set the params and fit the XGBoost model

params = {

    'objective' : 'binary:logistic',

    'max_depth' : 7,

    'learning_rate' : 1,

    'n_estimators' : 10, 

    'gamma' : 1

}
xgb = XGBClassifier(**params)

xgb.fit(X_train, y_train)
#Predict the test set result

y_pred = xgb.predict(X_test)
#Print the classsification report

print(classification_report(y_test,y_pred))