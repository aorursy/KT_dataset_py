# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
#Load the Pima Diabeties dataset
dataset = pd.read_csv('../input/diabetes.csv')
dataset.head()
dataset.describe(include="all")
dataset.info()
#Check For null values in the dataset
dataset.isnull().sum()
#Check for zero values in the dataset
dataset['Pregnancies'][dataset['Pregnancies'] == 0].count()
dataset['Glucose'][dataset['Glucose'] == 0].count()
dataset['BloodPressure'][dataset['BloodPressure'] == 0].count()
dataset['SkinThickness'][dataset['SkinThickness'] == 0].count()
dataset['Insulin'][dataset['Insulin'] == 0].count()
dataset['BMI'][dataset['BMI'] == 0].count()
dataset['Age'][dataset['Age'] == 0].count()
dataset['DiabetesPedigreeFunction'][dataset['DiabetesPedigreeFunction'] == 0].count()
sns.countplot(dataset['Outcome'])
#Look for outliers in the dataset
fig, axarr = plt.subplots(4, 2, figsize=(12, 10))
sns.boxplot(dataset['Outcome'], dataset['Pregnancies'],ax=axarr[0][0])
sns.boxplot(dataset['Outcome'], dataset['Glucose'],ax=axarr[0][1])
sns.boxplot(dataset['Outcome'], dataset['BloodPressure'],ax=axarr[1][0])
sns.boxplot(dataset['Outcome'], dataset['SkinThickness'],ax=axarr[1][1])
sns.boxplot(dataset['Outcome'], dataset['Insulin'],ax=axarr[2][0])
sns.boxplot(dataset['Outcome'], dataset['BMI'],ax=axarr[2][1])
sns.boxplot(dataset['Outcome'], dataset['Age'],ax=axarr[3][0])
sns.boxplot(dataset['Outcome'], dataset['DiabetesPedigreeFunction'],ax=axarr[3][1])
#replace zero values with median
dataset['Glucose'].replace(0,dataset['Glucose'].median(),inplace=True) 
dataset['BloodPressure'].replace(0,dataset['BloodPressure'].median(),inplace=True) 
dataset['SkinThickness'].replace(0,dataset['SkinThickness'].median(),inplace=True) 
dataset['Insulin'].replace(0,dataset['Insulin'].median(),inplace=True) 
dataset['BMI'].replace(0,dataset['BMI'].median(),inplace=True)
dataset.describe(include="all")
dataset.hist(figsize=(20,30))
sns.pairplot(dataset)
#find the corelation in the dataset
corr = dataset.corr()
corr
sns.heatmap(corr, annot=True)
fig, axarr = plt.subplots(3, 2, figsize=(12, 8))

sns.distplot(dataset['Pregnancies'],ax=axarr[0][0])
sns.distplot(dataset['Glucose'],ax=axarr[0][1])
sns.distplot(dataset['BloodPressure'],ax=axarr[1][0])
sns.distplot(dataset['SkinThickness'],ax=axarr[1][1])
sns.distplot(dataset['Insulin'],ax=axarr[2][0])
sns.distplot(dataset['BMI'],ax=axarr[2][1])

plt.subplots_adjust(hspace=1)
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="Pregnancies", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['Pregnancies'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['Pregnancies'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['Pregnancies'], ax=axarr[1][1])
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="Glucose", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['Glucose'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['Glucose'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['Glucose'], ax=axarr[1][1])
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="BloodPressure", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['BloodPressure'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['BloodPressure'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['BloodPressure'], ax=axarr[1][1])
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="SkinThickness", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['SkinThickness'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['SkinThickness'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['SkinThickness'], ax=axarr[1][1])
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="Insulin", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['Insulin'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['Insulin'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['Insulin'], ax=axarr[1][1])
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="Age", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['BMI'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['BMI'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['BMI'], ax=axarr[1][1])
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="DiabetesPedigreeFunction", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['DiabetesPedigreeFunction'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['DiabetesPedigreeFunction'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['DiabetesPedigreeFunction'], ax=axarr[1][1])
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.boxplot(x="Outcome", y="Age", data=dataset,ax=axarr[0][0])
sns.barplot(dataset['Outcome'], dataset['Age'],ax=axarr[0][1])
sns.stripplot(dataset['Outcome'], dataset['Age'], jitter=True,ax=axarr[1][0])
sns.swarmplot(dataset['Outcome'], dataset['Age'], ax=axarr[1][1])
fig, axarr = plt.subplots(3, 2, figsize=(12, 8))
plt.subplots_adjust(hspace=1)
sns.pointplot(dataset['Pregnancies'], dataset['Age'], hue=dataset['Outcome'],ax=axarr[0][0])
sns.pointplot(dataset['Glucose'], dataset['Age'], hue=dataset['Outcome'],ax=axarr[0][1])
sns.pointplot(dataset['BloodPressure'], dataset['Age'], hue=dataset['Outcome'],ax=axarr[1][0])
sns.pointplot(dataset['SkinThickness'], dataset['Age'], hue=dataset['Outcome'],ax=axarr[1][1])
sns.pointplot(dataset['Insulin'], dataset['Age'], hue=dataset['Outcome'],ax=axarr[2][0])
sns.pointplot(dataset['BMI'], dataset['Age'], hue=dataset['Outcome'],ax=axarr[2][1])
sns.lmplot(x="Pregnancies", y="Age",hue="Outcome", data=dataset)
sns.lmplot(x="Glucose", y="Age",hue="Outcome", data=dataset)
sns.lmplot(x="BloodPressure", y="Age",hue="Outcome", data=dataset)
sns.lmplot(x="SkinThickness", y="Age",hue="Outcome", data=dataset)
sns.lmplot(x="Insulin", y="Age",hue="Outcome", data=dataset)
sns.lmplot(x="BMI", y="Age",hue="Outcome", data=dataset)
sns.jointplot(dataset['Pregnancies'], dataset['DiabetesPedigreeFunction'], kind='hex')
sns.jointplot(dataset['Glucose'], dataset['DiabetesPedigreeFunction'], kind='hex')
sns.jointplot(dataset['BloodPressure'], dataset['DiabetesPedigreeFunction'], kind='hex')
sns.jointplot(dataset['SkinThickness'], dataset['DiabetesPedigreeFunction'], kind='hex')
sns.jointplot(dataset['Insulin'], dataset['DiabetesPedigreeFunction'], kind='hex')
sns.jointplot(dataset['BMI'], dataset['DiabetesPedigreeFunction'], kind='hex')
pd.pivot_table(dataset,index=['Outcome'],aggfunc=len)
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
reg=linear_model.LogisticRegression()
y=dataset['Outcome']
X=dataset.drop(['Outcome'],axis=1)
reg.fit(X,y)
reg.coef_,reg.intercept_
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data_X = dataset.iloc[:,0:8]
data_Y = dataset.iloc[:,8]
select_top_4 = SelectKBest(score_func=chi2, k = 4)
fit = select_top_4.fit(data_X,data_Y)
features = fit.transform(data_X)
features
dataset.head()
# Importing packages for cross validation, logistic regression, RandomForestClassifier, SVC,
# KNeighborsClassifier and XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
X = dataset.drop('Outcome',1)
y = dataset.Outcome                # Save target variable in separate dataset
#Using LogisticRegression
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = LogisticRegression(random_state=1)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
#Using RandomForestClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = RandomForestClassifier(random_state=1, max_depth=3, n_estimators=41)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
#Using XGBClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = XGBClassifier(n_estimators=50, max_depth=4)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
#Using SVC
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = SVC()
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
#Using KNeighborsClassifier
i=1
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
    
     model = KNeighborsClassifier(n_neighbors=20)
     model.fit(xtr, ytr)
     pred_test = model.predict(xvl)
     score = accuracy_score(yvl,pred_test)
     print('accuracy_score',score)
     i+=1
