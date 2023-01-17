# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df
df.info()
df.describe()
df['quality'].value_counts()
from matplotlib import pyplot as plt
import seaborn as sns

print(sns.distplot( df["quality"],color='red' ))
numerical_features=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates", "alcohol"]

for numerical_feature in numerical_features:
    plt.figure()
    df.boxplot(column=[numerical_feature],grid= False )
df=df[(df['volatile acidity']<1.1) & (df['citric acid']<0.9) & (df['residual sugar']<10.0) & (df['chlorides']<0.3) & (df['free sulfur dioxide']<45) & (df['total sulfur dioxide']<250.0) & (df['pH']<3.7) & (df['sulphates']<1.5)]
df
df.loc[(df['quality']<7) , 'quality'] = 0
df.loc[(df['quality']>=7) , 'quality'] = 1
import scipy.stats as stats

numerical_features=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates", "alcohol"]

for numerical_feature in numerical_features:
    p_value= stats.ttest_ind(df[numerical_feature][df['quality'] == 0],
               df[numerical_feature][df['quality'] == 1], equal_var=False
                  ).pvalue
    if p_value<0.05:
        print('We keep the', numerical_feature, 'in the model', p_value)
    else:
        print('We do not keep the', numerical_feature, 'in the model', p_value)
X1=df.drop(columns=['quality'])
y=df['quality']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler

X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

RF = RandomForestClassifier()
RF_reg_parameters = { 'n_estimators': [1,5,10,50,100,200,500] }
grid_RF_acc = GridSearchCV(RF, param_grid = RF_reg_parameters,cv=10)
grid_RF_acc.fit(X_train, y_train)
print(grid_RF_acc.best_estimator_.n_estimators)
from sklearn import metrics

y_pred = grid_RF_acc.predict(X_test)
print('RF Accuracy =', metrics.accuracy_score(y_test, y_pred))
RF.fit(X_train,y_train)

feature_imp = pd.Series(RF.feature_importances_,index=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates", "alcohol"]).sort_values(ascending=False)
feature_imp
X2=df.drop(columns=['quality','free sulfur dioxide','pH','chlorides'])
y=df['quality']
X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.3,random_state=0)
RF = RandomForestClassifier()
RF_reg_parameters = { 'n_estimators': [1,5,10,50,100,200,500] }
grid_RF_acc = GridSearchCV(RF, param_grid = RF_reg_parameters,cv=10)
grid_RF_acc.fit(X_train, y_train)
print(grid_RF_acc.best_estimator_.n_estimators)
y_pred = grid_RF_acc.predict(X_test)
print('RF Accuracy =', metrics.accuracy_score(y_test, y_pred))
from xgboost import XGBClassifier

X1=df.drop(columns=['quality'])
y=df['quality']

X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.3,random_state=0)


XGB = XGBClassifier()
XGB_reg_parameters = { 'n_estimators': [1,5,10,50,100,200,500] }
grid_XGB_acc = GridSearchCV(RF, param_grid = XGB_reg_parameters,cv=10)
grid_XGB_acc.fit(X_train, y_train)
print(grid_XGB_acc.best_estimator_.n_estimators)
y_pred = grid_XGB_acc.predict(X_test)
print('XGB Accuracy =', metrics.accuracy_score(y_test, y_pred))
XGB.fit(X_train,y_train)

feature_imp = pd.Series(XGB.feature_importances_,index=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates", "alcohol"]).sort_values(ascending=False)
feature_imp
X2=df.drop(columns=['quality','citric acid','pH','chlorides','residual sugar'])
y=df['quality']
X_train,X_test,y_train,y_test=train_test_split(X2,y,test_size=0.3,random_state=0)
XGB = XGBClassifier()
XGB_reg_parameters = { 'n_estimators': [1,5,10,50,100,200,500] }
grid_XGB_acc = GridSearchCV(RF, param_grid = XGB_reg_parameters,cv=10)
grid_XGB_acc.fit(X_train, y_train)
print(grid_XGB_acc.best_estimator_.n_estimators)
y_pred = grid_XGB_acc.predict(X_test)
print('XGB Accuracy =', metrics.accuracy_score(y_test, y_pred))