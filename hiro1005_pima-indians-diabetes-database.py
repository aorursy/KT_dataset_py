# Data file

import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns 



from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost.sklearn import XGBClassifier 

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC
pima_diabetes = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv', header=0)

pima_diabetes
# Missing table

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'Missing Data', 1:'%', 2:'type'})

    return missing_table_len.sort_values(by=['Missing Data'], ascending=False)



Missing_table(pima_diabetes)

train_feature = pima_diabetes.drop(columns='Outcome')

train_target = pima_diabetes['Outcome']



X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)
# RandomForest==============



rf = RandomForestClassifier()

rf.fit(X_train, y_train)

print('='*20)

print('RandomForestClassifier')

print(f'accuracy of train set: {rf.score(X_train, y_train)}')

print(f'accuracy of test set: {rf.score(X_test, y_test)}')



# SVC==============



svc = SVC(verbose=True, random_state=0)

svc.fit(X_train, y_train)

print('='*20)

print('SVC')

print(f'accuracy of train set: {svc.score(X_train, y_train)}')

print(f'accuracy of test set: {svc.score(X_test, y_test)}')





# LinearSVC==============



lsvc = LinearSVC(verbose=True)

lsvc.fit(X_train, y_train)

print('='*20)

print('LinearSVC')

print(f'accuracy of train set: {lsvc.score(X_train, y_train)}')

print(f'accuracy of test set: {lsvc.score(X_test, y_test)}')



# k-近傍法（k-NN）==============



knn = KNeighborsClassifier(n_neighbors=3) #引数は分類数

knn.fit(X_train, y_train)

print('='*20)

print('KNeighborsClassifier')

print(f'accuracy of train set: {knn.score(X_train, y_train)}')

print(f'accuracy of test set: {knn.score(X_test, y_test)}')





# 決定木==============



decisiontree = DecisionTreeClassifier(max_depth=3, random_state=0)

decisiontree.fit(X_train, y_train)

print('='*20)

print('DecisionTreeClassifier')

print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')

print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')





# SGD Classifier==============



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

print('='*20)

print('SGD Classifier')

print(f'accuracy of train set: {sgd.score(X_train, y_train)}')

print(f'accuracy of test set: {sgd.score(X_test, y_test)}')



# XGBClassifier==============



xgb = XGBClassifier()

xgb.fit(X_train, y_train)

print('='*20)

print('XGB Classifier')

print(f'accuracy of train set: {xgb.score(X_train, y_train)}')

print(f'accuracy of test set: {xgb.score(X_test, y_test)}')



# LGBMClassifier==============



lgbm = LGBMClassifier()

lgbm.fit(X_train, y_train)

print('='*20)

print('LGBM Classifier')

print(f'accuracy of train set: {lgbm.score(X_train, y_train)}')

print(f'accuracy of test set: {lgbm.score(X_test, y_test)}')



# CatBoostClassifier==============



catboost = CatBoostClassifier()

catboost.fit(X_train, y_train)

print('='*20)

print('CatBoost Classifier')

print(f'accuracy of train set: {catboost.score(X_train, y_train)}')

print(f'accuracy of test set: {catboost.score(X_test, y_test)}')
