# Data files

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Import

%matplotlib inline

import pandas as pd

import numpy as np

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
mushroom = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv', header=0)

mushroom.head(10)
# Checking Missing value

def Missing_table(df):

    # null_val = df.isnull().sum()

    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)

    percent = 100 * null_val/len(df)

    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型

    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)

    missing_table_len = Missing_table.rename(

    columns = {0:'欠損値', 1:'欠損値の割合(%)', 2:'type'})

    return missing_table_len.sort_values(by=['欠損値'], ascending=False)



Missing_table(mushroom)
mushroom['class'].value_counts()



# Replacing some str datas to int

mushroom['class'] = mushroom['class'].replace("e",0).replace("p",1)
# Creating Dummy Variables

capshape = pd.get_dummies(mushroom['cap-shape'])

capsurface = pd.get_dummies(mushroom['cap-surface'])

capcolor = pd.get_dummies(mushroom['cap-color'])

bruise = pd.get_dummies(mushroom['bruises'])

gillattachment = pd.get_dummies(mushroom['gill-attachment'])

odo = pd.get_dummies(mushroom['odor'])

gillspacing = pd.get_dummies(mushroom['gill-spacing'])

gillsize = pd.get_dummies(mushroom['gill-size'])

gillcolor = pd.get_dummies(mushroom['gill-color'])

stalkshape = pd.get_dummies(mushroom['stalk-shape'])

stalkroot = pd.get_dummies(mushroom['stalk-root'])

stalksurfaceabovering = pd.get_dummies(mushroom['stalk-surface-above-ring'])

stalksurfacebelowring = pd.get_dummies(mushroom['stalk-surface-below-ring'])

stalkcolorabovering = pd.get_dummies(mushroom['stalk-color-above-ring'])

stalkcolorbelowring = pd.get_dummies(mushroom['stalk-color-below-ring'])

veiltype = pd.get_dummies(mushroom['veil-type'])

veilcolor = pd.get_dummies(mushroom['veil-color'])

ringnumber = pd.get_dummies(mushroom['ring-number'])

ringtype = pd.get_dummies(mushroom['ring-type'])

sporeprintcolor = pd.get_dummies(mushroom['spore-print-color'])

populations = pd.get_dummies(mushroom['population'])

habitats = pd.get_dummies(mushroom['habitat'])

mushroom.columns
# Combining all of Dummy Variables

total_mushroom = pd.concat([capshape, capsurface, capcolor, bruise, odo, gillattachment, gillspacing,gillsize, gillcolor, stalkshape, stalkroot, stalksurfaceabovering, stalksurfacebelowring, stalkcolorabovering, stalkcolorbelowring, veiltype, veilcolor, ringnumber, ringtype, sporeprintcolor, populations, habitats], axis=1)
# Splitting data arrays into train, test

train_feature = total_mushroom

train_target = mushroom['class']



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
