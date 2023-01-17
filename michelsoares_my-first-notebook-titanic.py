#import the libraries



import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import preprocessing



from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_validate

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



import os

import warnings

warnings.filterwarnings('ignore')
#import the data

df1 = pd.read_csv('../input/titanic/train_and_test2.csv')

df1 = df1[['Passengerid','Age','Fare','Sex','sibsp','Pclass','Embarked','2urvived']]
#visualise df columns

df1.head(10)
# Visualise numerical features

display(df1.describe())
#Data Visualization and understanding the dataset

sns.heatmap(df1.corr(method='pearson'),annot=True,cmap="YlGnBu")
#Pclass is negatively correlated to Survided

sns.barplot(x = 'Pclass', y = '2urvived', order=[1,2,3], data=df1)
#Compare Pclass and sibsp to survided

fig, ax = plt.subplots(1,2)

sns.barplot(x = 'Pclass', y = '2urvived', order=[1,2,3,4,5,6,7], data=df1, ax=ax[0])

sns.barplot(x = 'sibsp', y = '2urvived', order=[1,2,3,4,5,6], data=df1, ax=ax[1])