import math

import numpy as np

import pandas as pd

import pandas_profiling

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn import datasets

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from pandas.plotting import scatter_matrix

import xgboost
df_adm = pd.read_csv("../input/Admission_Predict.csv", sep=",")



pandas_profiling.ProfileReport(df_adm)
df_adm.index = df_adm['Serial_No.']

df_adm = df_adm.drop('Serial_No.',axis=1)
df_adm.head()
df_adm.isnull().sum()
plt.figure(figsize=(6,6))

plt.subplot(2,2,1)

fig = df_adm.TOEFL_Score.hist(bins=25)

fig.set_title ('TOEFL_Score')



plt.figure(figsize=(6,6))

plt.subplot(2,2,1)

fig = df_adm.GRE_Score.hist(bins=25)

fig.set_title ('GRE_Score')



plt.figure(figsize=(6,6))

plt.subplot(2,2,1)

fig = df_adm.University_Rating.hist(bins=25)

fig.set_title ('University_Rating')

fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df_adm.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
df_adm['University_Rating'].value_counts()
sns.set(style="whitegrid", color_codes=True)

sns.boxplot(y="TOEFL_Score", x="University_Rating", data=df_adm, palette="PRGn")

sns.despine(offset=10, trim=True)
sns.set(style="whitegrid", color_codes=True)

sns.boxplot(y="GRE_Score", x="University_Rating", data=df_adm, palette="PRGn")

sns.despine(offset=10, trim=True)
fig = sns.relplot(x="TOEFL_Score", y="GRE_Score", hue='University_Rating', data=df_adm)

plt.title("TOEFL_Score e GRE_Score x University_Rating")

plt.show()
a = sns.pairplot(df_adm, height=6, kind="scatter", vars=["GRE_Score", "TOEFL_Score", "Chance_of_Admit_"], hue="University_Rating")

a = a.map_offdiag(plt.scatter,s=50,alpha=0.9)

#remove the top and the right lines

sns.despine()

#additional line to adjust someUniversity_Rating appearance issues

plt.subplots_adjust(top=0.9)
fig = sns.barplot(y="Chance_of_Admit_", x="University_Rating", data=df_adm)

plt.title("Chance_of_Admit_ x University_Rating")

plt.show()
from sklearn.model_selection import train_test_split



X = df_adm.drop(['University_Rating'], axis=1)

y = df_adm['University_Rating']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, shuffle=False)
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

#from xgboost import XGBRegressor

#from catboost import CatBoostRegressor

from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor

from sklearn.metrics import mean_squared_error



models = [['DecisionTree :',DecisionTreeRegressor()],

           ['Linear Regression :', LinearRegression()],

           ['RandomForest :',RandomForestRegressor()],

           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],

           ['SVM :', SVR()],

           ['AdaBoostClassifier :', AdaBoostRegressor()],

           ['GradientBoostingClassifier: ', GradientBoostingRegressor()],

           #['Xgboost: ', XGBRegressor()],

           #['CatBoost: ', CatBoostRegressor(logging_level='Silent')],

           ['Lasso: ', Lasso()],

           ['Ridge: ', Ridge()],

           ['BayesianRidge: ', BayesianRidge()],

           ['ElasticNet: ', ElasticNet()],

           ['HuberRegressor: ', HuberRegressor()]]



print("Results...")





for name,model in models:

    model = model

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))