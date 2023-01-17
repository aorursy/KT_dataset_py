import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

% matplotlib inline

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
train.info()
train.shape
train.describe()
train.head()
train.isnull().sum()
data=train.drop(columns=["Serial No."],axis=1)
data.head()
# let see distribution of  the variable

fig=sns.distplot(data['GRE Score'],kde=False)

plt.title("Distribution of GRE Score")

plt.show()



fig = sns.distplot(data['CGPA'],kde=False)

plt.title("Distribution of CGPA")

plt.show()



fig = sns.distplot(data["TOEFL Score"],kde=False)

plt.title("Distribution of TOEFL Score")

plt.show()



fig = sns.distplot(data["SOP"],kde=False)

plt.title("Distribution of SOP")

plt.show()



fig = sns.distplot(data["University Rating"],kde=False)

plt.title("Distribution of University Rating")

plt.show()
fig=sns.regplot(x="GRE Score",y="TOEFL Score",data=data)

plt.title("GRE Score VS TOEFL Score")

plt.show()
fig=sns.regplot(x="GRE Score",y="CGPA",data=data)

plt.title("GRE Score VS CGPA")

plt.show()
fig = sns.lmplot(x="CGPA", y="LOR ", data=data, hue="Research")

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.lmplot(x="GRE Score", y="LOR ", data=data, hue="Research")

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.regplot(x="CGPA", y="SOP", data=data)

plt.title("GRE Score vs CGPA")

plt.show()



fig = sns.regplot(x="GRE Score", y="SOP", data=data)

plt.title("GRE Score vs CGPA")

plt.show()
fig = sns.regplot(x="TOEFL Score", y="SOP", data=data)

plt.title("GRE Score vs CGPA")

plt.show()
corr = data.corr()

fig, ax = plt.subplots(figsize=(8, 8))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

dropSelf = np.zeros_like(corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)

plt.show()
# Split data train & test

from sklearn.model_selection import train_test_split



X = data.drop(['Chance of Admit '], axis=1)

y = data['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, shuffle=False)
# Applying alogorithms 

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor







models = [['DecisionTree :',DecisionTreeRegressor()],

           ['Linear Regression :', LinearRegression()],

           ['Lasso: ', Lasso()],

           ['Ridge: ', Ridge()],

           ['BayesianRidge: ', BayesianRidge()],

           ['ElasticNet: ', ElasticNet()],

           ['HuberRegressor: ', HuberRegressor()],

           ['RandomForest :',RandomForestRegressor()],

           ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],

           ['SVM :', SVR()],

           ['AdaBoostClassifier :', AdaBoostRegressor()],

           ['GradientBoostingClassifier: ', GradientBoostingRegressor()]]

           

          



print("Results...")



for name, model in models:

    model = model

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name, (np.sqrt(mean_squared_error(y_test, predictions))))
