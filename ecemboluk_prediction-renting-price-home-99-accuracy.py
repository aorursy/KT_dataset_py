# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data visualization

import seaborn as sns 

import matplotlib.pyplot as plt



# model selection and metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



# Model libraries

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")

data.head()
data.info()
sns.countplot(data["city"])

plt.show()
sns.countplot(data["animal"])

plt.show()
sns.countplot(data["furniture"])

plt.show()
sns.countplot(data["rooms"])

plt.show()
sns.countplot(data["bathroom"])

plt.show()
sns.countplot(data["parking spaces"])

plt.show()
f,ax = plt.subplots(figsize=(8, 8))

sns.countplot(data["floor"])

plt.show()
data["floor"] = data["floor"].replace("-",np.nan)

data["floor"] = pd.to_numeric(data["floor"])



print(np.mean(data["floor"])) # mean



data["floor"] = data["floor"].fillna(int(np.mean(data["floor"])))



data["animal"] = data["animal"].replace("acept",1)

data["animal"] = data["animal"].replace("not acept",0)

data["furniture"] = data["furniture"].replace("furnished",1)

data["furniture"] = data["furniture"].replace("not furnished",0)



cities = {"São Paulo" : 1, "Porto Alegre" : 2, "Rio de Janeiro": 3, "Campinas":4, "Belo Horizonte":5} 

data["city"] = data["city"].replace(cities)
f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
X = data.drop("total (R$)",axis=1)

Y = data.loc[:,"total (R$)"]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

print(x_train.shape[0])

print(y_train.shape[0])

print(x_test.shape[0])

print(y_test.shape[0])
linear_model = LinearRegression()

linear_model.fit(x_train,y_train)

linear_model_predict = linear_model.predict(x_test)

print("Score: ",r2_score(linear_model_predict,y_test))
ridge_model = Ridge()

ridge_model.fit(x_train,y_train)

ridge_model_predict = ridge_model.predict(x_test)

print("Score: ",r2_score(ridge_model_predict,y_test))
lasso_model = Lasso()

lasso_model.fit(x_train,y_train)

lasso_model_predict = lasso_model.predict(x_test)

print("Score: ",r2_score(lasso_model_predict,y_test))
elasticnet_model = ElasticNet()

elasticnet_model.fit(x_train,y_train)

elasticnet_model_predict = elasticnet_model.predict(x_test)

print("Score: ",r2_score(elasticnet_model_predict,y_test))
predict_price = pd.DataFrame({'Actual Price':y_test,'Prediction Price':linear_model_predict})

predict_price.to_csv('submission.csv', index=False)