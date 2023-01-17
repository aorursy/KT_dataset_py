import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler,PolynomialFeatures

%matplotlib inline
data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
data.head(5)
data.dtypes
data.drop(["id","date"], axis=1, inplace=True)
data.isnull().sum()
data.describe()
f, axes = plt.subplots(1, 2,figsize=(15,5))

sns.boxplot(x="bedrooms", y="price", data=data, ax=axes[0])

sns.boxplot(x="bathrooms", y="price", data=data, ax=axes[1])

f, axes = plt.subplots(1, 2,figsize=(15,5))

sns.boxplot(x="floors", y="price", data=data, ax=axes[0])

sns.boxplot(x="waterfront", y="price", data=data, ax=axes[1])

f, axes = plt.subplots(1, 2,figsize=(15,5))

sns.boxplot(x="view", y="price", data=data, ax=axes[0])

sns.boxplot(x="grade", y="price", data=data, ax=axes[1])
sns.regplot(x="sqft_living", y="price", data=data)
data['bedrooms'].value_counts().plot(kind='bar')

plt.title('Number of Bedrooms')

plt.xlabel('Bedrooms')

plt.ylabel('Count')
corrmat = data.corr()

f, ax1 = plt.subplots(figsize=(12,9))



ax1=sns.heatmap(corrmat,vmax = 0.8);
corrmat = data.corr()

f, ax1 = plt.subplots(figsize=(12,9))



ax1=sns.heatmap(corrmat,vmax = 0.8,annot = True);
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
x_data = data.drop('price',axis=1)

y_data = data['price']
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)



print("Number of test samples :", x_test.shape[0])

print("Number of training samples:",x_train.shape[0])
lm.fit(x_train,y_train)
lm.score(x_test,y_test)
lm.intercept_
lm.coef_
from sklearn import ensemble

clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, 

                                         learning_rate=0.1, loss='ls')
clf.fit(x_train,y_train)
clf.score(x_test,y_test)