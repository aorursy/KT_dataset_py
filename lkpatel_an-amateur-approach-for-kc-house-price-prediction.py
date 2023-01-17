# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

df.describe()

df.head()
df.info()

#fortunately we do not have null values
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#from correlation graph it is clear that 'sqft_living', 'grade' are most important feature

#Also feature 'id' has no significance at all. 

#We will not use these features in final prediction
#lets scatter plot all var against price to locate some outliers if possible

x_vars=df.columns[3:]

for x_var in x_vars:

   df.plot(kind='scatter',x=x_var,y='price') 
#lets find out the house having more than 30 bedrooms

df[df.bedrooms >15]

#one thing to notice here is that this particular data has only 1.75 bathrooms 

#which is strange for such a large mansion. So this indeed is an outlier or have some error

#let remove it

df = df[df.bedrooms <15]

#lets scale the data

X = df.as_matrix(['bedrooms', 'bathrooms', 'sqft_living',\

       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',\

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',\

       'lat', 'long', 'sqft_living15', 'sqft_lot15'])

y = df['price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)
#linear regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

#let us predict

y_pred=model.predict(X_test)

print (model.score(X_test, y_test))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=500)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print (model.score(X_test, y_test))
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=500)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print (model.score(X_test, y_test))
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=10)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print (model.score(X_test, y_test))
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print (model.score(X_test, y_test))