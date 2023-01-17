import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/housing/housing.csv')

df.head()
df.shape
df.describe()
df.ocean_proximity.value_counts()
df2 = pd.get_dummies(df, columns= ['ocean_proximity'])

df2.head()
df2.info()
sns.heatmap(df.corr(), annot= True)
df.hist(figsize=(15,12))
df2.isnull().sum()
df2['total_bedrooms'] = df2['total_bedrooms'].fillna(df2['total_bedrooms'].mean())

df2.isnull().sum()
df2 = df2.drop(['longitude', 'latitude'], axis = 1)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = df2.drop('median_house_value', axis = 1)

y = df2['median_house_value']
x_scaled = sc.fit_transform(x.values)

y_scaled = sc.fit_transform(y.values.reshape(-1, 1)).flatten()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.33, random_state = 1)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

y_pred
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
## Test Score

r2_score(y_test, y_pred)
## Train Score

r2_score(y_train, lr.predict(x_train))
mse = mean_squared_error(y_test, y_pred)

mse
mae = mean_absolute_error(y_test, y_pred)

mae
from sklearn.model_selection import cross_val_score

cross_val_score(lr, x_train, y_train, cv = 10)
## Test Score

cv_score = r2_score(y_test, lr.predict(x_test))

cv_score
lr.coef_
pd.DataFrame(lr.coef_, index= x.columns, columns= ['Coefficients']).sort_values(ascending = False, by = 'Coefficients')
from sklearn.linear_model import Ridge

rr = Ridge()
rr.fit(x_train, y_train)
## Train Score

r2_score(y_train, rr.predict(x_train))
## Test Score

ridge_score = r2_score(y_test, rr.predict(x_test))

ridge_score
rr.coef_
pd.DataFrame(rr.coef_, index= x.columns, columns= ['Coefficients']).sort_values(ascending = False, by = 'Coefficients')
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor()

dtree.fit(x_train, y_train)
## Train Score

r2_score(y_train, dtree.predict(x_train))
## since we got 99%. It seems to be overfit hence we will do cross validation

cross_val_score(dtree, x_train, y_train, cv=10)
## Test Score

dtree_score = r2_score(y_test, dtree.predict(x_test))

dtree_score
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
## Train Score

r2_score(y_train, rf.predict(x_train))
## since we got 94%. It seems to be overfit hence we will do cross validation

cross_val_score(rf, x_train, y_train, cv=10)
## Test Score

rf_score = r2_score(y_test, rf.predict(x_test))

rf_score
algorithm = [cv_score, ridge_score, dtree_score, rf_score]

index = ['Cross Validation','Ridge Regression', 'Decision Tree', 'Random Forest']

pd.DataFrame(algorithm, index=index, columns=['Scores']).sort_values(ascending = False, by=['Scores'])