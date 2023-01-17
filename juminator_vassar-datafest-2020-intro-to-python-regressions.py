import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns
# Read in Data

df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df = df.sample(10000, random_state=42)
# Look at first 5 lines of data

df.head()
from sklearn.linear_model import LinearRegression
lr = LinearRegression()



X = df[['number_of_reviews']]

y = df[['price']]

lr.fit(X,y) # train the model
# Coefficient(s)

lr.coef_
import statsmodels.api as sm



X = df[['number_of_reviews']].values

y = df[['price']].values



# fit a OLS model with intercept

X = sm.add_constant(X)

mod = sm.OLS(y, X).fit()



print(mod.summary())
from sklearn.preprocessing import LabelEncoder



lblenc = LabelEncoder()



df['neighbourhood_group_enc'] = lblenc.fit_transform(df['neighbourhood_group'].ravel())
# Dummy Variables!



borough_dummy = pd.get_dummies(df[['neighbourhood_group']], drop_first=True)
lr2 = LinearRegression()



X2 = pd.concat([ df[['number_of_reviews']], borough_dummy ], axis=1)

y2 = df[['price']]

lr2.fit(X2,y2) # train the model
lr2.coef_
from sklearn.model_selection import train_test_split



X = df[['number_of_reviews']].values

y = df[['price']].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
lr3 = LinearRegression()



lr3.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error
# Mean Squared Error (MSE)

mean_squared_error(lr3.predict(X_test), y_test)
# RMSE

np.sqrt(mean_squared_error(lr3.predict(X_test), y_test))
# from sklearn.model_selection import cross_val_score

# from sklearn.metrics import mean_squared_error, median_absolute_error, mean_absolute_error, r2_score

# from sklearn.model_selection import KFold