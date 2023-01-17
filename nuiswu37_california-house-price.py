import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset

dataset = pd.read_csv('../input/housing.csv')

dataset.fillna(method='ffill', inplace=True)
X = dataset.drop('median_house_value',axis=1)

y = dataset['median_house_value']



# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(),[8])],

    remainder='passthrough'

)



X = np.array(ct.fit_transform(X),dtype=np.float)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
import statsmodels.api as sm

X = np.append(arr=np.ones((20640,1)).astype(int),values = X,axis=1)
X_opt = X[:,[0,1,2,3,4,5,6,7,8]] # Initialize the metrix

regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()

regressor_OLS.summary()