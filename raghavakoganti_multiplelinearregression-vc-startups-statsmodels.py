import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import statsmodels.api as sm
dataset = pd.read_csv('/kaggle/input/vc-startups/VC_Startups.csv')

dataset
# Independent variables - R&D Spend, Administration, Marketing Spend and State

X = dataset.iloc[:,:-1].values

X
# Dependent variable - Profit

y = dataset.iloc[:,4].values

y
# Encoding 'State' categorical variable

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



st = ColumnTransformer([("State",OneHotEncoder(), [3])], remainder='passthrough')

X = st.fit_transform(X)

X
# Avoiding the Dummy Variable Trap



X = X[:,1:]

X
X.shape
# For StatsModels regression, we need the additional column as bias

# y = mx + c implies y = mx + c*1, so we will add an array of ones to X



X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

X.shape
X_opt = np.array(X[:,[0,1,2,3,4,5]], dtype=float)



regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
X_opt = np.array(X[:,[0,1,3,4,5]], dtype=float)



regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
X_opt = np.array(X[:,[0,3,4,5]], dtype=float)



regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
X_opt = np.array(X[:,[0,3,5]], dtype=float)



regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
X_opt = np.array(X[:,[0,3]], dtype=float)



regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
X_opt = np.array(X[:,[0,3,5]], dtype=float)



regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

regressor_OLS.summary()
# Create new data

new_data = pd.DataFrame({'const':1, 'R&D Spend':[50000, 200000, 400000], 'Marketing Spend':[300000, 150000, 75000]})

new_data = np.array(new_data)

new_data
predictions = regressor_OLS.predict(new_data)

predictions