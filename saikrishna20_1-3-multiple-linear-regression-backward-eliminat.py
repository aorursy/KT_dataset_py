import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('../input/50-startups/50_Startups.csv')
X = dataset.iloc[:, :-1].values # features
y = dataset.iloc[:, -1].values # target
X[:5]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# we don't need to del the one dummy varible column in the dataframe for linear regression
# but for other model we need to use n-1 dummy columns if there are n unique values in a particular column
# hence to avoid confussion we can del the first column of dummy columns created. 
print(X)
# The dummy variables are always created in the first columns.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred
np.set_printoptions(precision=2) # only two decimals after point
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
import statsmodels.api as sm
# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(float), values = X, axis = 1)
print(X)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)

X_opt
model = sm.OLS(endog = y, exog = X_opt)
regressor_OLS = model.fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:, [0,3, 4, 5]], dtype=float)
#X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
#X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_opt = np.array(X[:, [0,3]], dtype=float)
#X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
print(regressor.predict([[ 0, 0, 160000, 130000, 300000]]))
print(regressor.coef_)
print(regressor.intercept_)