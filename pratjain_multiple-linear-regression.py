import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings('ignore')



#print(os.listdir("../input"))
# Import Data

data = pd.read_csv('../input/50_Startups.csv')

data.head()
x = data.iloc[:,:-1].values

y = data.iloc[:,4].values
# Encode Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

le = LabelEncoder()

x[:,3] = le.fit_transform(x[:,3])

oh = OneHotEncoder(categorical_features = [3])

x = oh.fit_transform(x).toarray()
# Avoiding the Dummy Variable Trap, the column with lesser % of 1's can be dropped

# Since the sklearn library for LinearRegression takes care of Dummy variable trap we do not need to do it here

# Just explicitly mentioned for understanding

x = x[:, 1:]
# Split the data into train & test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)



#extracting data from model

print('y intercepts:',model.intercept_) #prints y intercept

print('model coeff:',model.coef_) #prints the coefficients in the same order as they are passed
y_pred = model.predict(x_test)

print('predictions:',y_pred)
# Model Evaluation

from sklearn import metrics



print('MAE (Mean Absolute error)',metrics.mean_absolute_error(y_test,y_pred))

print('MSE (Mean Squared Error)',metrics.mean_squared_error(y_test,y_pred))

print('RMSE (Root Mean Squared Error)',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# Building the optimal model using Backward Elimination

import statsmodels.formula.api as sm

# we need to add new column, made of ones to take into account b0 cooficient

x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()

regressor_OLS.summary()