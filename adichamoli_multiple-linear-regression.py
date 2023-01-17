# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import statsmodels.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the Dataset



dataset = pd.read_csv('../input/50-startups/50_Startups.csv')

print(dataset.head())
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:,4].values
# Encoding the Independent Variable



labelencoder_X = LabelEncoder()

X[:, 3] = labelencoder_X.fit_transform(X[:, 3])



onehotencoder = OneHotEncoder(categorical_features=[3])

X = onehotencoder.fit_transform(X).toarray()



# Avoiding Dummy Variable Trap



X = X[:, 1:]
# Splitting Dataset into Training and Test set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Fitting Simple Linear Regression to Training Set



regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting Test Set Results



y_pred = regressor.predict(X_test)

print(y_pred)
# Building the optimal model using Backward Elimination



X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



# Repeating Step 3 of Backward Elimination to get P > 0.05



X_opt = X[:, [0, 1, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



X_opt = X[:, [0, 1, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



X_opt = X[:, [0, 3, 4, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



X_opt = X[:, [0, 3, 5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



X_opt = X[:, [0, 3]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())