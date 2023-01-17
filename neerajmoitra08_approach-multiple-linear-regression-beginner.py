# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset= pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
X = dataset.iloc[:,0:11].values

Y = dataset.iloc[:,-1].values
print(X.shape)

print(Y.shape)
#use of backward elimination to remove column 4 and 8 that have alpha > 0.05

import statsmodels.api as sm

X = np.append(arr=np.ones((1599,1)).astype(int), values= X, axis = 1)

X_opt = X[:,[0,2,5,6,7,9,10,11]]

regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_OLS.summary()
print(X_opt.shape)
#fitting on sklearn lib

X_opt = X_opt[:,1:]

#splitting training and test data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_opt, Y, test_size = 0.2, random_state = 8)
#fitting Regression Model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train)
#predicting test set results

Y_pred = regressor.predict(X_test)

#Checking fit of our model

from sklearn.metrics import r2_score 

r2_score(Y_test, Y_pred)