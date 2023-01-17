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
#import dataset

dataset = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')



#Data Change

dataset.loc[dataset['yr_renovated']!=0, 'yr_renovated' ] = dataset['yr_renovated'] - dataset['yr_built']

dataset.loc[dataset['yr_built']!=0, 'yr_renovated' ] = 2015 - dataset['yr_built']



X = dataset.iloc[:,3:].values

y = dataset.iloc[:,2].values
#building optimal model using backward eliminational 

import statsmodels.api as sm

X = np.append(arr=np.ones((21613,1)).astype(int), values= X, axis = 1)

X_opt = X[:,[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 8)



#Fitting Linear Regression Model to Training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression(fit_intercept= False)

regressor.fit(X_train, y_train)



#Pedicting the test set results

y_pred = regressor.predict(X_test)



from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
