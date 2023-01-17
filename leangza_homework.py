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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

dataset = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

dataset.head()
dataset.isnull().sum()
dataset.columns
X = dataset[['bedrooms', 'bathrooms', 'sqft_living','sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade','sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode','lat', 'long','sqft_living15']] 

y = dataset['price']
X.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size =0.2,random_state =0)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
import statsmodels.regression.linear_model as sm

regressor_OLS =sm.OLS(endog=y,exog =X).fit()

regressor_OLS.summary()
X_opt = dataset[['bedrooms', 'bathrooms', 'sqft_living', 'waterfront', 'view', 'condition', 'grade','sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode','lat', 'long','sqft_living15']] 

regressor_OLS =sm.OLS(endog=y,exog =X_opt).fit()

regressor_OLS.summary()
sns.pairplot(dataset[['grade','bedrooms']])

plt.show()
sns.pairplot(dataset[['grade','long']])

plt.show()
g = sns.jointplot("grade", "bedrooms", data=dataset,height=5,kind="reg",ratio=3, color="g")
sns.jointplot(x=dataset["grade"], y=dataset["bedrooms"], kind='scatter', s=200, color='m', edgecolor="skyblue", linewidth=2)
