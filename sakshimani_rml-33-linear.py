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

# Import library for storing models

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns
# Load Describe and check data types

housing_data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

print(housing_data.dtypes)

housing_data.describe()
# check for missing values

missing_values = housing_data.isnull()

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
sns.boxplot(y='sqft_living',x='waterfront',data=housing_data)
from sklearn.model_selection import train_test_split

df = housing_data[['sqft_living', 'waterfront']]

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)


print(df_train.shape)

print(df_test.shape)
from sklearn.linear_model import LinearRegression

import sklearn

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



model = LinearRegression()

output_model=model.fit(df_train[['waterfront']], df_train['sqft_living'])



output_model
import sklearn

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



print('Coefficients: \n', model.coef_)



print('Mean squared error: %.2f' % mean_squared_error(df_test[['waterfront']], df_test['sqft_living']))

r_squared = r2_score(df_train[['waterfront']], df_train['sqft_living'])

print('\tTrain R_square_value :',r_squared)