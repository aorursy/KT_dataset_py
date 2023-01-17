# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # statistical data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# Ignore warnings

import warnings

warnings.filterwarnings('ignore')
# Load and preview data

df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

df.head()
# View summary of data

df.info()
# Plot the distribution of total bedrooms

df['total_bedrooms'].value_counts().plot.bar()
# Imputing missing values in total_bedrooms by median

df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
# now check for missing values in total bedrooms

df.isnull().sum()
# Declare feature vector and target variable

X = df[['longitude','latitude','housing_median_age','total_rooms',

        'total_bedrooms','population','households','median_income']]

y = df['median_house_value']
# Split the data into train and test data:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# Build the model with Random Forest Classifier :

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)**(0.5)

mse
# import shap library

import shap



# explain the model's predictions using SHAP

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train)



# visualize the first prediction's explanation 

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
# visualize the training set predictions

shap.force_plot(explainer.expected_value, shap_values, X_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)
shap.dependence_plot('median_income', shap_values, X_train)
shap.dependence_plot('longitude', shap_values, X_train)