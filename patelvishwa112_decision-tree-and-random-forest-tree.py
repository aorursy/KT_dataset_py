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

import matplotlib.pyplot as plt

import seaborn as sns
# Read the data from csv file

dataset = pd.read_csv("/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv")
# Explore the dataset

dataset.head(5)
# Describe the variables

dataset.describe()
#Check missing values from each column

dataset.apply(lambda x: sum(x.isnull()), axis=0)
# Create histograms for each feature

dataset.hist(figsize=(20,20))

plt.show()
# Co-relation matrix

corrmat = dataset.corr()

fig = plt.figure(figsize=(12,9))



sns.heatmap(corrmat, vmax = 0.8 , square = True )

plt.show()
# Seperating Independent and Dependent Variables

X = dataset.iloc[:, 1:12].values # All columns (Independent Variables) except 1st one i.e Ambient 

y = dataset.iloc[:, 0:2].values # Only ambient column which is our dependent variables

y = y[:, 0]
# Splitting Training and Testing variables

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state=0)
# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
# Fit the decesion tree with the training data

regressor.fit(X_train, y_train)
# Predict the values related to test data

y_pred = regressor.predict(X_test)
# Calculate the Mean squared error

from sklearn.metrics import mean_squared_error

mse_dt = mean_squared_error(y_test, y_pred)
print("Mean Squared Error", format(mse_dt))
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor_1 = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor_1.fit(X_train, y_train)
# Predicting the test values

y_pred_2 = regressor_1.predict(X_test)
# Calculate the Mean Squared Error

mse_rt = mean_squared_error(y_test, y_pred_2)

print("Mean Squared Error of Random Forest Tree", format(mse_rt))