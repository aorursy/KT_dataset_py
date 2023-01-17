# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing dataset

df = pd.read_csv("../input/polynomial-position-salary-data/Position_Salaries.csv")

df.head()
# Dividing dependent and independent variables

X = df[['Level']]

y = df[['Salary']]
# Checking for null values

missing_columns_X = [col for col in X.columns if X[col].isnull().any()]

missing_columns_Y = [col for col in y.columns if y[col].isnull().any()]
# Creating model

tree_reg = DecisionTreeRegressor(max_depth=4)

tree_reg.fit(X, y)
# Predicting and plotting line

y_hat = tree_reg.predict(X)

plt.scatter(X, y['Salary'], color='blue')

plt.plot(X, y_hat, color='red')

plt.title('Truth or Bluff (Decision Tree Regression)')

plt.xlabel("Level")

plt.ylabel("Salary")

plt.show()
# Decision Tree

plt.figure(figsize=[20, 10])

tree.plot_tree(tree_reg, rounded= True, filled= True)