# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

from sklearn.svm import SVR

import matplotlib.pyplot as plt



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
# Dividing dependent and independent variables

X = df[['Level']]

Y = df[['Salary']]
# Checking for null values

missing_columns_X = [col for col in X.columns if X[col].isnull().any()]

missing_columns_Y = [col for col in Y.columns if Y[col].isnull().any()]
# Analyzing by visualizing

plt.scatter(X, Y, color='blue')

plt.xlabel("Level")

plt.ylabel("Salary")

plt.show()
# Creating model

svm_poly_reg = SVR(kernel="poly", degree=5, C=100, epsilon=0.1)

svm_poly_reg.fit(X, Y['Salary'])
# Predicting and plotting line

plt.scatter(X, Y['Salary'], color='blue')

plt.plot(X, svm_poly_reg.predict(X), color='red')

plt.title('Truth or Bluff (Support Vector Regression Model)')

plt.xlabel("Level")

plt.ylabel("Salary")

plt.show()