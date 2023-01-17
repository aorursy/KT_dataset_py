# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the dataset
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
data.drop(columns = ['Serial No.'], axis = 1, inplace = True)
data
# Exploring the dataset
print(data.info(), '\n\n')
print('Information about null values in dataset\n')
print(data.isnull().sum())
#Trying to look at anything of significance
data.describe(include = 'all')
#Listing the columns under the dataset
data.columns
#Looking at how the data behaves with each other
sns.pairplot(data, hue = 'Research')
plt.show()
#Creating the independent vector
X = data.drop(['Chance of Admit '], axis = 1)
#Creating the dependent vector
y = data['Chance of Admit ']
#Printing the two vectors
print(X)
print(y)
#Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#Fitting the X and Y model on our regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)
#Creating y_hat or the predicted values against the testing dataset
y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)
#Evaluating the regressor
from sklearn import metrics as m
print('The mean absolute error is :: ', m.mean_absolute_error(y_test,y_pred))
print('The mean squared error is :: ',m.mean_squared_error(y_test,y_pred))
print('The RMSE is :: ',np.sqrt(m.mean_squared_error(y_test,y_pred)))
