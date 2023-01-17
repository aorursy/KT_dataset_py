# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the dataset
dataset = pd.read_csv('../input/Salary_Data.csv')
X = dataset.iloc[:, :-1].values #independent variable, matrix of independent variable
y = dataset.iloc[:, 1].values #dependent variable salary; vector 

#checking sample Data
print(dataset)
#check data for any missing values
print("\nSummary Stats of our Data:\n",dataset.describe())

# Splitting the dataset into the Training set and Test set (usually will have a validation set to validate over model, 70-15-15)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
plt.plot(X_train,y_train,'+') # Salary vs Year_experience
plt.xlabel('Years_of_Experience')
plt.ylabel('Salary')
plt.show()
#checking for outliers 
dataset.boxplot(column='Salary')

'''
#to show how boxplot help spot outliers
aa = [1,2,3,4,20,100]
dl = pd.DataFrame(data=aa)
dl.boxplot()
'''
#Modeling 
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

def rmse(predictions, targets):
    print( np.sqrt(((predictions - targets) ** 2).mean()))
    
rmse(regressor.predict(X_train),y_train) #how the model performed against train data
#rmse(y_pred,y_test) #how the model performed against test data
# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
