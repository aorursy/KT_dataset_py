# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import library yang dibutuhkan

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#impor dataset

dataset = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values
#Membagi data menjadi training set dan test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#Flitting simple linear regression terhadap training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#memprediksi hasil test set

y_pred = regressor.predict(X_test)
#Visualisasi hasil training set

plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Gaji vs pengalaman (Training set)')

plt.xlabel('Tahun Bekerja')

plt.ylabel('Gaji')

plt.show()
#visualisasi hasil test set

plt.scatter(X_test, y_test, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Gaji vs pengalaman (Test set)')

plt.xlabel('Tahun Bekerja')

plt.ylabel('Gaji')

plt.show()
#import all the lib

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
# read the dataset using pandas

data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
#This display the top 5 rows of the data

data.head()
#provides some information regarding the columns in the data

data.info()
#this describes the basic stat behind the dataset used

data.describe()
#these plots help to explain the values and how they are scattered

plt.figure(figsize=(12,6))

sns.pairplot(data,x_vars=['YearsExperience'],y_vars=['Salary'],size=7,kind='scatter')

plt.xlabel('Years')

plt.ylabel('Salary')

plt.title('Salary Prediction')

plt.show()
#Cooking the data

x = data['YearsExperience']

x.head()
#import Segregating data from scrikit learn

from sklearn.model_selection import train_test_split
#Split the data for train and test

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=0)
#Create new axis for x column

x_train = x_train[:, np.newaxis]

x_test = x_test[:, np.newaxis]
#importing linear regression model from scikit learn

from sklearn.linear_model import LinearRegression
#Fittting the model

lr = LinearRegression()

lr.fit(x_train, y_train)
#predicting the salary for the test values

y_pred = lr.predict(x_test)
#plotting the actual and predicted values

c = [i for i in range (1,len(y_test)+1,1)]

plt.plot(c,y_test,color='r',linestyle='-')

plt.plot(c,y_pred,color='b',linestyle='-')

plt.xlabel('Salary')

plt.ylabel('Index')

plt.title('Prediction')

plt.show()
#plotting the error

c = [i for i in range (1,len(y_test)+1,1)]

plt.plot(c,y_test-y_pred,color='green',linestyle='-')

plt.xlabel('index')

plt.ylabel('Error')

plt.title('Error Value')

plt.show()
#Importing metrics for the evaluation of the model

from sklearn.metrics import r2_score,mean_squared_error
#calculate mean square error

mse = mean_squared_error(y_test,y_pred)
#calcualte R square vale

rsq = r2_score(y_test,y_pred)
print('mean squared error:',mse)

print('r square:',rsq)
#just plot actual and predicted values for the more insights

plt.figure(figsize=(12,6))

plt.scatter(y_test,y_pred,color='r',linestyle='-')

plt.show()
#intecept and coeff of the line

print('Intercept of the model:',lr.intercept_)

print('Coefficient of the line:',lr.coef_)