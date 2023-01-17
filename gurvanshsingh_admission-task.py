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
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
dataset = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

print(dataset)
dataset.isna().sum()
for i in range(1,9):

    print('mean and median of column', i,end = " ")

    print(dataset.iloc[:,i].mean(),'\t',dataset.iloc[:,i].median())
# splitting X and Y and creating the model



X = dataset.iloc[:,1:-1].values

Y = dataset.iloc[:,-1].values

reg = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)

print('Coefficients: ', reg.coef_)

print('Intercept: ', reg.intercept_)
# Printing the predicted and test values



results = pd.DataFrame({

    'Actual': np.array(Y_test).flatten(),

    'Predicted': np.array(Y_pred.round(decimals = 2)).flatten(),

})

print(results)



#Visualizing Actual and predicted values



results= results.astype('float')

df1 = results.head(8)

df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
import sklearn.metrics as met

mse = met.mean_squared_error(Y_test,Y_pred)

print('MSE : ',mse)

r2 = met.r2_score(Y_test,Y_pred)

print('R-square_score : ',r2)

rmse = np.sqrt(mse)

print('RMSE : ',rmse)
from sklearn.ensemble import RandomForestRegressor

# n_estimators is the Ntress i explained the ppt.

regressor = RandomForestRegressor(n_estimators = 2000, random_state = 1001)

regressor.fit(X_train,Y_train)

y_pred = reg.predict(X_test)
mse = met.mean_squared_error(Y_test,y_pred)

print('MSE : ',mse)

r2 = met.r2_score(Y_test,y_pred)

print('R-square_score : ',r2)

rmse = np.sqrt(mse)

print('RMSE : ',rmse)