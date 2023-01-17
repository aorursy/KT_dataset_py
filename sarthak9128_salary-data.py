# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing of dataset

data_frame = pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")

X = data_frame.iloc[:, :-1].values

y = data_frame.iloc[:, 1].values
#printing all the data set to verify if its imported or not

print(data_frame)
#this describes the basic stat behind the dataset used

data_frame.describe()
# Plotting of values directly between Years of Experience and Salary

plt.scatter(data_frame["YearsExperience"], data_frame["Salary"],  color='red')

plt.title('Salary vs Years Of Experience')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import math
def TestingSplit(X, y, split_ratio):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=0)

    linreg=LinearRegression()

    linreg.fit(X_train, y_train)

    y_pred = linreg.predict(X_test)            #predicting the salary for the test values

    

    #plotting the actual and predicted values

    compare = [i for i in range (1,len(y_test)+1,1)]

    plt.plot(compare,y_test,color='green',linestyle='-')

    plt.plot(compare,y_pred,color='red',linestyle='-')

    plt.xlabel('Salary')

    plt.ylabel('Index')

    plt.title('Prediction')

    plt.show()

    

    #plotting the error

    compare = [i for i in range (1,len(y_test)+1,1)]

    plt.plot(compare,y_test-y_pred,color='black',linestyle='-')

    plt.xlabel('Index')

    plt.ylabel('Error')

    plt.title('Error Value')

    plt.show()

    

    #calculate mean square error

    mse = mean_squared_error(y_test,y_pred)

    print(f"MSE = {mse}")

    # calculate root mean square error

    rmse = math.sqrt(mse)

    print(f"RMSE acc to SKLEARN = {rmse}")

    print('Cross Check Mean squared error using sklearn: %.2f' % mean_squared_error(y_test,y_pred))

    rms = np.sqrt(mean_squared_error(y_test, y_pred))

    print('Cross Check Root mean squared error using sklearn: %.2f' % rmse)

    return linreg, X_test, y_test

    
def outputplot(regr, X_test, y_test):

    plt.scatter(X_test, y_test,  color='red')

    plt.plot(X_test, regr.predict(X_test), color='black', linewidth=3)

    plt.title('Salary vs Years Of Experience')

    plt.xlabel('Years of Experience')

    plt.ylabel('Salary')

    plt.show()
# Prediction, Error, MSE and RMSE value and final Plot as output for 50% training data and 50% test data

reg50, Xt50, yt50 = TestingSplit(X, y, 0.5)

outputplot(reg50, Xt50, yt50)
# Prediction, Error, MSE and RMSE value and final Plot as output for 70% training data and 30% test data

reg70, Xt70, yt70 = TestingSplit(X, y, 0.3)

outputplot(reg70, Xt70, yt70)
# Prediction, Error, MSE and RMSE value and final Plot as output for 80% training data and 20% test data

reg80, Xt80, yt80 = TestingSplit(X, y, 0.2)

outputplot(reg80, Xt80, yt80)