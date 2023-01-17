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
# Reading the dataset

df = pd.read_csv("/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv")
# Displaying the first 5 rows of the dataset

df.head()
# Scatter Plot of the Feature vs the Target Label

plt.scatter(df["YearsExperience"], df["Salary"],  color='black')
# The high influence is expected as there is only one column in fluencing the output

df.corr() 
# Plotting correlation matrix

import seaborn as sn

import matplotlib.pyplot as plt



corrMatrix = df.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
X, y = df.iloc[:, :-1], df.iloc[:, [-1]]
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

import math



def fitFunction(X, y, split_ratio):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

    regr = LinearRegression()

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    # Regression Coefficient

    print(f"Regression Coefficient = {regr.coef_}")

    # MSE

    mse = mean_squared_error(y_test, y_pred)

    print(f"MSE acc to SKLEARN = {mse}")

    # RMSE

    print(f"RMSE acc to SKLEARN = {math.sqrt(mse)}")

    # R-2 Score

    print(f"R-2 Score is = {r2_score(y_test, y_pred)}")

    

    # Custom MSE calculator

    custom_mse = np.square(np.subtract(y_test,y_pred)).mean()

    print(f"Custom MSE = {custom_mse}")

    

    # Custom RMSE

    custom_rmse = math.sqrt(custom_mse)

    print(f"Custom RMSE = {custom_rmse}")

    return regr, X_test, y_test
regression_50, X_test50, y_test50 = fitFunction(X, y, 0.5)
regression_70, X_test70, y_test70 = fitFunction(X, y, 0.3)
regression_80, X_test80, y_test80 = fitFunction(X, y, 0.2)
# Plot outputs

def plotter(regr, X_test, y_test):

    plt.scatter(X_test, y_test,  color='black')

    plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)



    plt.xticks(())

    plt.yticks(())



    plt.show()
plotter(regression_50, X_test50, y_test50)
plotter(regression_70, X_test70, y_test70)
plotter(regression_80, X_test80, y_test80)