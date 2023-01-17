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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
dat = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')
print(dat)
dat.describe().transpose()
X = dat.iloc[:, :-1] # YearsofExperience as X
y = dat.iloc[:, [-1]] # Salary as y
def Test_and_fit(X, y, split_factor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_factor, random_state=0)
    lireg = LinearRegression()
    lireg.fit(X_train, y_train)
    y_pred = lireg.predict(X_test)
    
    # Regression Coefficient
    print(f"Regression Coefficient = {lireg.coef_}")
    
    # Mean Square Error
    mse=np.square(np.subtract(y_test,y_pred)).mean()
    print(f"Mean Square Error:  {mse}")
    
    
    # Root Mean Square Error
    RMSE=math.sqrt(mse)
    print(f"Root Mean Square Error: {RMSE}")
    
    plt.scatter(X_test, y_test,  color='red', marker = 'x')
    plt.plot(X_test, lireg.predict(X_test), color='blue', linewidth=3)

    return lireg, X_test, y_test
regression_50, X_test50, y_test50 = Test_and_fit(X, y, 0.5)
regression_70, X_test70, y_test70 = Test_and_fit(X, y, 0.3)
regression_80, X_test80, y_test80 = Test_and_fit(X, y, 0.2)