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
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
dataset = pd.read_csv("../input/advtlr/Advertising.csv")
print(dataset.head())
X= dataset.iloc[:,1:3]
y=dataset.iloc[:,4]
regressor = LinearRegression()  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor.fit(X_train, y_train)
X2 = sm.add_constant(X_train)
est1 = sm.OLS(y_train, X2)
est1 = est1.fit()
print(est1.summary())
regressor = LinearRegression()  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
regressor.fit(X_train, y_train)
X2 = sm.add_constant(X_train)
est2 = sm.OLS(y_train, X2)
est2 = est2.fit()
print(est2.summary())