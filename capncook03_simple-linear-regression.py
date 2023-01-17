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
# suppress warnings



import warnings

warnings.filterwarnings("ignore")
# importing data using the pandas library



import numpy as np

import pandas as pd
# reading the data



advertising = pd.read_csv("/kaggle/input/linear-regression/advertising.csv")

advertising
advertising.shape
advertising.info()
advertising.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# let's do a pairplot to visualize which variables are correlated with Sales the most



sns.pairplot(advertising)
# let's visualize the same using a heatmap



sns.heatmap(advertising.corr(),cmap = "YlGnBu",annot = True)
X = advertising['TV']

y = advertising['Sales']
import statsmodels

import statsmodels.api as sm
import sklearn

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state = 100)
# adding a constant

# the coefficient of this constant will be the intercept



X_train_sm = sm.add_constant(X_train)

X_train_sm
# fitting the model



lr = sm.OLS(y_train,X_train_sm).fit()

lr.params
# performing a summary operation lists out all the different parameters of the regression line fitted

lr.summary()
plt.scatter(X_train,y_train)

plt.plot(X_train,6.948 + 0.054*X_train,'r')
y_train_pred = lr.predict(X_train_sm)

residuals = y_train - y_train_pred
# plotting the residuals



sns.distplot(residuals)
plt.scatter(y = residuals,x = X_train)
# adding a constant



X_test_sm = sm.add_constant(X_test)
# predictions on the test set



y_test_pred = lr.predict(X_test_sm)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
# RMSE



np.sqrt(mean_squared_error(y_test,y_test_pred))
r2 = r2_score(y_true = y_test, y_pred = y_test_pred)

r2
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
X_train = X_train.values.reshape(-1,1)

X_test = X_test.values.reshape(-1,1)
lm.fit(X_train,y_train)
# coefficient of TV



lm.coef_
# intercept



lm.intercept_
# predictions on test set



y_test_pred = lm.predict(X_test)
r2 = r2_score(y_true = y_test,y_pred = y_test_pred)

r2