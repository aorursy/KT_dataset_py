# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import warnings

import warnings

warnings.filterwarnings('ignore')
#import libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
#import data

data = pd.read_csv('../input/salary-data/Salary_Data.csv')

data.head()
data.shape
data.describe()
sns.pairplot(y_vars = 'Salary', x_vars = 'YearsExperience' ,data = data)
# checking the correlation of the data

data.corr()
X = data['YearsExperience']

y = data['Salary']
X_train,X_test,y_train, y_test = train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 100)
X_train.shape
X_test.shape
X_train_sm = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())
# Let us show the line fitting for train data:

plt.scatter(X_train,y_train)

plt.plot(X_train, 25200 + X_train * 9731.2038,'r')

plt.show()
y_train_pred = model.predict(X_train_sm)
y_train_pred.head()
residual = (y_train - y_train_pred)

residual.head()
sns.distplot(residual)
#checking if there is any pattern to residual

sns.scatterplot(X_train,residual)
X_test_sm = sm.add_constant(X_test)
y_pred = model.predict(X_test_sm)

y_pred.head()
RMSE = np.sqrt(mean_squared_error(y_test,y_pred))

RMSE
r2_score(y_test,y_pred)
# Let us show the line fitting in test data:

plt.scatter(X_test,y_test)

plt.plot(X_test, 25200 + X_test * 9731.2038,'r')

plt.show()