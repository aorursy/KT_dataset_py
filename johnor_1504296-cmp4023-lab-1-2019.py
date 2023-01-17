# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
randomGen = np.random.RandomState(1)

X1 = randomGen.randint(low=500, high=2000, size=50)

X2 = randomGen.randint(low=100, high=500, size=50)

X3 = X1 * 3 + randomGen.rand()

Y = X3 + X2



data = pd.DataFrame({

    'X1':X1,

    'X2':X2,

    'X3':X3,

    'Y':Y

})

curr1 = pd.DataFrame({

    'X1':X1,

    'Y':Y

})

curr2 = pd.DataFrame({

    'X2':X2,

    'Y':Y

})

curr3 = pd.DataFrame({

    'X3':X3,

    'Y':Y

})

data
curr1.corr()
curr2.corr()
curr3.corr()
plt.scatter(X1, Y)

plt.show()
plt.scatter(X2, Y)

plt.show()
from sklearn.model_selection import train_test_split 

from sklearn import linear_model

import statsmodels.api as sm
data2 = data.copy()
X1_data = data2[['X1']]

X2_data = data2[['X2']]

Y_data = data2['Y']

Y2_data = data2['Y']



X_train, X_test, y_train, y_test = train_test_split(X1_data, Y_data, test_size=0.30)

X2_train,X2_test, y2_train,y2_test = train_test_split(X2_data, Y2_data, test_size=0.30)

pd.DataFrame(X_test)
Regres = linear_model.LinearRegression()

model1 = Regres.fit(X_train,y_train)

model2 = Regres.fit(X2_train,y2_train)

prediction1= Regres.predict(pd.DataFrame(X_test))

prediction2 = Regres.predict(pd.DataFrame(X2_test))
plt.scatter(X_test,prediction1, color='blue')

plt.show()
plt.scatter(X2_test,prediction2, color='red')

plt.show()
import seaborn as sns


sns.set(style="whitegrid")



# Plot the residuals after fitting a linear model

sns.residplot(X_test, y_test, lowess=True, color="b")
sns.residplot(X2_test, y2_test, lowess=True, color="b")
sns.residplot(Regres.predict(X_train), Regres.predict(X_train)-y_train, lowess=True, color="r")

sns.residplot(Regres.predict(pd.DataFrame(X_test)), Regres.predict(pd.DataFrame(X_test))-y_test, lowess=True, color="g")

plt.title('Residual Plot using Training (red) and test (green) data ')

plt.ylabel('Residuals')
sns.residplot(Regres.predict(X2_train), Regres.predict(X2_train)-y_train, lowess=True, color="r")

sns.residplot(Regres.predict(pd.DataFrame(X2_test)), Regres.predict(pd.DataFrame(X2_test))-y_test, lowess=True, color="g")

plt.title('Residual Plot using Training (red) and test (green) data ')

plt.ylabel('Residuals')
from sklearn.metrics import r2_score

r2_score(y_test,prediction1)
from sklearn.metrics import r2_score

r2_score(y2_test,prediction2)
print(Regres.coef_)
print(Regres.intercept_)
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("Mean squared error of X1: %.2f" % mean_squared_error(y_test, prediction1))

print("Mean squared error of X2: %.2f" % mean_squared_error(y2_test, prediction2))
print("Mean squared error of X1: %.2f" % mean_absolute_error(y_test, prediction1))

print("Mean squared error of X2: %.2f" % mean_absolute_error(y2_test, prediction2))

print("Mean squared error of X1: %.2f" % np.sqrt(((prediction1 - y_test) ** 2).mean()))

print("Mean squared error of X2: %.2f" % np.sqrt(((prediction2 - y2_test) ** 2).mean()))