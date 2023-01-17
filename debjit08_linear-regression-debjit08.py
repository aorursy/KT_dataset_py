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
# lmport Libraries

import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import statsmodels.api as sm
dataset = pd.read_csv("/kaggle/input/Advertising.csv")

print(dataset.shape)

print(dataset.head(5))
dataset.describe()
dataset.plot(x='TV', y='Sales', style='o')  

plt.title('Sales and TV Spend')  

plt.xlabel('TV')  

plt.ylabel('Sales')  

plt.show()
# Selecting the Second, Third and Fouth Column

X= dataset.iloc[:,1:4]

# Selecting Fouth Columnn

y=dataset.iloc[:,4]
# Splitting the Data and output in training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  

regressor.fit(X_train, y_train)
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)
y_pred = regressor.predict(X_test)

df = pd.DataFrame({ 'Actual':y_test.values,'Predicted': y_pred})

ax1 = df.plot.scatter(x='Actual',

                      y='Predicted')
X2 = sm.add_constant(X_train)

est = sm.OLS(y_train, X2)

est2 = est.fit()

print(est2.summary())