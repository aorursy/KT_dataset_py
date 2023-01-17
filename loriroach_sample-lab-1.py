# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X1 = np.random.randint(low=500, high=2000, size= 50)

print(X1)

X2 =np.random.randint(low=100, high=500,size=50)

print(X2)

X3 = 3 * X1 + np.random.rand()

print(X3)

Y = X3-X2

print(Y)



#Question 1

df = pd.DataFrame(list(zip(X1, X2,X3,Y)), 

               columns =['X1', 'X2','X3','Y']) 

df

data = df.copy()

print(data)
#Question 2 

#Pearson's Correlation Coefficent

data.corr()

#Question 3



data.plot(kind='scatter',

           x='Y', y='X1',

            title='X1 vs Y')

data.plot(kind='scatter',

           x='Y', y='X2',

           title='X2 vs Y')



import seaborn as sns



sns.pairplot(data,x_vars=['X1','X2'],y_vars=['Y'],size=4)
#Question 4

from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
# separate our data into dependent (Y) and independent(X) variables

X = data[['X1','X2']]

#X2= data[['X2']]

y= data['Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#X2_train, X2_test,y_train,y_test = train_test_split(X2, y, test_size=0.30)



print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)



predictions = lm.predict(X_test)#predicted values for Y



print("R-squared:",model.score(X_test,y_test))

print('Intercept:', model.intercept_)

print('Coefficient:', model.coef_) 



residual= y_test - predictions



plt.scatter(y_test,residual)

#models = sm.OLS(X_train, y_train).fit()

#models.summary()
print(predictions)
sns.residplot(y_test,predictions, lowess=True, color="g")
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

print('Variance score: %.2f' % r2_score(y_test, predictions))

print("Mean squared error: %.2f" % mean_squared_error(y_test,predictions))

print("Mean absolute error: %.2f" % mean_absolute_error(y_test,predictions))

print("Root Mean squared error: %.2f" %np.sqrt(mean_squared_error(y_test,predictions)))