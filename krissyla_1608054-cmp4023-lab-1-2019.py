# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
rng = np.random.RandomState(1) #random state name
frame = pd.DataFrame([],columns = ['X1','X2','X3','Y'])

frame
frame.X1 = [rng.randint(500,2000) #populate frame column X1

for x in rng.rand(50)]

frame

frame.X2 = [rng.randint(100,500) #populate frame column X2

for p in rng.rand(50)]

frame
x = rng.rand(50) + 2 #a vector is a 1D array

x
frame.X3 = 3 * frame.X1 + x

frame
frame.Y = frame.X3 - frame.X2

frame
frame.corr()
#correlation between X1 and Y

print('correlation between X1 and Y usine Pearson algorithm')

frame[['X1','Y']].corr()
#correlation between X2 and Y

print('correlation between X2 and Y')

frame[['X2','Y']].corr()
#correlation between X3 and Y

print('correlation between X3 and Y using spearman algorithm')

frame[['X3','Y']].corr(method="spearman")
plt.scatter(frame.X1, frame.Y)

plt.ylabel("Y Axis")

plt.xlabel("X Axis")

plt.suptitle("Correlation between X1 and Y")

plt.scatter(frame.X2, frame.Y)

plt.ylabel("Y Axis")

plt.xlabel("X Axis")

plt.suptitle("Correlation between X2 and Y")

from sklearn.model_selection import train_test_split

from sklearn import linear_model #linear model package
x_info = frame[['X1','X2']] #independent

y_info = frame['Y'] #dependent
x_train, x_test, y_train, y_test = train_test_split(x_info, y_info, test_size = 0.30)

linreg = linear_model.LinearRegression() #linear regression instance
linreg.fit(x_train,y_train)
linreg.coef_
x_train.columns
print("..Regression Coefficients..")

pd.DataFrame(linreg.coef_[0],index=x_train.columns, columns=["Coefficient"])
import seaborn as sns
plt.title('Residual Plot with Training data (yellow) and test data (blue)  ')

plt.ylabel('Residuals')               

plt.scatter(linreg.predict(x_train), linreg.predict(x_train)-y_train,c='y',s=20) #s -> size of circles

plt.scatter(linreg.predict(x_test),linreg.predict(x_test)-y_test,c='b',s=20)

plt.hlines(y=0,xmin=np.min(linreg.predict(x_test)),xmax=np.max(linreg.predict(x_test)),color='red',linewidth=2) #np - library instance(linear algebra)
from sklearn.metrics import  r2_score
print('Coefficient of Determination: %.2f' % r2_score(y_test, linreg.predict(x_test))) 

#x is predicted value

#y is actual value
linreg.intercept_
#y - dependent variable

#c = intercept

#x = independent

# 2.99992419, -0.99999342 coefficient training data of X1 and X2 respectively
prediction = linreg.predict(x_test)
prediction
outcome = pd.DataFrame({'Original': y_test, 'Prediction' : prediction})

outcome
k = 700

c = 300

values = {'x' : [k], 'y' : [c]}

newdata= pd.DataFrame(values)

linreg.predict(newdata)
from sklearn import metrics

from sklearn.metrics import mean_squared_error
print('Mean squared error: ') 

metrics.mean_absolute_error(y_test, prediction)
print('Mean squared error:')

metrics.mean_squared_error(y_test , prediction)
print('Root mean squared error:')

np.sqrt(metrics.mean_absolute_error(y_test , prediction))