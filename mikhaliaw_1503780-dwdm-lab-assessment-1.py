# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random 



from scipy.stats import pearsonr

from scipy.stats import spearmanr

import matplotlib.pyplot as plt

from sklearn import metrics

%matplotlib inline

from sklearn.model_selection import train_test_split 

from sklearn import linear_model

from sklearn.linear_model import Ridge

from yellowbrick.datasets import load_concrete

from yellowbrick.regressor import ResidualsPlot

import seaborn as sns

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X1 = []

for m in range (1,51):

    val = random.randint (500,2000)

    X1.append(val)

X2 = []

for m in range (1,51):

    val = random.randint(100,500)

    X2.append(val)

X3 = []

c = 0



while c < 50:

    val = random.random()

    X3.append(X1[c])

    X3[c] = X3 [c] + val * 3

    

    c = c+1
len(X2)
X2[0]
Y = []

q = 0

while q < 50:

    a = X2[q]

    b = X3[q]

    c = a - b

    Y.append(c)

    q =  q + 1

    

    
len (Y)
dataf = {'X1': X1, 'X2': X2,'X3': X3, 'Y': Y}

data = pd.DataFrame(dataf)
data


correlation1, _ = pearsonr(data.X1, data.Y)

print('Pearsons correlation of X1 and Y is %.3f' % correlation1)

correlation1, _ = pearsonr(data.X2, data.Y)

print('Pearsons correlation of X2 and Y is %.3f' % correlation1)

correlation1, _ = pearsonr(data.X3, data.Y)

print('Pearsons correlation of X3 and Y is %.3f' % correlation1)

correlation2, _ = spearmanr(data.X1, data.Y)

print('Spearmans correlation X1 and Y is %.3f' % correlation2)

correlation2, _ = spearmanr(data.X2, data.Y)

print('Spearmans correlation X2 and Y is %.3f' % correlation2)

correlation2, _ = spearmanr(data.X3, data.Y)

print('Spearmans correlation X3 and Y is %.3f' % correlation2)
data.plot(kind='scatter',x='X1',y='Y',

          title="Correlation Table for X1 and Y ",

          figsize=(10,6))
data.plot(kind='scatter',x='X2',y='Y',

          title="Correlation Table for X2 and Y ",

          figsize=(10,6))
X_data = data[['X1','X2']]

Y_data = data['Y']
reg = linear_model.LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
X_train.columns
reg.fit(X_train,y_train)
reg.coef_
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
model = Ridge()

visualizer = ResidualsPlot(model)

visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)

visualizer.score(X_test, y_test)



visualizer.show()
reg.intercept_
reg.score(X_test,y_test)
predicted = reg.predict(X_test)



da = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})

da
a=234

b=95

q = {'x':[a],'y':[b]}

new=pd.DataFrame(q)

reg.predict(new)
X_data

mean_absolute_error(y_test, predicted)
mean_squared_error(y_test, predicted)
rms = sqrt(mean_squared_error(y_test, predicted))

rms