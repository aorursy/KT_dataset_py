# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame, Series

import random

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Question 1

dta = { 'X1' : [random.randrange(500,2000,1) for n in range(50)], 'X2': [random.randrange(100,500,1) for n in range(50)],  }

rand_vectors = np.array([random.randrange(0,500,1) for n in range(50)]  ) #random.randrange(0,2000,3)

#print(rand_vectors)

lst = [(dt * 3) for dt in dta['X2']]

dta3 =  [lst[i] + rand_vectors[i] for i in range(len(lst))]

dta['X3'] = dta3

lsty = [dta['X3'][i] + dta['X2'][i] for i in range(50)]

dta['Y'] = lsty



df = pd.DataFrame(dta)

print(df)
#Question 2

corr_X1_Y = df['X1'].corr(df['Y'])

corr_X2_Y = df['X2'].corr(df['Y'])

corr_X3_Y = df['X3'].corr(df['Y'])

print("Correlations \n"

     + "X1 and Y: " + str(corr_X1_Y)

     + "\nX2 and Y: " + str(corr_X2_Y) 

     + "\nX3 and Y: " + str(corr_X3_Y))
#Question 3

#X1 AND Y

df.plot(kind="scatter",

x='X1',

y='Y',

figsize=(12,8)

)

plt.title("Relationship between X1 and Y" ,size=8)

plt.ylabel("Y")

plt.xlabel("X1")



#X2 AND Y

df.plot(kind="scatter",

x='X2',

y='Y',

figsize=(12,8)

)

plt.title("Relationship between X2 and Y" ,size=8)

plt.ylabel("Y")

plt.xlabel("X2")
#Question 4

Xdata = df[['X1','X2']]

Ydata = df['Y']

X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.3)

reg = LinearRegression()  

reg.fit(X_train, y_train)

#reg.coef_

print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficients"])

#reg.intercept_

test_predicted = reg.predict(X_test)



df1 = pd.DataFrame({'Actual': y_test, 'Predicted': test_predicted})

df1.sort_index()
#Question 6

print('Variance score: %.2f' % r2_score(y_test, test_predicted))

#This means that  of the variance in Y can be predicted by the model
#Question 7

coeff_df = pd.DataFrame(reg.coef_, Xdata.columns, columns=['Coefficient'])  

coeff_df

#For every change in X1, Y increases by 0.048187 while for every change

#in X2, Y increases by 4.037307



#Question 8

#y = 0.048187x1 + 4.037307x2 + 
plt.scatter(X_test['X2'],y_test)

plt.plot(X_test['X2'],test_predicted, color='red')

plt.show()

#Question 10

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_predicted))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_predicted))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_predicted)))