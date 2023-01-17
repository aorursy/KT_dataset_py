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
import pandas as pd

advertising = pd.read_csv("../input/advertising/advertising.csv")
advertising.head()
advertising.tail()
advertising.info()
advertising.describe()
# Step_2: Visualizing Data

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's plot a pair plot of all variables in our dataframe

sns.pairplot(advertising)
# Putting Feature variaable to x

x = advertising[['TV','Radio','Newspaper']]



# Putting Response variable to y

y = advertising['Sales']
# random_state is the seed used by the random number generator. It can be any integer.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=100)
from sklearn.linear_model import LinearRegression
# Representing LinearRegression as lr(craeting LinearRegression object)

lr = LinearRegression()
# fit the model to the training data

lr.fit(x_train,y_train)
# print the intercept

print(lr.intercept_)

print(lr.coef_)
# Let's see the coefficient

coeff_df = pd.DataFrame(lr.coef_,x_test.columns,columns=['coefficient'])

coeff_df
# Making predictions using the model

y_pred = lr.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test,y_pred)

r_squared = r2_score(y_test,y_pred)
print("Mean_Square_Error :",mse)

print("r_square_value :",r_squared)
import statsmodels.api as sm

x_train_sm = x_train

# Unlike SKLearn, statsmodel don't automatically fit a constant,

# so you need to use the method sm.add_constant(x) in order to add a constant.

x_train_sm = sm.add_constant(x_train_sm)

# create a fitted model in one line

lm_1 = sm.OLS(y_train,x_train_sm).fit()





# print the coefficients

lm_1.params
print(lm_1.summary())

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline


plt.figure(figsize = (5,5))

sns.heatmap(advertising.corr(),annot = True)
x_train_new = x_train[['TV','Radio']]

x_test_new = x_test[['TV','Radio']]
# Model Building

lr.fit(x_train_new,y_train)
# Make predictions

y_pred_new = lr.predict(x_test_new)
c = [i for i in range(1,61,1)]

fig = plt.figure()

plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")

plt.plot(c,y_pred,color="red",linewidth=2.5,linestyle="-")

fig.suptitle("Actual and Predicted",fontsize=20)

plt.xlabel("Index",fontsize=18)

plt.ylabel("sales",fontsize=16)
c = [i for i in range(1,61,1)]

fig = plt.figure()

plt.plot(c,y_test,color="blue",linewidth=2.5,linestyle="-")

#plt.plot(c,y_pred,color="red",linewidth=2.5,linestyle="-")

fig.suptitle("Error Terms",fontsize=20)

plt.xlabel("Index",fontsize=18)

plt.ylabel("ytest-ypred",fontsize=16)
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test,y_pred_new)

r_squared = r2_score(y_test,y_pred_new)
print("Mean_Squared_Error",mse)

print("r_square_value",r_squared)
x_train_final = x_train_new

x_train_final = sm.add_constant(x_train_final)

lm_final = sm.OLS(y_train,x_train_final).fit()

print(lm_final.summary())