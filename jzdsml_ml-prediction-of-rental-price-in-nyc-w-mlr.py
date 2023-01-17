# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



# Of course I'm gonna pick Manhattan to work on first



df = pd.read_csv("../input/manhattan.csv")



print(df.head())

df.head()

# import train_test_split

from sklearn.model_selection import train_test_split



streeteasy = pd.read_csv("../input/manhattan.csv")



df = pd.DataFrame(streeteasy)



x = df[['bedrooms',

'bathrooms',

'size_sqft',

'min_to_subway',

'floor',

'building_age_yrs',

'no_fee',

'has_roofdeck',

'has_washer_dryer',

'has_doorman',

'has_elevator',

'has_dishwasher',

'has_patio',

'has_gym']]



y = df[['rent']]



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)



print(len(x_train)/len(x))

from sklearn.linear_model import LinearRegression



mlr = LinearRegression()

mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)



sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 0, 1, 0]]





predict = mlr.predict(sonny_apartment)

print("Predicted rent: $%.2f" % predict)

import matplotlib.pyplot as plt



plt.figure()

plt.scatter(y_test, y_predict, alpha =0.4)

plt.title('Predicted Rent vs Actual Rent')

plt.ylabel('Predicted Rent')

plt.xlabel('Actual Rent (from test dataset)')

plt.show()
print(mlr.coef_)
plt.scatter(df[['size_sqft']], df[['rent']], alpha=0.4)

plt.scatter(df[['min_to_subway']], df[['rent']], alpha=0.4)

plt.scatter(df[['has_gym']], df[['rent']], alpha=0.1)

plt.show()
print(mlr.score(x_train, y_train))

print(mlr.score(x_test, y_test))

residuals = y_predict - y_test

plt.scatter(y_predict, residuals, alpha=0.4)

plt.title('Residual Analysis')

plt.show()
x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]



y = df[['rent']]



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)



lm = LinearRegression()



model = lm.fit(x_train, y_train)



y_predict= lm.predict(x_test)



print("Train score:")

print(lm.score(x_train, y_train))



print("Test score:")

print(lm.score(x_test, y_test))



plt.scatter(y_test, y_predict)

plt.plot(range(20000), range(20000))



plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Actual Rent vs Predicted Rent")



plt.show()





print(lm.coef_)
