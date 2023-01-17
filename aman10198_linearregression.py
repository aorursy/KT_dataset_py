import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col = 0)
data.head()
data.shape
#visualize the relationship between the features and the response using scatterplots

sns.pairplot(data,x_vars = ['TV', 'radio','newspaper'], y_vars = 'sales', size = 7,

            aspect = 0.7)
### SCIKIT-LEARN ###



# create X and y

feature_cols = ['TV']

X = data[feature_cols]

y = data.sales



# instantiate and fit

lm2 = LinearRegression()

lm2.fit(X, y)



# print the coefficients

print(lm2.intercept_)

print(lm2.coef_)
# There was a spendation of $50.000 in a market what would the expected sales.



lm2.predict([[50]])
sns.pairplot(data, x_vars=['TV','radio','newspaper'],

             y_vars='sales',

             size=7, aspect=0.7, kind='reg')