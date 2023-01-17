import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/bostonhoustingmlnd/housing.csv')
df
df.describe()
sns.pairplot(df)
df.describe
x = df[['RM',  'LSTAT',  'PTRATIO']]

y = df['MEDV']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=130)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.coef_)
coeff = pd.DataFrame(lm.coef_, x.columns, columns = ['Coefficiente'])
coeff
predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test - predictions), bins = 50)