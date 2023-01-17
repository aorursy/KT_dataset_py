# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))
USA_Housing = pd.read_csv("../input/usa-housingcsv/USA_Housing.csv")

USA_Housing.head()
USA_Housing.info()
USA_Housing.describe()
USA_Housing.columns
sns.pairplot(USA_Housing)
sns.distplot(USA_Housing['Price'])
sns.heatmap(USA_Housing.corr())
x = USA_Housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

               'Avg. Area Number of Bedrooms', 'Area Population']]

y = USA_Housing['Price']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
# print the intercept

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)
sns.distplot(y_test-predictions, bins=50);
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,predictions))

print('MSE:',metrics.mean_squared_error(y_test,predictions))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))