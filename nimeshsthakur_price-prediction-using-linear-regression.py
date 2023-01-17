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
import matplotlib.pyplot as plt 

import seaborn as sns

df =pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')
%matplotlib inline
df.head()
df.info()
df.describe()
sns.pairplot(df)
sns.heatmap(df.corr(),annot = True)
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]

y = df['Price']
from sklearn.model_selection import train_test_split

X_train , X_test, y_train ,y_test = train_test_split(

    X ,y ,test_size = 0.4, random_state = 101

)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train , y_train )
print("intercept")

print(lm.intercept_)
print("coefficeints")

print(lm.coef_)
predictions = lm.predict(X_test)

predictions
plt.scatter(y_test,predictions)
# we would like to visualize the preditions

sns.distplot(y_test - predictions)
from sklearn.metrics import mean_absolute_error , mean_squared_error
mean_absolute_error(y_test , predictions)
mean_squared_error(y_test , predictions)