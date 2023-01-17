# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
pd.plotting.register_matplotlib_converters()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, ElasticNet
# Any results you write to the current directory are saved as output.
US_Housing = '../input/usa-housing/USA_Housing.csv'
us_housing_data = pd.read_csv(US_Housing)
us_housing_data.head()
us_housing_data.describe()
us_housing_data.columns
sns.pairplot(us_housing_data)
plt.figure(figsize=(15,10))
sns.heatmap(us_housing_data.corr(), annot=True)
plt.figure(figsize=(10,5))
sns.distplot(us_housing_data['Price'])
X = us_housing_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
      'Avg. Area Number of Bedrooms', 'Area Population']]
#X = us_housing_data.drop(['Price'], axis=1)
y = us_housing_data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
housing_model = ElasticNet()
housing_model.fit(X_train, y_train)

pred = housing_model.predict(X_test)

print("mae :", mean_absolute_error(y_test, pred))
print("msa :", mean_squared_error(y_test, pred))