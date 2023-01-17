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

%matplotlib inline
USAhousing = pd.read_csv('../input/usa-housing/USA_Housing.csv')
USAhousing.head()
USAhousing.info()
USAhousing.describe()
USAhousing.columns
USAhousing.isna().sum()
sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price']) 
plt.figure(figsize=(10,10))

sns.heatmap(USAhousing.corr(), annot=True)
plt.figure(figsize=(15,10))



plt.subplot(2,3,1)

plt.title('Area Population')

plt.boxplot(USAhousing['Area Population'])



plt.subplot(2,3,2)

plt.title('Average Area Income')

plt.boxplot(USAhousing['Avg. Area Income'])



plt.subplot(2,3,3)

plt.title('Average Area House Age')

plt.boxplot(USAhousing['Avg. Area House Age'])



plt.subplot(2,3,4)

plt.title('Average Area Number of Rooms')

plt.boxplot(USAhousing['Avg. Area Number of Rooms'])



plt.subplot(2,3,5)

plt.title('Average Area Number of Bedrooms')

plt.boxplot(USAhousing['Avg. Area Number of Bedrooms'])



plt.subplot(2,3,6)

plt.title('Price')

plt.boxplot(USAhousing['Price'])
X = USAhousing.drop(['Address', 'Price'], axis = 1)

y = USAhousing['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print('MSLE:', metrics.mean_squared_log_error(y_test, predictions))

print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(y_test, predictions)))
print('R Square:', metrics.r2_score(y_test, predictions))