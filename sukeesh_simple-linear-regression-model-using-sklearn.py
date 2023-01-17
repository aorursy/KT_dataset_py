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

housing_data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
housing_data.head()
import datetime

current_time = datetime.datetime.now()

housing_data["age"] = current_time - pd.to_datetime(housing_data["date"])

housing_data["age"] = housing_data["age"].dt.days
features = [

    u'age',  u'bedrooms', u'bathrooms', u'sqft_living', u'sqft_lot', u'waterfront', u'view', u'condition', u'grade',

    u'sqft_above', u'sqft_basement', u'sqft_living15', u'sqft_lot15'

]

x = housing_data[features]

y = housing_data["price"]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=3)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
accuracy = regressor.score(x_test, y_test)

"Accuracy: {}%".format(int(round(accuracy * 100)))
y_pred = regressor.predict(x_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)
import matplotlib.pyplot as plt
plt.scatter(x_test['age'], y_test, color='gray')

plt.plot(x_test['age'], y_pred, color='red')

plt.show()
housing_data.shape
housing_data.describe()
housing_data.isnull().any()
x1 = x

y1 = y
import seaborn as seabornInstance

plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(housing_data["price"])
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, x1.columns, columns=['Coefficient'])  

coeff_df
y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 = df.head(25)

print(df1)
df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))