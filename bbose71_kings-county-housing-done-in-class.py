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
2+3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
# %matplotlib inline

HOUSING_PATH = "../input/kc_house_data.csv"

#Load the data using pandas : Create a DataFrame named housing 
housing = pd.read_csv(HOUSING_PATH)

housing.head(10)
housing.info()
housing.describe()
housing['bedrooms'].value_counts().plot(kind='bar')
plt.title('number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count')
plt.show()

plt.scatter(housing.bedrooms,housing.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(housing.lat, housing.long)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()

plt.scatter(housing.zipcode,housing.price)
plt.title("Which is the pricey location by zipcode?")
plt.show()

from sklearn.linear_model import LinearRegression

attributes = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']
X = np.c_[housing[attributes]]
y = np.c_[housing["price"]]

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

model.score(X,y)
y_predicted = model.predict(X)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_absolute_error(y,y_predicted)

mean_squared_error(y,y_predicted)
r2_score(y, y_predicted)
