# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#reading data
data = pd.read_csv('/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv')
print(data.shape)
data.head()
#drop un-necessary columns
data = data.drop(columns=['Unnamed: 0', 'vin','lot','country','condition'], axis=1)
data.describe()
#check for null values
data.isnull().sum()
#check duplicates
duplicate = data[data.duplicated()]
duplicate.shape
data.year.value_counts()
#some rows have price and mileage zero, lets filter them
data = data[data['price']!=0]
data = data[data['mileage']!=0]

data = data[data['year']>2000]
for brand in data["brand"].unique():
    print(brand, end=',\t')
# Plotting a Histogram for brands and number of cars
data['brand'].value_counts().plot(kind='bar', figsize=(20,5))
plt.title("Number of cars by brand")
plt.ylabel("Number of cars")
plt.xlabel("Brand");
# Plotting a scatter plot for brand and price
fig, ax = plt.subplots(figsize=(30,6))
ax.scatter(data['brand'], data['price'])
ax.set_xlabel('Brand')
ax.set_ylabel('Price');
print('Highest price entry \n', data[data.price == data.price.max()])
# Plot Bar chart of number of cars in each year
data['year'].value_counts().nlargest(10).plot(kind='bar', figsize = (20,5))
plt.title("Number of cars vs year")
plt.ylabel("Number of cars")
plt.xlabel("Year");
# Plot Bar chart of number of cars vs color
data['color'].value_counts().nlargest(20).plot(kind='bar', figsize = (20,5))
plt.title("Number of cars vs year")
plt.ylabel("Number of cars")
plt.xlabel("Color");
## Preprocess data
X = data.iloc[:, [1,2,3,4,5,6,7]].values  # input columns are: brand, model, year, title_status, mileage, color, state
y = data.iloc[:, 0].values # output columns are: price

#Encoding text columns
print('Input row before encoding:', X[0])

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X[:,1]=le.fit_transform(X[:,1]) 
X[:,3]=le.fit_transform(X[:,3])
X[:,5]=le.fit_transform(X[:,5])
X[:,6]=le.fit_transform(X[:,6])
X[:,0]=le.fit_transform(X[:,0])

print('Input row after encoding:', X[0])
# Split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
