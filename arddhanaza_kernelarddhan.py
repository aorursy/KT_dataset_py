# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from math import sqrt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_test = pd.read_csv("../input/dasprodatathon/test.csv")

data_train = pd.read_csv("../input/dasprodatathon/train.csv")
print("Data Test")

display(data_test)

print("Describe Data Test")

display(data_test.describe())
print("Data Train")

display(data_train)

print("Describe Data Train")

display(data_train.describe())
from sklearn import linear_model



#column list

# ID

# Living Area

# Total Area

# Above the Ground Area

# Basement Area

# Neighbors Living Area

# Bedrooms

# Bathrooms

# Floors

# Waterfornt

# View

# Condition

# Grade

# Year Built

# Year Renovated

# Zipcode

# Latitude

# Longtitude

# Price

X = data_train[['Living Area','Total Area','Above the Ground Area','Basement Area','Neighbors Living Area','Bedrooms','Bathrooms','Floors','Waterfront','View','Condition','Grade','Year Built']]

# X = data_train[['Living Area','Total Area','Above the Ground Area','Basement Area','Neighbors Living Area','Bedrooms','Bathrooms','Floors','Waterfront','View','Condition','Grade','Year Built','Year Renovated','Latitude','Longitude']]

y = data_train[['Price']]
fig, ax = plt.subplots(figsize = (15,15))

k = 19

cols = data_train.corr().nlargest(k,'Price')['Price'].index

cm = np.corrcoef(data_train[cols].values.T)

sns.heatmap(cm, annot=True, square=True, fmt='.2f',yticklabels=cols.values, xticklabels=cols.values, ax=ax)
sns.set()

cols = ['Price','Living Area','Total Area','Above the Ground Area','Basement Area','Neighbors Living Area','Bedrooms','Bathrooms','Floors','Waterfront','View','Condition','Grade','Year Built','Year Renovated']

sns.pairplot(data_train[cols], height = 2.5)

plt.show()
regresi = linear_model.LinearRegression()

regresi.fit(X,y)

print(regresi.coef_)
df_train = pd.DataFrame()

df_train['ID'] = data_train['ID']

df_train['Price'] = y

df_train.head()
X_test = data_test[['Living Area','Total Area','Above the Ground Area','Basement Area','Neighbors Living Area','Bedrooms','Bathrooms','Floors','Waterfront','View','Condition','Grade','Year Built']]

# X_test = data_test[['Living Area','Total Area','Above the Ground Area','Basement Area','Neighbors Living Area','Bedrooms','Bathrooms','Floors','Waterfront','View','Condition','Grade','Year Built','Year Renovated','Latitude','Longitude']]

ID_test = data_test['ID']
y_prediksi = regresi.predict(X_test)

print(y_prediksi)
sns.distplot(y[0:len(y_prediksi)]-y_prediksi)

rms = sqrt(mean_squared_error(y[0:len(y_prediksi)], y_prediksi))

print(rms)

# 3573345928.23891

# 6956379711.063916

df_prediksi = pd.DataFrame()

df_prediksi['ID'] = ID_test

df_prediksi['Price'] = y_prediksi

df_prediksi.head()
rms = sqrt(mean_squared_error(y[0:len(y_prediksi)], y_prediksi))

print(rms)
dfSubmit = pd.DataFrame()

dfSubmit['ID'] = ID_test

dfSubmit['Price'] = y_prediksi

dfSubmit.head()
dfSubmit.to_csv('ZAFTestSubmit.csv', index=False)