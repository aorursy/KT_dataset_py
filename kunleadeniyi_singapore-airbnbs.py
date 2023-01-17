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
raw_data = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')

raw_data.head()
raw_data.describe(include='all')
raw_data.dtypes
raw_data.isna().sum()
data = raw_data.copy()

data = data.drop(['last_review', 'reviews_per_month'], axis=1)
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as ex
print('Neighbourhood Groups: ', data['neighbourhood_group'].unique())

print('Room types: ' ,data['room_type'].unique())
data['neighbourhood'].unique()
plt.figure(figsize=(12,6))

sns.scatterplot(x=data['longitude'], y=data['latitude'], hue=data['neighbourhood_group'])
plt.figure(figsize=(12,6))

sns.scatterplot(x=data['longitude'], y=data['latitude'], hue=data['room_type'])
by_room_type = data.groupby('room_type').agg({'room_type':'count'})
by_room_type
sns.barplot(x=by_room_type.index, y=by_room_type['room_type'])
fig = go.Figure(data=[go.Pie(labels=by_room_type.index, values=by_room_type['room_type'], hole=.5)])

fig.show()
sns.distplot(data['price'])
plt.figure(figsize=(10,7))

sns.catplot(data=data, x='room_type', y='price')



plt.show()
sns.countplot(x=data['room_type'], hue=data['neighbourhood_group'])
average_price_by_roomtype = data.groupby('room_type').agg({'price':'mean'}).sort_values('price', ascending=False)
average_price_by_roomtype
price_quantile_99 = data['price'].quantile(0.99)

print (price_quantile_99)
new_data = data[data['price']<price_quantile_99]

new_data.describe()
sns.distplot(new_data['price'])
plt.figure(figsize=(8,6))

sns.boxplot(y=new_data['price'], x=new_data['neighbourhood_group'])
# sorting by number of reviews

sorted_new_data=new_data.sort_values(by=['number_of_reviews'],ascending=False).head(1000)

sorted_new_data.head()
plt.figure(figsize=(15,8))

plt.scatter(x=new_data['longitude'], y=new_data['latitude'], c=new_data['availability_365'], edgecolor='black', linewidth=1\

            , alpha=1)

cbar = plt.colorbar()
# drop irrelevant features

model_data = new_data.copy()

model_data = model_data.drop(['id','name','host_name','host_id'], axis=1)
model_data.columns
model_data.dtypes
# dealing with categorical variables

from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

number.fit(model_data['neighbourhood'])

model_data['neighbourhood'] = number.transform(model_data['neighbourhood'])



number.fit(model_data['neighbourhood_group'])

model_data['neighbourhood_group'] = number.transform(model_data['neighbourhood_group'])



number.fit(model_data['room_type'])

model_data['room_type'] = number.transform(model_data['room_type'])
model_data.head(50)
# declare features and targets

features = model_data.drop('price', axis=1)

target = model_data['price'] 
# Scale features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(features)

scaled_features = scaler.transform(features)
# Splitting Data sets into training and testing data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaled_features, target, random_state=42, test_size=0.2)
# using linear regression model

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)
# making preditions

predictions = model.predict(x_test)
# Checking accuracy

results = pd.DataFrame({

    'Actual':y_test,

    'Predicted':predictions

})

results.head()
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, predictions)
model.score(x_test, y_test)
new_data['price'].describe()