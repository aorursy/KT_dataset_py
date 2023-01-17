import numpy as np

import plotly.express as px

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



car_data=pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv")

car_data.head()

#It function gives us 5 line at the top in the data.
car_data.describe().T

#Some statistical explanation. Do you know Box Plot ? 

# It displays %25 is first quartile. 

# %50 is median and gives at the middle value when you ascending data.

# %75 is third quartile.

#min-max value is value that gives at lowest value and at highest value.
car_data.columns

#Columns that we have in data.
car_data.dtypes

#we can see what's the type of columns.
#missing data

car_data.isnull().sum().sort_values(ascending=False)
median_price = car_data['price'].median()

car_data['price'] = car_data['price'].astype(int)

car_data['price'].replace(0,median_price ,inplace=True)
brand_of_car = car_data.groupby('brand')['model'].count().reset_index().sort_values('model',ascending = False).head(10)

brand_of_car = brand_of_car.rename(columns = {'model':'count'})

fig = px.bar(brand_of_car, x='brand', y='count', color='count')

fig.show()



#You can reach a lot of information about car brand and their count 
expensive_cars = car_data.sort_values('price',ascending = False).head(2)

fig = px.bar(expensive_cars, x='brand', y='price', color='price')

fig.show()

#We saw which car brand is expensive in this vis.
sns.swarmplot(x="price", y="title_status" ,data=car_data);

plt.show()

#I want to display relationship between clean car and salvage insurance status with price.
data = car_data[['price','year']]

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.regplot(x='year', y="price", data=data)
# Pair Plots

# Plot pairwise relationships in a dataset



cont_col= ['year','price','mileage']

sns.pairplot(car_data[cont_col],  kind="reg", diag_kind = "kde"   )

plt.show()