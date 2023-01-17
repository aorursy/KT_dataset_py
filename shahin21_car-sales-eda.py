import pandas as pd

from matplotlib import pyplot as plt

import numpy as np

import seaborn as sns
car_data = pd.read_csv("../input/Car_sales.csv")
car_data.info()
car_data.head()
car_data.shape
car_data.describe()
drop_cols = ['Curb weight']

car_data = car_data.drop(drop_cols, axis=1)
car_data.info()
car_data = car_data.drop_duplicates(keep='first')
car_data_sort_sales = car_data.sort_values(by='Sales in thousands' , ascending=False)

car_data_sort_sales.head()
car_data_sort_fuel = car_data.sort_values(by='Fuel efficiency', ascending=False)

car_data_sort_fuel.head()
def maximum_minimum_values(column):

    top = car_data[column].idxmax()

    top_obs = pd.DataFrame(car_data.loc[top])

    

    bottom = car_data[column].idxmin()

    bottom_obs = pd.DataFrame(car_data.loc[bottom])

    

    min_max_obs = pd.concat([top_obs, bottom_obs], axis=1)

    

    return min_max_obs
maximum_minimum_values('Sales in thousands')
num_bins = 50

plt.hist(car_data['Sales in thousands'], num_bins)
sns.distplot(car_data['Sales in thousands'], 15)
make_dist = car_data.groupby('Manufacturer').size()
make_dist
make_dist.plot()
car_numeric = car_data.select_dtypes(include=['float64','int64'])

car_numeric.head()
car_numeric.hist(bins=20)
sns.boxenplot(y="Manufacturer", x='Sales in thousands', data=car_data)