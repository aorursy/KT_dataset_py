                                           # Pandas Excercise 1 
import numpy as np

import pandas as pd
# Reading the dataset
df = pd.read_csv("../input/Automobile_data.csv")
# Q1
df.head()
df.tail()
# Q2
df.replace(['?','n.a'],np.nan)
# Q3
df[['company','price']][df.price == df['price'].max()]
# Q4
df[df['company']=='toyota']
# Q5
df.company.value_counts()
# Q6
car_manufactures = df.groupby('company')
highest_price_by_company = car_manufactures['company','price'].max()
highest_price_by_company
# Q7
avg_mileage = car_manufactures['company','average-mileage'].mean()
avg_mileage
# Q8
df.sort_values('price',ascending=False)
# We can replace NaN values in price by minimium price if we want to work on sorted data
# Q9




GermanCars = {'Company': ['Ford', 'Mercedes', 'BMV', 'Audi'], 'Price': [23845, 171995, 135925 , 71400]}

carsDf1 = pd.DataFrame.from_dict(GermanCars)



japaneseCars = {'Company': ['Toyota', 'Honda', 'Nissan', 'Mitsubishi '], 'Price': [29995, 23600, 61500 , 58900]}

carsDf2 = pd.DataFrame.from_dict(japaneseCars)



carsDf = pd.concat([carsDf1, carsDf2], keys=["Germany", "Japan"])

carsDf
# Q10




Car_Price = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'Price': [23845, 17995, 135925 , 71400]}

carPriceDf = pd.DataFrame.from_dict(Car_Price)



car_Horsepower = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'horsepower': [141, 80, 182 , 160]}

carsHorsepowerDf = pd.DataFrame.from_dict(car_Horsepower)



carsDf = pd.merge(carPriceDf, carsHorsepowerDf, on="Company")

carsDf
# Practised Pandas library from https://pynative.com/python-pandas-exercise/   Thank You. 