import numpy as np

import pandas as pd
#1

data = pd.read_csv('../input/sample-autombile-data/Automobile_data.csv')

data.head()
data.tail()
#2

data=pd.read_csv('../input/sample-autombile-data/Automobile_data.csv',na_values={

    "company":["?","n.a"],

    "body-style":["?","n.a"],

    "wheel-base":["?","n.a"],

    "length":["?","n.a"],

    "engine-type":["?","n.a"],

    "num-of-cylinders":["?","n.a"],

    "horsepower":["?","n.a"],

    "average-mileage":["?","n.a"],

    "price":["?","n.a"]})

data
#3 

data[data.price==max(data.price)][["company","price"]]
#4 

data[data.company=="toyota"]
#5

data['company'].value_counts() #data.company.value_counts()
data.company.value_counts()
#6

data.groupby(["company"])["company","price"].max()
#7

data.groupby(["company"])["company","average-mileage"].mean()
#8

data.sort_values(by="price",ascending=False)
#9

GermanCars = {'Company': ['Ford', 'Mercedes', 'BMV', 'Audi'], 'Price': [23845, 171995, 135925 , 71400]}

japaneseCars = {'Company': ['Toyota', 'Honda', 'Nissan', 'Mitsubishi '], 'Price': [29995, 23600, 61500 , 58900]}
germandf=pd.DataFrame(GermanCars)

germandf
japandf=pd.DataFrame(japaneseCars)

japandf
data1=pd.concat([germandf,japandf],keys=["Germany","Japan"],axis=0)

data1
#10

Car_Price = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'Price': [23845, 17995, 135925 , 71400]}

car_Horsepower = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'horsepower': [141, 80, 182 , 160]}

dfprice=pd.DataFrame(Car_Price)

dfHorseporwer=pd.DataFrame(car_Horsepower)
df=pd.merge(dfprice,dfHorseporwer,on="Company")

df