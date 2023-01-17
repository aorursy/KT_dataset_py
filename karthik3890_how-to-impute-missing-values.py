import pandas as pd
import numpy as np
cars = pd.read_csv('../input/imports-85.data.txt') # imported the dataset
cars.head() # shows the first five rows which helps to undertsand how data is presented
columnnames=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 
           'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
            'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
# we have updated the column names from the link given by you and it also given us ? indicates missing value

cars = pd.read_csv('../input/imports-85.data.txt',names=columnnames)

cars=cars.replace("?",np.nan)
cars.isnull().sum()

# The below are the missing values in the data set
cars["num-of-doors"].value_counts() # just wanted to check how many times four accounts and two accounts
convert = {"num-of-doors": {"four": 4, "two": 2}}
cars.replace(convert, inplace=True) # here we are converting the categorical one to numerical
cars.head()
to_drop = ["symboling", "make", "fuel-type", "aspiration", "body-style", "drive-wheels", "engine-location", "engine-type", "num-of-cylinders", "fuel-system", "engine-size"]

cars = cars.drop(to_drop, axis=1)

# here we can drop columns which are not continous for effective visulization using BAR chart in below
cars = cars.astype("float")
cars.isnull().sum()
cars.describe()
import missingno as msno
import matplotlib.pyplot as plt
msno.bar(cars, figsize=(10, 5), fontsize=10, color='cyan')
cars["bore"]=cars["bore"].fillna(cars["bore"].mean())
cars["stroke"]=cars["stroke"].fillna(cars["stroke"].mean())
cars["num-of-doors"]=cars["num-of-doors"].fillna(cars["num-of-doors"].median())
cars["horsepower"]=cars["horsepower"].fillna(cars["horsepower"].mean())
cars["peak-rpm"]=cars["peak-rpm"].fillna(cars["peak-rpm"].mean())
cars.isnull().sum()
from fancyimpute import KNN    
X_filled_knn = KNN(k=3).complete(cars[['horsepower', 'peak-rpm', 'price']])
X_filled_knn = KNN(k=3).complete(cars[['horsepower', 'peak-rpm', 'price']])
X_filled_knn = pd.DataFrame(X_filled_knn, columns = ['horsepower', 'peak-rpm', 'price'])
cars['price'] = np.round(X_filled_knn['price'], 0)
cars.isnull().sum()