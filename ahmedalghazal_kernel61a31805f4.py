import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))





%matplotlib inline

plt.style.use("seaborn")
# Import data #

dpath = '../input/'



diamonddf = pd.read_csv("../input/diamonds/diamonds.csv")
diamonddf.head()
diamonddf.info()
diamonddf.columns

diamonddf.drop(columns=['Unnamed: 0'], inplace=True)

diamonddf.info()
print("Column\tHaveing a missing value?")

cols_missing_vals = diamonddf.isnull().any()

print(cols_missing_vals)

print("The table {0} missing values".format("has" if cols_missing_vals.any() else "hasn't"))

def ploting(column):

    colors = sns.color_palette("deep")

    sns.distplot(diamonddf[column], color = colors[0])

    plt.show()



for col in ['depth', 'table', 'price', 'carat']:

    ploting(col)

    
price_avg = diamonddf['price'].mean()

print("The average diamond price = ", price_avg)

most_feq_price = diamonddf['price'].value_counts(sort=True).max()

print("The most frequent price is", most_feq_price)



plt.scatter(diamonddf['color'],diamonddf['price'])

plt.ylabel("Price")

print("It seems that there is no relation between diamond color and its price")