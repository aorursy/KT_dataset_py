import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

# We can override the default matplotlib styles with those of Seaborn

import seaborn as sns

sns.set()

#import os

#print(os.listdir("../input"))

# Load the data from a .csv in the same folder

data = pd.read_csv('../input/real_estate_price_size_year.csv')

data.describe()


data.corr()





y=data['price']

x1 = data [['size','year']]

x=sm.add_constant(x1)

result=sm.OLS(y,x).fit()

result.summary()
ax = sns.heatmap(

    data.corr(), 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)