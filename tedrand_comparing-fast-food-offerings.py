import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.misc import imread

import matplotlib.cbook as cbook

%matplotlib inline

sns.set(style="darkgrid")



import requests

from PIL import Image

from io import BytesIO







def make_series_set(df, value, filt, filt_values):

    series_set = []

    for val in filt_values:

        s = df[value][df[filt] == val]

        series_set.append(s)

    return series_set
df = pd.read_csv('../input/menu.csv')

df.head()
pd.unique(df['Category'])
df['Fat Ratio'] = df['Calories from Fat']/df['Calories']

df['Sat Fat Ratio'] = df['Saturated Fat']/df['Calories']
x1 = make_series_set(df, 'Fat Ratio', 'Category', ['Breakfast', 'Beef & Pork', 'Chicken & Fish', 'Snacks & Sides'])



fig, axes = plt.subplots(figsize=(12, 10))

axes.set_facecolor('#aaaaaa')





colors = ['#dd1021', '#ffc300', '#f9f5f5', '#27742d']

axes.hist(

    x1, bins=10, normed=1, color=colors, 

    histtype='bar', label=['Breakfast', 'Beef & Pork', 'Chicken & Fish', 'Snacks & Sides']

)

axes.legend()

axes.set_title('Fat Ratio By Meal Type (Calories from Fat / Calories)')
plt.scatter(data=df, x='Carbohydrates', y='Calories')
plt.scatter(data=df, x='Protein', y='Calories')
plt.scatter(data=df, x='Sugars', y='Calories')
plt.scatter(data=df, x='Total Fat', y='Calories')
