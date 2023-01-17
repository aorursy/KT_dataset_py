#Let's start with adding libraries I will use



import numpy as np 

import seaborn as sns



# plotly

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt



from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

gps = pd.read_csv("../input/googleplaystore.csv")



gps.head()
gps.info()
gps["Size"] = gps["Size"].str.replace("M", "")

gps.Size.replace("M"," ")

gps["Size"] = gps["Size"].str.replace("k", "")

gps.Size.replace("k"," ")

gps["Size"] = gps["Size"].str.replace(",", "")

gps.Size.replace(",",".")

gps["Reviews"] = gps["Reviews"].str.replace("M", "")

gps.Reviews.replace("M","000000000")
gps['Size'] = gps['Size'].astype(float)

gps['Reviews'] = gps['Reviews'].astype(float)

gps['Price'] = gps['Price'].astype(float)
gps.info()
Size_mean = gps.Size.mean()

gps["Size_level"] = ["Big" if Size_mean < each else "Small" for each in gps.Size]



Price_mean = gps.Price.mean()

gps["Price_level"] = ["Expensive" if Price_mean < each else "Cheap" for each in gps.Price]

print(gps.Category.unique())
print(gps.describe())
bad = gps[(gps['Rating']>0.9) & (gps['Rating']<2.0)]

normal = gps[(gps['Rating']>2.0) & (gps['Rating']<3.0)]

good = gps[(gps['Rating']>3.0) & (gps['Rating']<4.0)]

amazing = gps[(gps['Rating']>4.0) & (gps['Rating']<5.1)]

bad.info()

normal.info()

good.info()

amazing.info()
# ı want to see rating bigger than 4.5 and size bigger than 35 and price lower than 1

filter1 = gps.Rating > 4.5

filter2 = gps.Size > 35

filter3 = gps.Price < 1

filtered_data = gps[filter1 & filter2 & filter3]

filtered_data.count()

filtered_data.info()