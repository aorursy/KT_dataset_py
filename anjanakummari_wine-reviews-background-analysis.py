# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/wine-reviews/winemag-data-'

wine_data = pd.read_csv(path+'130k-v2.csv')

print(wine_data.info())

print("Wine DataFrame shape: {}".format(wine_data.shape))

print("Wine DataFrame Columns: {}".format(wine_data.columns))
wine_data.head()
'''

For the initial analysis of the dataset, the Null values have been ignored. All the Null values have been dropped!!

Find the average price for each point score to analyze if price increases with an increase in points.

'''

wd_price = wine_data.groupby('points')['price'].mean().reset_index()

wd_price.head()
import plotly.express as px



fig = px.scatter(x=wd_price['points'], y=wd_price['price'], title = 'Price Distribution against Points')

fig.show()
wd_variety = wine_data.groupby('variety')['price'].mean().reset_index()

print("The average price of Wine per variety")

wd_variety.head()
wd_variety = wd_variety.sort_values('price', ascending=False)[0:10]



fig = px.pie(names=wd_variety['variety'], values=wd_variety['price'], title='Most expensive Wine varieties')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
wd_country = wine_data.groupby('country')['price'].mean().reset_index()

print("The average price of Wine per country")

wd_country.head()
wd_country = wd_country.sort_values('price', ascending=False)[0:10]



fig = px.pie(names=wd_country['country'], values=wd_country['price'], title='Countries thar produce expensive Wines')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()

wd_variety = wine_data.groupby('variety')['points'].mean().reset_index()

wd_variety.head()
wd_variety = wd_variety.sort_values('points', ascending = False)[0:7]



fig = px.pie(values=wd_variety['points'], names=wd_variety['variety'], title='Wine varieties with most points')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()