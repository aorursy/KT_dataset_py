# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

data_source = check_output(["ls", "../input"]).decode("utf8")
print(data_source)



df_beer = pd.read_csv("../input/beersdata/beers.csv", sep=",",header=0)
df_brew = pd.read_csv("../input/breweries/breweries.csv",sep=",",header=0)

df_brew['brewery_id'] = df_brew.index

df = df_beer.merge(df_brew, on="brewery_id")
print(df.shape)



# Any results you write to the current directory are saved as output.
print(df.head())

print(df.describe())
print(df.corr)

plot = df.state.value_counts().plot(kind='bar', title="No of Breweries in Each State", \
                             figsize=(10,8), colormap='summer')
plot.set_xlabel('State')
plot.set_ylabel('Number of Breweries')


plot = df.state.value_counts().plot(kind='bar', title="No of Breweries in Each State", \
                             figsize=(7,8), colormap='summer')
plot.set_xlabel('State')
plot.set_ylabel('Number of Breweries')
mean_line = plot.axhline(df.state.value_counts().mean(), color='r',\
                         label='Avg. Number of Breweries')

plot2 = df.groupby('style')['abv'].mean().nlargest(15).plot(kind='bar', \
               title='Beer Styles with Highest Avg Alcohol by Volume', \
               colormap='summer')
plot2.set_ylabel('Average % Alcohol Brewed')

f,ax = plt.subplots(figsize=(18, 18))
sn.heatmap(df_beer.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plot = df_beer.groupby('style')['name'].count().nlargest(15).plot(kind='bar', \
               title='Most Brewed Beer Styles', \
               colormap='summer'  )

plot.set_ylabel('No. of Different Beers')

