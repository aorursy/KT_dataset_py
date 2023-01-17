# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output



data_source = check_output(["ls", "../input"]).decode("utf8")

print(data_source)

# Any results you write to the current directory are saved as output.
df_beer = pd.read_csv("../input/beers.csv")

df_brew = pd.read_csv("../input/breweries.csv")



df_brew['brewery_id'] = df_brew.index



df = df_beer.merge(df_brew, on="brewery_id")

print(df.head())
df = df.rename(index=str, columns={"name_x":"beer_name", "name_y":"brewery_name"})



## these 2 columns are just the index as well as the brewery ID repeated

df = df.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)

## Make ABV a percentage for readability in the plots

df['abv'] = df['abv']*100

print(df.head())
plot = df.state.value_counts().plot(kind='bar', title="Number of Breweries in Each State", \

                             figsize=(8,6), colormap='summer')

plot.set_xlabel('State')

plot.set_ylabel('Number of Breweries')

mean_line = plot.axhline(df.state.value_counts().mean(), color='r',\

                         label='Average Number of Breweries')

plt.legend()
plot5 = df.groupby('city')['brewery_name'].count().nlargest(15).plot(kind='bar', \

               title='Cities with the Most Breweries', \

               colormap='summer',  )

plot5.set_ylabel('Number of Breweries')




plot1 = df.groupby('state')['abv'].mean().sort_values(ascending=False).plot(kind='bar',\

                                                                    title="Average Alcohol by Volume Brewed in each State", \

                                                                    figsize=(8,6), ylim=(5, 7), colormap='summer')

plot1.set_xlabel('State')

plot1.set_ylabel('Average % Alcohol Brewed')

mean_line1 = plot1.axhline(df.abv.mean(), color='r',\

                         label='National Average')

plt.legend()
## print(df['style'].nunique())

## output: 99







plot2 = df.groupby('style')['abv'].mean().nlargest(15).plot(kind='bar', \

               title='Beer Styles with Highest Average Alcohol by Volume', \

               colormap='summer', ylim=(7.8,11) )

plot2.set_ylabel('Average % Alcohol Brewed')

plot3 = df.groupby('style')['beer_name'].count().nlargest(15).plot(kind='bar', \

               title='Most Brewed Beer Styles', \

               colormap='summer',  )



plot3.set_ylabel('Number of Different Beers')
