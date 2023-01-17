# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_beer = pd.read_csv("../input/beers.csv")

data_brew= pd.read_csv("../input/breweries.csv")
data_beer.head()

data_brew.head()
data_brew['brewery_id'] = data_brew.index

data_brew.head()
df= data_beer.merge(data_brew, on="brewery_id")

df.head()
df= df.rename(index= str, columns={"name_x": "beer_name", "name_y":"brewery_name" })

df.head()
df = df.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)

#df.head()



df['abv'] = df['abv'] * 100

df.head()
import matplotlib as plt

#states

plot= df.state.value_counts().plot(kind= 'bar', title='breweries per state',

                                  figsize= (8, 6), colormap= 'winter')

plot.set_xlabel('state')

plot.set_ylabel('breweries')

plot.legend()
plot2= df.groupby('city')['brewery_name'].count().nlargest(20).plot(kind= 'bar', title='top 20 breweries in state', colormap='winter')

plot2.set_ylabel('brewery count')

plot2.legend()
plot1 = df.groupby('state')['abv'].mean().sort_values(ascending=False).plot(kind='bar',\

                                                                    title="Average Alcohol by Volume Brewed in each State", \

                                                                    figsize=(8,6), ylim=(5, 7), colormap='winter')

plot1.set_xlabel('State')

plot1.set_ylabel('Average % Alcohol Brewed')

mean_line1 = plot1.axhline(df.abv.mean(), color='r',\

                         label='National Average')

plot1.legend()