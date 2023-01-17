import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
beer=pd.read_csv("../input/beers.csv")

brewery=pd.read_csv("../input/breweries.csv")
beer.head()
brewery.head()
beer.shape
brewery.shape
beer.describe()
brewery.describe()
df=brewery.groupby('state').count()

df1=df.sort(columns='name',axis=0, ascending=False)

df1.head()
cities=brewery.groupby('city').count()

cities1=cities.sort(columns='name', ascending=False)



cities1.head()
craft_beer=beer.groupby('style').count()

craft_beer1=craft_beer.sort(columns='id', ascending=False)

craft_beer1.head()
brewery['brewery_id']=brewery.index

brewery.head()
new_data=beer.merge(brewery, on="brewery_id")

new_data.head()
new_data=new_data.rename(index=str, columns={"name_x":"beer_name", "name_y":"brewery_name"})

new_data.head()
new_data=new_data.drop(['Unnamed: 0_x', 'Unnamed: 0_y'], axis=1)

new_data.head()
plt.figure(figsize=(9,6))

plot1=new_data.state.value_counts().plot(kind='bar', title='Breweries by state')



plot1.set_xlabel('state')
plt.figure(figsize=(9,6))



plot1=new_data.groupby('city')['brewery_name'].count().nlargest(10).plot(kind='bar',title='Cities with most breweries')



plot1.set_ylabel('Number of breweries')
plt.figure(figsize=(9,6))

plot2=new_data.groupby('style')['beer_name'].count().nlargest(10).plot(kind='bar',title='Popular brewed beer styles')

plot2.set_ylabel('Number of Different Beers')