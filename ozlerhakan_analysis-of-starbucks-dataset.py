# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# seaborn
import seaborn as sns

# matplotlib
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline

sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
pd.__version__
np.__version__
sns.__version__
plotly.__version__
matplotlib.__version__
df = pd.read_csv('../input/store-locations/directory.csv')
df.head()
def update_column(column):
    return column.replace(' ', '_').lower()
starbucks = df.copy()
starbucks.columns = starbucks.columns.map(update_column)
starbucks.head()
starbucks.info()
starbucks.dropna(axis=0, subset=['longitude', 'latitude'], inplace=True)
starbucks.dropna(axis=1, inplace=True)
starbucks.info()
starbucks.ownership_type.unique()
starbucks.country.unique()
country_indices, country_labels = starbucks.country.factorize()
country_labels
country_indices
starbucks["country_indice"] = country_indices
countries = pd.read_csv('../input/all-countries-with-their-2-digit-codes/countries.csv', names=['country_name', 'code'])
countries.head()
starbucks = starbucks.merge(countries, left_on='country', right_on='code')
starbucks.drop('code', axis=1, inplace=True)
starbucks.head()
starbucks.plot(kind="scatter", x="longitude", y="latitude", 
               alpha=0.4, 
               figsize=(20,10), 
               c=starbucks.country_indice,
               s=starbucks["country_indice"] * 10 / len(country_labels), label="country",
               cmap=plt.get_cmap("jet"), 
               colorbar=False
)
plt.legend();
plt.figure(figsize=(20,10))

sns.scatterplot(x="longitude", y="latitude", data=starbucks, 
                hue="country", 
                legend=False, 
                palette=sns.color_palette('coolwarm', n_colors=len(country_labels)));
co_lat_long = pd.Series(starbucks.country_name + " <br> "+starbucks.latitude.astype(str) + ' : ' + starbucks.longitude.astype(str))
trace = go.Scattergeo(
    lat=starbucks.latitude, 
    lon=starbucks.longitude, 
    mode='markers',
    hoverinfo = 'text', 
    text=co_lat_long,
    marker = dict( 
        size=2, 
        color= starbucks.country_indice,
        colorscale='jet',
        autocolorscale = False,
    )
)

data = [trace]

layout = dict(
    title = 'Starbucks Locations Worldwide<br>(Hover for locations)',
    showlegend = False, 
    geo = dict(
        showframe=False,
        showland = True,
        showlakes = False,
        showcountries = True,
        showcoastlines=False, 
        showocean = False,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
        projection = dict(
            type = 'natural earth'
        )
    )
)

fig = dict(data = data, layout = layout)
iplot(fig)
first_ten_countries = starbucks.country_name.value_counts()[:10]
first_ten_countries
country_names =  first_ten_countries.index
country_values = first_ten_countries.values
plt.figure(figsize=(15,10))

sns.barplot(x=country_names, y=country_values, palette=sns.color_palette('BuGn', n_colors=10))
plt.xticks(rotation= 45)
plt.xlabel('Countries')
plt.ylabel('Size');
plt.figure(figsize=(20,10))

wordcloud = WordCloud(background_color='white',
                    width=712,
                    height=384)\
            .generate(" ".join(starbucks.country_name.value_counts().index))

plt.imshow(wordcloud)
plt.axis('off');
starbucks.head()
sns.countplot(starbucks.ownership_type, palette=sns.color_palette('Purples'))
plt.ylabel('Number of Ownership')
plt.xlabel('Owenership Type')
plt.title('Ownership Types over Startbucks', color = 'blue', fontsize=15);
list_ten_countries = country_names.tolist()
ten_countries = starbucks.query('country_name in @list_ten_countries')
group_country_ownership = ten_countries.groupby(['country_name', 'ownership_type'])['brand'].count().reset_index()
pivot_country_ownership = pd.pivot_table(group_country_ownership, values='brand', index='country_name', columns='ownership_type', fill_value=0)
pivot_country_ownership
sns.set_palette(sns.color_palette('coolwarm'))
pivot_country_ownership.plot(kind='bar',figsize=(20,10))
plt.xlabel('Countries')
plt.ylabel('Ownership Size')
plt.title('Starbucks Ownership Size Over Ten Countries', fontsize=15);
max_ownership_type = pivot_country_ownership.sum(axis=0)
pivot_country_ownership["Company Owned Ratio"] = pivot_country_ownership["Company Owned"] / max_ownership_type["Company Owned"]
pivot_country_ownership["Franchise Ratio"] = pivot_country_ownership["Franchise"] / max_ownership_type["Franchise"]
pivot_country_ownership["Joint Venture Ratio"] = pivot_country_ownership["Joint Venture"] / max_ownership_type["Joint Venture"]
pivot_country_ownership["Licensed Ratio"] = pivot_country_ownership["Licensed"] / max_ownership_type["Licensed"]
fig, ax = plt.subplots(figsize=(20,10))

sns.barplot(x=pivot_country_ownership["Company Owned Ratio"], y=pivot_country_ownership.index, color='green',alpha = 0.8,label='Company Owned')
sns.barplot(x=pivot_country_ownership["Franchise Ratio"], y=pivot_country_ownership.index, color='blue',alpha = 0.5,label='Franchise')
sns.barplot(x=pivot_country_ownership["Joint Venture Ratio"], y=pivot_country_ownership.index, color='red',alpha = 0.5,label='Jount Venture')
sns.barplot(x=pivot_country_ownership["Licensed Ratio"], y=pivot_country_ownership.index, color='yellow',alpha = 0.5,label='Licensed')

ax.legend(loc='center right',frameon = True)
ax.set(xlabel='Ownership Size Ratio', ylabel='Countries',title = "Ratio For Each Starbucks Ownership Over Type Ten Countries");
starbucks.brand.unique()
brands = ['Teavana', 'Evolution Fresh', 'Coffee House Holdings']
result = starbucks.query('brand in @brands')
result.shape
result.ownership_type.unique()
result.country_name.unique()
