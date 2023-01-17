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
data = pd.read_csv("/kaggle/input/nobel-prize/complete.csv")

print(data.shape)

data.info()
import plotly.express as px

import datetime as dt

from wordcloud import WordCloud

import matplotlib.pyplot as plt
year = data.groupby("awardYear")['awardYear'].count().reset_index(name = 'Count')

fig = px.bar(year, x='awardYear', y='Count')

fig.show()
cate = data.groupby('category')['category'].count().reset_index(name = 'Count')

fig = px.bar(cate, x='category', y = 'Count', color = 'category')

fig.show()
fig = px.histogram(data, x="prizeAmount")

fig.show()
date = data.groupby('dateAwarded')['dateAwarded'].count().reset_index(name = 'count')

fig = px.histogram(date, x ='dateAwarded', color = 'dateAwarded')

fig.show()
#who = data.groupby('laureate_or_org')['laureate_or_org'].count().reset_index(name='count')

#fig = px.pie(who, names = 'laureate_or_org', values='count', color = 'laureate_or_org')

#fig.show()
dd= data.groupby('name')['name'].count().reset_index(name = 'count').sort_values(by='count', ascending = False)

dd.head(10)
port = data.groupby('portion')['portion'].count().reset_index(name = 'count')

fig = px.pie(port, values = 'count', names = 'portion')

fig.show()
gen = data.groupby('gender')['gender'].count().reset_index(name = 'count')

fig = px.pie(gen, names = 'gender', values = 'count')

fig.show()
import folium

m = folium.Map()

url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'

country_shapes = f'{url}/world-countries.json'

country = data.groupby('birth_country')['birth_country'].count().reset_index(name = 'count')

country.replace('USA', "United States of America", inplace = True)

country.replace('Tanzania', "United Republic of Tanzania", inplace = True)

country.replace('Democratic Republic of Congo', "Democratic Republic of the Congo", inplace = True)

country.replace('Congo', "Republic of the Congo", inplace = True)

country.replace('Lao', "Laos", inplace = True)

country.replace('Syrian Arab Republic', "Syria", inplace = True)

country.replace('Serbia', "Republic of Serbia", inplace = True)

country.replace('Czechia', "Czech Republic", inplace = True)

country.replace('UAE', "United Arab Emirates", inplace = True)



folium.Choropleth(

    #The GeoJSON data to represent the world country

    geo_data=country_shapes,

    name='Country wise',

    data=country,

    #The column aceppting list with 2 value; The country name and  the numerical value

    columns=['birth_country', 'count'],

    key_on='feature.properties.name',

    fill_color='PuRd',

    nan_fill_color='white'

).add_to(m)

m
cont = data.groupby("birth_continent")['birth_continent'].count().reset_index(name = 'count')

fig = px.bar(cont, x='birth_continent', y = 'count', color = 'birth_continent')

fig.show()
city = data.groupby('birth_city')['birth_city'].count().reset_index(name = 'count').sort_values(by='count', ascending = False)

fig = px.bar(city[0:20], x='birth_city', y = 'count', color = 'birth_city')

fig.show()
death_city = data.groupby('death_city')['death_city'].count().reset_index(name = 'count').sort_values(by='count', ascending = False)

fig = px.bar(death_city[0:20], x='death_city', y = 'count', color = 'death_city')

fig.show()
#data = pd.read_csv("/kaggle/input/nobel-prize/complete.csv")

data.birth_date = data.birth_date.str.replace('-00','-01')

data.birth_date = pd.to_datetime(data.birth_date)

data['year'] = data.birth_date.dt.year

data['month'] = data.birth_date.dt.month
month = data.groupby('month')['month'].count().reset_index(name = 'count')

fig = px.bar(month, x='month', y = 'count', color = 'month')

fig.show()
year = data.groupby('year')['year'].count().reset_index(name = 'count')

fig = px.bar(year, x='year', y = 'count', color = 'year')

fig.show()
con = data[['birth_country','death_country']]

data['country'] = np.where(con['birth_country']==con['death_country'],"same","diff")

bd_country = data.groupby("country")['country'].count().reset_index(name = 'count')

fig = px.pie(bd_country, values = 'count', names = 'country')

fig.show()
prizeStatus = data.groupby("prizeStatus")['prizeStatus'].count().reset_index(name = 'count')

fig = px.pie(prizeStatus, values = 'count', names = 'prizeStatus')

fig.show()
wordcloud = WordCloud(width=600, height=500, margin=0,background_color="skyblue").generate(data['motivation'].to_string())

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
data['affiliations'] = data['affiliations'].str.lstrip('[\'')

data['affiliations'] = data['affiliations'].str.rstrip('\']')

affiliations = data.groupby('affiliations')['affiliations'].count().reset_index(name = 'count').sort_values(by='count', ascending = False)

fig = px.bar(affiliations[0:10], x='affiliations', y = 'count', color = 'affiliations')

fig.show()
cat_year = data.groupby(['awardYear','category'])['category'].count().reset_index(name = 'count')

fig = px.bar(cat_year, x='awardYear', y = 'count', color = 'category')

fig.show()
gen_year = data.groupby(['awardYear','gender'])['category'].count().reset_index(name = 'count')

fig = px.bar(gen_year, x='awardYear', y = 'count', color = 'gender')

fig.show()
gen_cate = data.groupby(['category','gender'])['category'].count().reset_index(name = 'count')

fig = px.bar(gen_cate, x='category', y = 'count', color = 'gender')

fig.show()
cat_city = data.groupby(['birth_city','category'])['category'].count().reset_index(name = 'count')

fig = px.bar(cat_city, x='count', y = 'category', color = 'birth_city')

fig.show()
cat_dcity = data.groupby(['death_city','category'])['category'].count().reset_index(name = 'count')

fig = px.bar(cat_dcity, x='count', y = 'category', color = 'death_city')

fig.show()
cat_dcnty = data.groupby(['death_country','category'])['category'].count().reset_index(name = 'count')

fig = px.bar(cat_dcnty, x='count', y = 'category', color = 'death_country')

fig.show()
cat_bcnty = data.groupby(['birth_country','category'])['category'].count().reset_index(name = 'count')

fig = px.bar(cat_bcnty, x='count', y = 'category', color = 'birth_country')

fig.show()
org_founded_country = data.groupby('org_founded_country')['org_founded_country'].count().reset_index(name = 'count')

fig = px.bar(org_founded_country, x='org_founded_country', y = 'count', color = 'org_founded_country')

fig.show()
cat_bcnty = data.groupby(['category','prizeAmount'])['prizeAmount'].count().reset_index(name = 'count')

fig = px.bar(cat_bcnty, x='count', y = 'category', color = 'prizeAmount')

fig.show()
year_amt = data.groupby(['awardYear','prizeAmount'])['prizeAmount'].count().reset_index(name = 'count')

fig = px.bar(year_amt, x='awardYear', y = 'count', color = 'prizeAmount')

fig.show()