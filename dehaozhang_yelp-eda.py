import pandas as pd

import numpy as np

import json

import os

import re

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from wordcloud import WordCloud

from nltk.corpus import stopwords
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
business = []

with open('../input/yelp-dataset/yelp_academic_dataset_business.json') as fl:

    for i, line in enumerate(fl):

        business.append(json.loads(line)) #use 'loads' since line is string

df_business = pd.DataFrame(business)



# print the first five rows

df_business.head()
df_business.shape
df_business.columns
df_business['categories'].isna().mean()
df_business = df_business[df_business['categories'].notna()] #take out missing ones

df_business.shape
cat_temp1 = ';'.join(df_business['categories'])

cat_temp2 = re.split(';|,', cat_temp1)

bus_cat_trim = [item.lstrip() for item in cat_temp2]

df_bus_cat = pd.DataFrame(bus_cat_trim,columns=['category'])
bus_cat_count = df_bus_cat.category.value_counts()

bus_cat_count = bus_cat_count.sort_values(ascending = False)

bus_cat_count = bus_cat_count.iloc[0:10]



# plot

fig = plt.figure(figsize=(10, 6))

ax = sns.barplot(bus_cat_count.index, bus_cat_count.values)

plt.title("Top Business Categories",fontsize = 20)

x_locs,x_labels = plt.xticks()

plt.setp(x_labels, rotation = 60)

plt.ylabel('Number of Businesses', fontsize = 12)

plt.xlabel('Category', fontsize = 12)



#text labels

r = ax.patches

labels = bus_cat_count.values

for rect, label in zip(r, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')
df_bus_res = df_business.loc[[i for i in df_business['categories'].index if re.search('Restaurants', df_business['categories'][i])]]
fig = plt.figure(figsize=(10, 6))

plt.title("Geographic View of Restaurant Locations",fontsize = 20)

m=Basemap(projection='cyl', lon_0 = 0, lat_0=0, resolution='c')

m.fillcontinents(color='#FAFFCA',lake_color='#003875')

m.drawmapboundary(fill_color='#003875') 

m.drawcountries(linewidth=0.2, color="black")

m_coords = m(df_bus_res["longitude"].tolist(), df_bus_res["latitude"].tolist())

m.scatter(m_coords[0], m_coords[1], s=5, c='red', lw=3, zorder=5);
# coordinates range for North America

lon_min, lon_max = -150, -50

lat_min, lat_max = 10,60



plt.figure(figsize=(10,6))

m1 = Basemap(projection='merc', llcrnrlat=lat_min, urcrnrlat=lat_max,

             llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='c')

plt.title("North America Region", fontsize = 20)

m1.fillcontinents(color='#FAFFCA',lake_color='#003875')

m1.drawmapboundary(fill_color='#003875')    

m1.drawcountries(linewidth=0.5, color="black") 

m1_coords = m1(df_bus_res["longitude"].tolist(), df_bus_res["latitude"].tolist())

m1.scatter(m1_coords[0], m1_coords[1], s=20, c="red", lw=5, zorder=5);
df_bus_res['city_state'] = df_bus_res['city'] + ',' + df_bus_res['state']
city_res_count = df_bus_res.city_state.value_counts()

city_res_count = city_res_count.sort_values(ascending = False)

city_res_count = city_res_count.iloc[0:10]



# plot

fig = plt.figure(figsize=(10, 6))

ax = sns.barplot(city_res_count.index, city_res_count.values)

plt.title("Top Cities",fontsize = 20)

x_locs,x_labels = plt.xticks()

plt.setp(x_labels, rotation = 60)

plt.ylabel('Number of Restaurants', fontsize = 12)

plt.xlabel('City,State', fontsize = 12)



#text labels

r = ax.patches

labels = city_res_count.values

for rect, label in zip(r, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))



#coordinates for Toronto (Source: Google)

lat_t = 43.6532

lon_t = -79.3832



#rectangular view

lon_t_min, lon_t_max = lon_t-0.5,lon_t+0.5

lat_t_min, lat_t_max = lat_t-0.5,lat_t+0.5



#subset the data

df_res_tor=df_bus_res[(df_bus_res["longitude"]>lon_t_min) & (df_bus_res["longitude"]<lon_t_max) &\

                    (df_bus_res["latitude"]>lat_t_min) & (df_bus_res["latitude"]<lat_t_max)]



#plot

df_res_tor.plot(kind='scatter', x='longitude', y='latitude',

                color='#52fff3', s=0.02, alpha=.6, subplots=True, ax=ax1)

ax1.set_title("Restaurants in Toronto")

ax1.set_facecolor('black')



#coordinates for Las Vegas (Source: Google)

lat_v = 36.1699

lon_v = -115.1398



#rectangular view

lon_v_min, lon_v_max = lon_v-0.5,lon_v+0.5

lat_v_min, lat_v_max = lat_v-0.5,lat_v+0.5



#subset the data

df_res_lv=df_bus_res[(df_bus_res["longitude"]>lon_v_min) & (df_bus_res["longitude"]<lon_v_max) &\

                    (df_bus_res["latitude"]>lat_v_min) & (df_bus_res["latitude"]<lat_v_max)]



#plot

df_res_lv.plot(kind='scatter', x='longitude', y='latitude',

                color='#52fff3', s=.02, alpha=.6, subplots=True, ax=ax2)

ax2.set_title("Restaurants in Las Vegas")

ax2.set_facecolor('black')



f.tight_layout(pad=5.0);
res_count = df_bus_res.name.value_counts()

res_count = res_count.sort_values(ascending = False)

res_count = res_count.iloc[0:15]



# plot

fig = plt.figure(figsize=(10, 6))

ax = sns.barplot(res_count.index, res_count.values)

plt.title("Restaurants with High Occurences",fontsize = 20)

x_locs,x_labels = plt.xticks()

plt.setp(x_labels, rotation = 60)

plt.ylabel('Number of Restaurants', fontsize = 12)

plt.xlabel('Restaurant', fontsize = 12)



#text labels

r = ax.patches

labels = res_count.values

for rect, label in zip(r, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')
df_bus_res.loc[df_bus_res.name == 'Subway', 'name'] = 'Subway Restaurants'
res_count = df_bus_res.name.value_counts()

res_count = res_count.sort_values(ascending = False)

res_count = res_count.iloc[0:15]



# plot

fig = plt.figure(figsize=(10, 6))

ax = sns.barplot(res_count.index, res_count.values)

plt.title("Restaurants with High Occurences",fontsize = 20)

x_locs,x_labels = plt.xticks()

plt.setp(x_labels, rotation = 60)

plt.ylabel('Number of Restaurants', fontsize = 12)

plt.xlabel('Restaurant Name', fontsize = 12)



#text labels

r = ax.patches

labels = res_count.values

for rect, label in zip(r, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 10, label, ha='center', va='bottom')
#subset data

sub_mb = df_bus_res.loc[(df_bus_res.name == 'McDonald\'s') | (df_bus_res.name == 'Burger King')]

sub_pd = df_bus_res.loc[(df_bus_res.name == 'Pizza Hut') | (df_bus_res.name == 'Domino\'s Pizza')]

sub_sj = df_bus_res.loc[(df_bus_res.name == 'Subway Restaurants') | (df_bus_res.name == 'Jimmy John\'s')]

sub_tc = df_bus_res.loc[(df_bus_res.name == 'Taco Bell') | (df_bus_res.name == 'Chipotle Mexican Grill')]
fig = plt.figure(figsize=(8, 6))

sns.boxplot(x = 'name', y = 'stars', data = sub_mb)

plt.title("Rating Comparison: MD vs BK",fontsize = 20)

plt.ylabel('Stars', fontsize = 12)

plt.xlabel('Restaurant Name', fontsize = 12);
fig = plt.figure(figsize=(8, 6))

sns.boxplot(x = 'name', y = 'stars', data = sub_pd)

plt.title("Rating Comparison: PH vs DP",fontsize = 20)

plt.ylabel('Stars', fontsize = 12)

plt.xlabel('Restaurant Name', fontsize = 12);
fig = plt.figure(figsize=(8, 6))

sns.boxplot(x = 'name', y = 'stars', data = sub_sj)

plt.title("Rating Comparison: SR vs JJ",fontsize = 20)

plt.ylabel('Stars', fontsize = 12)

plt.xlabel('Restaurant Name', fontsize = 12);
fig = plt.figure(figsize=(8, 6))

sns.boxplot(x = 'name', y = 'stars', data = sub_tc)

plt.title("Rating Comparison: TB vs CG",fontsize = 20)

plt.ylabel('Stars', fontsize = 12)

plt.xlabel('Restaurant Name', fontsize = 12);
df_bus_res.loc[df_bus_res.name == 'Chipotle Mexican Grill'].stars.median()
fig = plt.figure(figsize=(8, 6))

sns.scatterplot(x = 'stars', y = 'review_count', data = df_bus_res)

plt.title("Reviews vs Rating",fontsize = 20)

plt.ylabel('Number of Reviews', fontsize = 12)

plt.xlabel('Rating', fontsize = 12);
df_bus_res['attributes'] = df_bus_res['attributes'].apply(lambda x: {} if x is None else x)

df_att = pd.json_normalize(df_bus_res.attributes)
df_att.head()
# attributes of restaurants

df_att.columns
df_att.BusinessAcceptsBitcoin.value_counts()
# price range vs coatcheck

pd.crosstab(df_att.RestaurantsPriceRange2, df_att.CoatCheck, margins=True, normalize = 'index')
# stars vs delivery

pd.crosstab(df_bus_res.stars, df_att.RestaurantsDelivery, margins=True, normalize = 'index')
df_tor = df_bus_res.loc[df_bus_res['city'] == 'Toronto']



# open restaurants more than 100 reviews with rating above 3.5, accepts takeout, credit cards, and price range of $$.

crit = (df_tor['stars'] > 3.5) & (df_tor['review_count'] > 100) & (df_tor['is_open'] == 1) & (df_att.RestaurantsTakeOut == 'True') & (df_att.RestaurantsPriceRange2 == '2') & (df_att.BusinessAcceptsCreditCards == 'True')

df_tor_sub = df_tor.loc[crit]

fig = plt.figure(figsize=(12, 6))

sns.barplot(x = 'name', y = 'stars', data = df_tor_sub.sort_values(by=['stars', 'review_count'], ascending = False)[0:15])

plt.title("Top 15 Restaurants Based on Defined Criteria",fontsize = 20)

x_locs,x_labels = plt.xticks()

plt.setp(x_labels, rotation = 60)

plt.yticks(np.arange(0.0, 5.5, 0.5))

plt.ylabel('Rating', fontsize = 12)

plt.xlabel('Restaurant', fontsize = 12);
df_tor_sub.sort_values(by=['stars', 'review_count'], ascending = False)[0:15]['name']
tips = []

with open('../input/yelp-dataset/yelp_academic_dataset_tip.json') as fl:

    for i, line in enumerate(fl):

        tips.append(json.loads(line))

df_tips = pd.DataFrame(tips)



# print the first five rows

df_tips.head()
df_RI = df_bus_res.loc[(df_bus_res['name'] == 'Ramen Isshin') & crit]

df_RI_tips = df_tips.loc[df_tips['business_id'].isin(df_RI.business_id)]
df_RI_tips.text.values[0:10]
def text_prep(text):

    # filer out non-letters and lowercase them

    text = re.sub('[^a-z\s]', '', text.lower())

    # remove stopwords

    text = [w for w in text.split() if w not in stopwords.words('english')]

    return ' '.join(text)
pd.set_option('mode.chained_assignment', None)

df_RI_tips['text_cl'] = df_RI_tips['text'].apply(text_prep)
wc = WordCloud(width=1600, height=800, random_state=42, max_words=1000000)



# generation

wc.generate(str(df_RI_tips['text_cl']))



plt.figure(figsize=(15,10), facecolor='black')

plt.title("Tips on Ramen Isshin", fontsize=40, color='white')

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.tight_layout(pad=10)
df_FFC = df_bus_res.loc[(df_bus_res['name'] == 'Fresco\'s Fish & Chips') & crit]

df_FFC_tips = df_tips.loc[df_tips['business_id'].isin(df_FFC.business_id)]
df_FFC_tips.text.values[0:10]
pd.set_option('mode.chained_assignment', None)

df_FFC_tips['text_cl'] = df_FFC_tips['text'].apply(text_prep)



wc = WordCloud(width=1600, height=800, random_state=42, max_words=20000)



# generation

wc.generate(str(df_FFC_tips['text_cl']))



plt.figure(figsize=(15,10), facecolor='black')

plt.title("Tips on Fresco\'s Fish & Chips", fontsize=40, color='white')

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.tight_layout(pad=10)