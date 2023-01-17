# libraries for data wrangling and visualisation are imported

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
zomato = pd.read_csv('../input/zomato-restaurants-in-india/zomato_restaurants_in_India.csv')

zomato.head()
zomato.drop(['res_id','address','city_id','country_id',

             'zipcode','url','latitude','longitude','locality_verbose',

             'currency','opentable_support','delivery','takeaway','timings','rating_text'],axis=1,inplace=True)

zomato.columns
print('No. of features:',zomato.shape[1],'\nNo. of resturants:',zomato.shape[0])
print('No.of duplicate entries:',zomato[zomato.duplicated()].count()[0])

zomato.drop_duplicates(inplace=True)

print('After removal:\nNo. of features:',zomato.shape[1],'\nNo. of resturants:',zomato.shape[0])
zomato.isnull().sum()
zomato.cuisines.fillna('NA',inplace=True)

zomato.cuisines = zomato.cuisines.apply(lambda x : x.split(sep=','))

zomato['establishment'] = zomato.establishment.apply(lambda x : 'NA' if x=='[]' else x[2:-2])
zch = zomato[zomato['city'] == 'Chennai']

print('No. of resturants in Chennai:',zch.shape[0])
rd_type = zch.establishment.value_counts().reset_index().set_index('index')

plt.figure(figsize=(8,8))

sns.barplot(x=rd_type.index,y=rd_type.establishment)

plt.xlabel('Restaurant Type')

plt.ylabel('Count')

plt.xticks(rotation='vertical')
rd_loc = zch.locality.value_counts().head(20).reset_index().set_index('index') 

plt.figure(figsize=(8,8))

sns.barplot(x=rd_loc.index,y=rd_loc.locality)

plt.xlabel('Locality')

plt.ylabel('Number')

plt.xticks(rotation='vertical')
rd_pr = zch.price_range.value_counts().reset_index().set_index('index') 

f,ax = plt.subplots(1,2,figsize=(20,10))

sns.barplot(x=rd_pr.index,y=rd_pr.price_range,ax=ax[1])

plt.xlabel('Price_Range')

plt.xticks(ticks=(0,1,2,3),labels=('Low','Medium','High','Very High'))

plt.ylabel('Count')

sns.distplot(zch.average_cost_for_two,ax=ax[0],color='r')
zch_rat = zch[zch.aggregate_rating != 0]

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.distplot(zch_rat.aggregate_rating,bins=20,kde=True,color='r',ax=ax[0])

sns.boxplot(zch_rat.aggregate_rating,ax=ax[1],color='r',saturation=0.5)
lis = []

for i in range(0,zch.shape[0]):

    for j in zch.iloc[i,4]:

        lis.append(j)

for k in range(0,len(lis)):

    lis[k] = lis[k].strip()

    

from collections import Counter

cuisine_count = Counter(lis)



from wordcloud import WordCloud

wc = WordCloud(background_color='white')

wc.generate_from_frequencies(cuisine_count)

plt.figure(figsize=(12,8))

plt.imshow(wc,interpolation='bilinear')

plt.axis('off')

plt.show()
import squarify

plt.figure(figsize=(18,10))

squarify.plot(sizes=zch.locality.value_counts().head(40),label=zch.locality.value_counts().head(40).index,

              color=sns.color_palette('RdYlGn_r',52))

plt.axis('off')
rd = pd.crosstab(zch.locality,zch.establishment).loc[['Nungambakkam', 'T. Nagar', 'Anna Nagar East', 'Velachery', 

       'Adyar','Anna Nagar West', 'Alwarpet', 'Phoenix Market City, Velachery','Besant Nagar', 'Mylapore','Kilpauk', 

       'Thuraipakkam', 'Porur','Egmore', 'Mogappair', 'Ashok Nagar', 'Purasavakkam', 'Ramapuram',

       'Forum Vijaya Mall,Vadapalani', 'Ambattur'],[ 'Casual Dining', 'Quick Bites', 'Dessert Parlour', 'Café',

       'Beverage Shop', 'Fine Dining', 'Bakery', 'Mess', 'Kiosk', 'Bar','Sweet Shop', 'Food Court', 'NA', 'Pub', 

       'Lounge']].T

f,ax=plt.subplots(1,2,figsize=(20,10))

rd.plot(kind='bar',stacked=True,ax=ax[0],)

ax = (rd.div(rd.sum(1), axis=0).plot(kind='bar',stacked=True,ax=ax[1]))

ax.legend('off')
tp = pd.crosstab(zch.establishment,zch.price_range).loc[[ 'Casual Dining', 'Quick Bites', 'Dessert Parlour', 'Café',

       'Beverage Shop', 'Fine Dining', 'Bakery', 'Mess', 'Kiosk', 'Bar','Sweet Shop', 'Food Court', 'NA', 'Pub', 

       'Lounge'],:]

f,ax=plt.subplots(1,2,figsize=(20,10))

tp.plot(kind='bar',stacked=True,ax=ax[0],)

ax = (tp.div(tp.sum(1), axis=0).plot(kind='bar',stacked=True,ax=ax[1]))
rp = pd.crosstab(zch.locality,zch.price_range).loc[['Nungambakkam', 'T. Nagar', 'Anna Nagar East', 'Velachery', 

       'Adyar','Anna Nagar West', 'Alwarpet', 'Phoenix Market City, Velachery','Besant Nagar', 'Mylapore', 

       'Kilpauk', 'Thuraipakkam', 'Porur','Egmore', 'Mogappair', 'Ashok Nagar', 'Purasavakkam', 'Ramapuram',

       'Forum Vijaya Mall, Vadapalani', 'Ambattur']]

f,ax=plt.subplots(1,2,figsize=(20,10))

rp.plot(kind='bar',stacked=True,ax=ax[0],)

cost_loc = zch.groupby('locality')['average_cost_for_two'].median().reset_index().set_index('locality').loc[['Nungambakkam', 

        'T. Nagar', 'Anna Nagar East', 'Velachery', 'Adyar',

       'Anna Nagar West', 'Alwarpet', 'Phoenix Market City, Velachery',

       'Besant Nagar', 'Mylapore', 'Kilpauk', 'Thuraipakkam', 'Porur',

       'Egmore', 'Mogappair', 'Ashok Nagar', 'Purasavakkam', 'Ramapuram',

       'Forum Vijaya Mall, Vadapalani', 'Ambattur']]

sns.barplot(x=cost_loc.index,y=cost_loc.average_cost_for_two,ax=ax[1])

plt.xticks(rotation='vertical')

plt.ylabel('Median cost for two')
rat_loc = zch_rat.groupby('locality')['aggregate_rating'].median().reset_index().set_index('locality').loc[['Nungambakkam', 'T. Nagar', 'Anna Nagar East', 'Velachery', 'Adyar',

       'Anna Nagar West', 'Alwarpet', 'Phoenix Market City, Velachery',

       'Besant Nagar', 'Mylapore', 'Kilpauk', 'Thuraipakkam', 'Porur',

       'Egmore', 'Mogappair', 'Ashok Nagar', 'Purasavakkam', 'Ramapuram',

       'Forum Vijaya Mall, Vadapalani', 'Ambattur']]

plt.figure(figsize=(10,10))

sns.barplot(x=rat_loc.index,y=rat_loc.aggregate_rating)

plt.xticks(rotation='vertical')

plt.ylabel('Median rating')
zch_rat = zch[zch.aggregate_rating != 0]

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.boxplot(x='price_range',y='aggregate_rating',data=zch_rat,ax=ax[1])

sns.lineplot(x=zch_rat.aggregate_rating,y=zch_rat.average_cost_for_two,ax=ax[0])

plt.xticks(ticks=(0,1,2,3),labels=('Low','Medium','High','Very High'))
plt.figure(figsize=(10,10))

sns.boxplot(x='establishment',y='aggregate_rating',data=zch_rat)

plt.xticks(rotation='vertical')