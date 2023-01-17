# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from wordcloud import WordCloud





from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MaxAbsScaler





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pd.set_option('display.max_rows',500)

pd.set_option('display.max_columns',100)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ab_nyc = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
ab_nyc.head(2)
ab_nyc['demand']  = round((365-ab_nyc['availability_365'])/365,2)
ab_nyc.head(2)
ab_nyc.shape
ab_nyc['neighbourhood_group'].unique()
ab_nyc.rename(columns= {'id':'Id', 

                        'name':'Descr', 

                        'host_id':'Host_Id',

                        'host_name':'Host_Name',

                        'neighbourhood_group': 'Borough', 

                        'neighbourhood':'Neighbourhood', 

                        'latitude':'Latitude',

                        'longitude':'Longitude', 

                        'room_type':'Type', 

                        'price':'Price', 

                        'minimum_nights':'Min_nights', 

                        'number_of_rev': 'Num_Reviews', 

                        'last_review':'Recent_review', 

                        'reviews_per_month':'Rating', 

                        'calculated_host_listings_count': 'Listings_Count', 

                        'availability_365':'Availability', 

                        'demand':'Demand'}, inplace = True)
ab_nyc.head(2)
plt.figure(figsize = (10,10))

corr= ab_nyc.drop(['Id', 'Host_Id','Latitude', 'Longitude'], axis = 1).corr(method= 'pearson')

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, 

            annot= True, 

            linewidths=3,

            cmap = matplotlib.cm.magma,

            square = True,

            mask = mask)

ab_nyc.drop(['Latitude', 'Longitude', 'Host_Id', 'Id'],axis = 1).skew()
price_log = np.log(ab_nyc.loc[ab_nyc['Price']!=0,'Price'])
print(price_log.skew())

sns.distplot(price_log, kde=True)
sns.pairplot(ab_nyc.drop(['Latitude', 'Longitude', 'Host_Id', 'Id'],axis = 1),diag_kind = 'scatter', kind = 'reg', hue= 'Type')
# ab_nyc.loc[ab_nyc['Price']==0,'Price'] = 

ab_nyc.groupby('Host_Id')['Price'].mean()
import nltk

descr = ab_nyc['Descr'].str.lower().str.replace('\|-|&|/|!'or'-','').str.cat(sep = ' ')

stopwords = nltk.corpus.stopwords.words('english')

stopwords.extend(('room', 'apartment', 'bedroom', 'private','apt', 'location','brooklyn'))

# descr
plt.figure(figsize = (10,10))

wordcloud = WordCloud(stopwords = stopwords, background_color= 'white', collocations= False).generate(descr)

plt.title('Description of Rooms', fontdict = {'size':20, 'weight':'bold'})

plt.imshow(wordcloud, interpolation= 'bilinear')

plt.tight_layout(pad = 0)

plt.axis('off')

plt.show()
plt.figure(figsize = (15,8))

ax = sns.scatterplot(x = ab_nyc['Latitude'], 

                y = ab_nyc['Longitude'], 

                hue = ab_nyc['Borough'], 

                s =ab_nyc['Price']*0.3,

                palette = 'magma'

               )

ax.set_title('Boroughs of NYC', fontdict = {'size' : 15, 'weight' : 'bold'})

plt.axis('off')
plt.figure(figsize = (15,8))

ax = sns.scatterplot(x = ab_nyc['Latitude'], 

                y = ab_nyc['Longitude'], 

                hue = ab_nyc['Type'], 

                s = ab_nyc['Price']*0.3

               )

ax.set_title('Room Types in Boroughs of NYC', fontdict = {'size' : 15, 'weight' : 'bold'})

plt.axis('off')
plt.figure(figsize = (15,8))

ax = sns.scatterplot(x = ab_nyc['Latitude'], 

                y = ab_nyc['Longitude'], 

                hue = ab_nyc['Neighbourhood'], 

                s = ab_nyc['Price']*0.3

               )

ax.set_title('Neighbourhoods of NYC', fontdict = {'size' : 15, 'weight' : 'bold'})

plt.axis('off')
plt.figure(figsize = (15,10))

ax = sns.barplot(x = ab_nyc.groupby('Borough')['Price'].mean().reset_index()['Borough'],

            y = ab_nyc.groupby('Borough')['Price'].mean().reset_index()['Price'], 

            palette= 'YlOrRd_r')

ax.set_title('Average Price of Boroughs', fontdict = {'size': 15, 'weight':'bold'})

ax.set_xlabel('Boroughs', fontdict = {'size': 15})

ax.set_ylabel('Average Price',fontdict = {'size': 15})

plt.xticks(rotation = 90)

plt.grid(True, color = 'black')

ax = pd.crosstab(ab_nyc['Borough'], ab_nyc['Type']).plot.barh(color = sns.color_palette('magma_r'), width = 0.8, align = 'edge')

ax.set_xlabel('Number of Listings')

ax.set_title("Number of Listings with it's Type in each Borough", fontdict = {'size':15, 'weight':'bold'})

fig = plt.gcf()

fig.set_size_inches(18,6)

plt.grid(True, color = 'black')
ax = pd.pivot_table(ab_nyc, columns= ['Type'], index= ['Borough'], values= 'Price', aggfunc= np.mean).plot.barh(width = 0.8, align = 'edge', color = sns.color_palette('magma'))

ax.set_xlabel('Average Price')

ax.set_title("Average Price of Room Type in each Borough", fontdict = {'size':15, 'weight':'bold'})

fig = plt.gcf()

fig.set_size_inches(18,6)

plt.grid(True, color = 'black')
plt.figure(figsize = (15,10))

ax = sns.barplot(x = ab_nyc.groupby('Host_Name')['Listings_Count'].max().reset_index().sort_values(by = 'Listings_Count', ascending = False).iloc[:20]['Host_Name'],

            y = ab_nyc.groupby('Host_Name')['Listings_Count'].max().reset_index().sort_values(by = 'Listings_Count', ascending = False).iloc[:20]['Listings_Count'], 

                 palette= 'magma')

ax.set_title('Top 20 Hosts', fontdict = {'size': 15, 'weight':'bold'})

ax.set_xlabel('Hosts', fontdict = {'size': 15})

ax.set_ylabel('Number of Listings',fontdict = {'size': 15})

plt.xticks(rotation = 90)

plt.grid(True, color = 'black')
ab_nyc['Booked'] = (ab_nyc['Availability'].apply(lambda x : np.round(x/30))).apply(lambda x : 12-int(x))
plt.figure(figsize = (15,10))

ax = sns.barplot(x = ab_nyc[ab_nyc['Booked'] != 0].groupby('Booked')['Price'].mean().reset_index()['Booked'], 

            y = ab_nyc[ab_nyc['Booked'] != 0].groupby('Booked')['Price'].mean().reset_index()['Price'],

           palette='magma')

ax.set_title('Pricing according to listings booked for months', fontdict = {'size': 15, 'weight':'bold'})

ax.set_xlabel('Booked n Months', fontdict = {'size': 15})

ax.set_ylabel('Price of Listings',fontdict = {'size': 15})

plt.grid(True, color = 'black')