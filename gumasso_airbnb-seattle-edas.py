# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import folium

from folium import plugins

from folium.plugins import HeatMap

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.offline as py

import plotly.graph_objs as go

pd.set_option('display.max_columns', 500)

sns.set_style('darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("../input/listings.csv")

df2 = pd.read_csv("../input/calendar.csv")

df3 = pd.read_csv("../input/reviews.csv")
df1.head(3)
drop = ['listing_url','scrape_id','last_scraped','name','summary', 'space', 'description',

        'experiences_offered','neighborhood_overview', 'notes', 'transit', 'thumbnail_url', 'medium_url',

        'picture_url', 'xl_picture_url', 'host_url', 'host_about', 'host_thumbnail_url',

        'host_picture_url', 'street', 'license', 'host_name', 'host_location',

        'host_neighbourhood', 'neighbourhood','neighbourhood_cleansed',

        'neighbourhood_group_cleansed', 'city', 'state', 'zipcode',

        'market', 'experiences_offered', 'smart_location', 'host_acceptance_rate', 'country',

        'country_code', 'has_availability', 'calendar_last_scraped', 'requires_license',

        'jurisdiction_names', 'square_feet', 'weekly_price', 'monthly_price', 'security_deposit',

        'cleaning_fee', 'host_listings_count']

df = df1.drop(columns=drop) 
df.info()
df.head(2)
def correct_number(df_value):

        try:

            value = float(df_value[1:])

        

        except ValueError:

            value = np.NaN

        except TypeError:

            value = np.NaN

        return value

    

def correct_number1(df_value):

        try:

            value = float(df_value[:-1])

        

        except TypeError:

            value = np.NaN

        return value
df['host_since'] = pd.to_datetime(df['host_since'], format='%Y-%m-%d')

df['first_review'] = pd.to_datetime(df['first_review'], format='%Y-%m-%d')

df['last_review'] = pd.to_datetime(df['last_review'], format='%Y-%m-%d')



df['host_verifications_count'] = df['host_verifications'].apply(lambda x: x.count(' ') + 1)

df['amenities_count'] = df['amenities'].apply(lambda x: x.count(' ') + 1) 

df['property_type_new'] = df['property_type'] .replace(['Cabin', 'Camper/RV', 'Bungalow'], 'Category 1')

df['property_type_new'] = df['property_type_new'] .replace(['Condominium', 'Townhouse', 'Loft', 'Bed & Breakfast'], 'Category 2')

df['bed_type_new'] = df['bed_type'].replace(['Futon',' Pull-out Sofa', 'Airbed', 'Couch'], 'Other')



df['price_normal'] = df['price'].apply(correct_number)

df['extra_people_normal'] = df['extra_people'].apply(correct_number)

df['host_response_rate_normal'] = df['host_response_rate'].apply(correct_number1)



df = df.drop(columns=['host_verifications', 'amenities', 'property_type', 

                      'bed_type', 'extra_people', 'host_response_rate'])



df2['price_calendar'] = df2['price'].apply(correct_number)
object_c = df.select_dtypes(include='object')

numeric_c = df.select_dtypes(include='number')

for c in object_c.columns:

    l = len(object_c[c][object_c[c].notnull()].unique())

    print('Column {} has {} unique values'.format(c, l))

print('\n'*3,'NUMERIC')

for n in numeric_c.columns:

    print('Column', n)
print(df.isnull().sum().sort_values(ascending=False))

df.isnull().any(axis=1).value_counts()
f, (ax1,ax2) = plt.subplots(1,2, figsize=(14,8))

g = df2['price_calendar'].groupby(df2['date']).mean()

h = df2['available'][df2['available'] == 't'].groupby(df2['date']).count()

ax1.plot(g)

ax1.set_xticks(g.index[::30])

ax1.tick_params('x', labelrotation=90)

ax2.plot(h)

ax2.set_xticks(h.index[::30])

ax2.tick_params('x', labelrotation=90)

f.suptitle('Price and availability over the year', fontsize=20)

ax1.set_title('Mean price')

ax2.set_title('Availability')

plt.show()
a = df['host_id'].value_counts().reset_index()

a.rename(columns={'index': 'host_id', 'host_id': 'num_of_listings'}, inplace=True)

b = df['review_scores_rating'].groupby(df['host_id']).mean().reset_index()

c = a.merge(b, how='left', on='host_id')

#c = c[['index', 'review_scores_rating']]

c.sort_values(by='num_of_listings', ascending=False).head(10)



plt.figure(figsize=(14,8))

sns.scatterplot(x=c['num_of_listings'],y=c['review_scores_rating'], alpha=0.5)

plt.ylabel('Review rating')

plt.xlabel('Number of listings per host')

plt.title('Host involvement', fontsize=20)
map_hooray = folium.Map(location=[47.60, -122.24], zoom_start = 11) 



for i in range(0, df.shape[0]):

    folium.Circle(

    location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],

    popup=df.iloc[i]['price'],

    radius=df.iloc[i]['price_normal']/3,

    color='red',

    fill=True,

    fill_color='black').add_to(map_hooray)





map_hooray



#points = sns.scatterplot('latitude', 'longitude', data=df1, hue='price_normal')

df_high = df[df['price_normal'] > 400]

plt.figure(figsize=(14,8))

points = plt.scatter(df['latitude'], df['longitude'], c=df["price_normal"], s=20, cmap="viridis") #set style options

#add a color bar

plt.colorbar(points)

map_hooray = folium.Map(location=[47.60, -122.24], zoom_start = 11)



heat_data = [[row['latitude'],row['longitude']] for index, row in

             df[['latitude', 'longitude']].iterrows()]



hh =  HeatMap(heat_data).add_to(map_hooray)



map_hooray
def cross(c1,c2, xlabel, title):

    p = pd.crosstab(df[c1], df[c2])

    p.plot.bar(stacked=True, figsize=(14,8))

    plt.xlabel(xlabel)

    plt.suptitle(title, fontsize=20)

    plt.show()
cross('host_is_superhost', 'host_response_time', 'Superhost status', 'Status vs response time')
d = df['review_scores_rating'].groupby(df['host_response_time']).mean()

d = d.sort_values(ascending=False)

f, ax = plt.subplots(1,1, figsize=(14,8))

ax.plot(d, 'o-')

ax.set_xticklabels(d.index)

ax.tick_params('x', labelrotation=45)

plt.ylabel('Review rating')

plt.xlabel('Respone time')

plt.title('Mean rating of response time', fontsize=20)

plt.show()
plt.figure(figsize=(14,8))

sns.boxplot(x='host_is_superhost', y='review_scores_rating', data=df)

plt.ylabel('Review rating')

plt.xlabel('Superhost status')

plt.show()
cross('room_type', 'property_type_new', 'Room type', 'Types of rooms and properties')
cross('cancellation_policy', 'instant_bookable', 'Cancellation policy', 'Booking and cancellation')
plt.figure(figsize=(14,8))

sns.heatmap(df.drop(columns=['id','host_id', 'latitude', 'longitude']).corr(),cmap='Blues', annot=True, fmt='.1f', linewidths=0.5)
sns.pairplot(x_vars=['review_scores_rating', 'review_scores_accuracy',

                     'review_scores_checkin', 'review_scores_cleanliness',

                     'review_scores_communication', 'review_scores_location'],

             y_vars=['review_scores_rating', 'review_scores_accuracy',

                     'review_scores_checkin', 'review_scores_cleanliness',

                     'review_scores_communication', 'review_scores_location'],

            data=df, kind='reg', diag_kind='hist')
plt.figure(figsize=(14,8))

plt.hist(df['review_scores_rating'], bins=15, histtype='stepfilled', label='review_scores_rating', alpha=0.5)

for p in ['review_scores_accuracy',

          'review_scores_checkin', 'review_scores_cleanliness',

          'review_scores_communication', 'review_scores_location']:

    plt.hist(df[p]*10, bins=15, histtype='stepfilled', label=p, alpha=0.3)

    plt.legend(loc='upper left')

plt.suptitle('Histogram of ratings', fontsize=20)

plt.show()
x=df['beds']

y=df['bedrooms']

z=df['accommodates']





trace1 = go.Scatter3d(

    x=x,

    y=y,

    z=z,

    mode='markers',

    marker=dict(

        size=12,

        color=df['price_normal'],               

        colorscale='Viridis',   

        opacity=0.8,

        colorbar=dict(

                title='Price'

            ),

        

    ),text=df['price']

)



data = [trace1]

layout = go.Layout(

    scene=dict(

    xaxis=dict(

        title='beds'),

    yaxis=dict(

        title='bedrooms'),

    zaxis=dict(

        title='accommodates')),

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='3d-scatter-colorscale')