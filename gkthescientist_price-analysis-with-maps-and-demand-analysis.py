import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# This will be needed to work with maps

!pip install folium
# This is required for wordcloud

!pip install wordcloud
import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib as mpl

import scipy

from scipy.stats import norm

from scipy import stats

import matplotlib.pyplot as plt

import folium

import datetime

import warnings

from math import sin, cos, sqrt, atan2, radians



%matplotlib inline

pd.set_option('display.max_columns', 500)

mpl.style.use(['seaborn-darkgrid'])

warnings.filterwarnings('ignore')
df_listing_summ = pd.read_csv("/kaggle/input/berlin-airbnb-data/listings_summary.csv")

print("listing_summ : " + str(df_listing_summ.shape))
df_rsumm = pd.read_csv("/kaggle/input/berlin-airbnb-data/reviews_summary.csv")

print("reviews_summary : " + str(df_rsumm.shape))
#ignore unwanted columns like URLs

columns_to_keep = ['id','host_has_profile_pic','host_since',

                   'latitude', 'longitude','property_type', 'room_type', 'accommodates', 'bathrooms',  

                   'bedrooms', 'bed_type', 'amenities', 'price', 'cleaning_fee',

                   'security_deposit', 'minimum_nights',  

                   'instant_bookable', 'cancellation_policy','availability_365']

df_listing_summ = df_listing_summ[columns_to_keep].set_index('id')
print("listing_summ : " + str(df_listing_summ.shape))
df_listing_summ.isnull().sum()
#Convert f,t to 0 or 1

df_listing_summ['instant_bookable'] = df_listing_summ['instant_bookable'].map({'f':0,'t':1})
#fill f for N/A in host_has_profile_pic column for further correct mapping

set(df_listing_summ['host_has_profile_pic'])

df_listing_summ['host_has_profile_pic'].fillna('f',inplace=True)
#Convert f,t to 0 or 1

df_listing_summ['host_has_profile_pic'] = df_listing_summ['host_has_profile_pic'].map({'f':0,'t':1})
#Remove $ from price, fee columns and convert to float

df_listing_summ['price'] = df_listing_summ['price'].str.replace('$', '').str.replace(',', '').astype(float)

df_listing_summ['cleaning_fee'] = df_listing_summ['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype(float)

df_listing_summ['security_deposit'] = df_listing_summ['security_deposit'].str.replace('$', '').str.replace(',', '').astype(float)
#cleaning_fee cleanup of N/a replace with median value

df_listing_summ['cleaning_fee'].fillna(df_listing_summ['cleaning_fee'].median(), inplace=True)
#security_deposit cleanup of N/a replace with median value

df_listing_summ['security_deposit'].fillna(df_listing_summ['security_deposit'].median(), inplace=True)
#cleanup bathroom , bedroom columns

df_listing_summ['bathrooms'].fillna(1,inplace=True)

df_listing_summ['bedrooms'].fillna(1,inplace=True)
#Check distribution of price column

df_listing_summ['price'].describe()
df_listing_summ.drop(df_listing_summ[ (df_listing_summ.price > 200) | (df_listing_summ.price == 0) | (df_listing_summ.price == 1) ].index, axis=0, inplace=True)

df_listing_summ['price'].describe()
# boxplot of price column

red_square = dict(markerfacecolor='r', markeredgecolor='r', marker='.')

df_listing_summ['price'].plot(kind='box', xlim=(0, 175), vert=False, flierprops=red_square, figsize=(10,2));
df_listing_summ['No_of_amentities'] = df_listing_summ['amenities'].apply(lambda x:len(x.split(',')))
df_listing_summ['Laptop_friendly_workspace'] = df_listing_summ['amenities'].str.contains('Laptop friendly workspace')

df_listing_summ['TV'] = df_listing_summ['amenities'].str.contains('TV')

df_listing_summ['Family_kid_friendly'] = df_listing_summ['amenities'].str.contains('Family/kid friendly')

df_listing_summ['Host_greets_you'] = df_listing_summ['amenities'].str.contains('Host greets you')

df_listing_summ['Smoking_allowed'] = df_listing_summ['amenities'].str.contains('Smoking allowed')

df_listing_summ['Hot_water'] = df_listing_summ['amenities'].str.contains('Hot water')

df_listing_summ['Fridge'] = df_listing_summ['amenities'].str.contains('Refrigerator')
# dropping amenities as we have inferred above as different categories

dropped = ['amenities']

df_listing_summ.drop(dropped,axis=1,inplace=True)
#Convert false,true to 0 or 1

df_listing_summ['Laptop_friendly_workspace'] = df_listing_summ['Laptop_friendly_workspace'].astype(int)

df_listing_summ['TV'] = df_listing_summ['TV'].astype(int)

df_listing_summ['Family_kid_friendly'] = df_listing_summ['Family_kid_friendly'].astype(int)

df_listing_summ['Host_greets_you'] = df_listing_summ['Host_greets_you'].astype(int)

df_listing_summ['Smoking_allowed'] = df_listing_summ['Smoking_allowed'].astype(int)

df_listing_summ['Hot_water'] = df_listing_summ['Hot_water'].astype(int)

df_listing_summ['Fridge'] = df_listing_summ['Fridge'].astype(int)
#Calculate distance from central berlin

def haversine_distance_central(row):

    berlin_lat,berlin_long = radians(52.5200), radians(13.4050)

    R = 6373.0

    long = radians(row['longitude'])

    lat = radians(row['latitude'])

    

    dlon = long - berlin_long

    dlat = lat - berlin_lat

    a = sin(dlat / 2)**2 + cos(lat) * cos(berlin_lat) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
#Calculate distance from airport

def haversine_distance_airport(row):

    berlin_lat,berlin_long = radians(52.3733), radians(13.5064)

    R = 6373.0

    long = radians(row['longitude'])

    lat = radians(row['latitude'])

    

    dlon = long - berlin_long

    dlat = lat - berlin_lat

    a = sin(dlat / 2)**2 + cos(lat) * cos(berlin_lat) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
#Calculate distance from berlin railway station

def haversine_distance_rail(row):

    berlin_lat,berlin_long = radians(52.5073), radians(13.3324)

    R = 6373.0

    long = radians(row['longitude'])

    lat = radians(row['latitude'])

    

    dlon = long - berlin_long

    dlat = lat - berlin_lat

    a = sin(dlat / 2)**2 + cos(lat) * cos(berlin_lat) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
df_listing_summ['distance_central'] = df_listing_summ.apply(haversine_distance_central,axis=1)

df_listing_summ['distance_airport'] = df_listing_summ.apply(haversine_distance_airport,axis=1)

df_listing_summ['distance_railways'] = df_listing_summ.apply(haversine_distance_rail,axis=1)

df_listing_summ['distance_avg'] = ( df_listing_summ['distance_central'] + df_listing_summ['distance_airport'] + df_listing_summ['distance_railways'] )/3.0
df_listing_summ.sort_values(by='price',ascending=False,axis=0,inplace=True) #sorting frame by price desc
df_list_summ_top10000 = df_listing_summ.head(10000)

df_list_summ_top1000 = df_listing_summ.head(1000)
sns.set(style="white")

corr = df_listing_summ.corr()



# generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# set up the matplotlib figure

fig, ax = plt.subplots(figsize=(20, 15))



# generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink":.5},cbar=True);
pearson_coef, p_value = stats.pearsonr(df_listing_summ['accommodates'], df_listing_summ['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
pearson_coef, p_value = stats.pearsonr(df_listing_summ['security_deposit'], df_listing_summ['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df_listing_summ['No_of_amentities'], df_listing_summ['price'])

print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
# Setting a base map

lat = 52.509

long = 13.381

base = folium.Map(location=[lat,long], zoom_start=12) #base map setting

base
neighbourhoods = folium.map.FeatureGroup()
lat_long_list = [[52.520,13.405],[52.373,13.506],[52.507,13.332]] #locatioms of central berlin , railway stn, airport
for i in range(0,len(lat_long_list)):

    neighbourhoods.add_child(

        folium.CircleMarker(

        lat_long_list[i],

        radius = 16,

        color='yellow',

        fill=True,

        fill_color='red',

        fill_opacity=0.6

        )

    )

base.add_child(neighbourhoods)
neighbourhoods = folium.map.FeatureGroup()

for inc_lat,inc_long in zip(df_list_summ_top1000.longitude,df_list_summ_top1000.latitude):

    neighbourhoods.add_child(

    folium.CircleMarker(

    [inc_long,inc_lat],

    radius = 5,

    color='yellow',

    fill=True,

    fill_color='blue',

    fill_opacity=0.6

    )

)

base.add_child(neighbourhoods)
fig = plt.figure(figsize=(10,6))

ax0 = fig.add_subplot(2, 2, 1)

ax1 = fig.add_subplot(2, 2, 2)

ax2 = fig.add_subplot(2, 2, 3)



sns.distplot(df_list_summ_top1000["distance_central"], bins=10, kde=False,ax=ax0)

ax0.set_title('Distances central berlin to apartments')

ax0.set_xlabel('distance_central')

ax0.set_ylabel('#properties')



sns.distplot(df_list_summ_top1000["distance_railways"], bins=10, kde=False,ax=ax1)

ax1.set_title('Distances railway station to apartments')

ax1.set_xlabel('distance_railways')

ax1.set_ylabel('#properties')



sns.distplot(df_list_summ_top1000["distance_airport"], bins=10, kde=False,ax=ax2)

ax2.set_title('Distances airport to apartments')

ax2.set_xlabel('distance_airport')

ax2.set_ylabel('#properties')



plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.5)

plt.show()
def mapper(month):

    date = datetime.datetime(2000, month, 1)  # You need a dateobject with the proper month

    return date.strftime('%b')  # %b returns the months abbreviation, other options [here][1]
df_rsumm['date'] = pd.to_datetime(df_rsumm['date'])

df_rsumm['year'] = df_rsumm['date'].dt.year

df_rsumm['month'] = df_rsumm['date'].dt.month

df_rsumm['day'] = df_rsumm['date'].dt.day



df_rsumm['year'] = df_rsumm['year'].astype(int)

df_rsumm['month'] = df_rsumm['month'].astype(int)

df_rsumm['day'] = df_rsumm['day'].astype(int)
df_rsumm.sort_values(['year', 'month'], ascending=[True, True],axis=0,inplace=True) #sorting frame by year,month asc
df_rsumm['month'] = df_rsumm['month'].apply(mapper) ##convert month to month name
df_rsumm_orig = df_rsumm.copy(deep=False) 
dropped = ['reviewer_name','comments']

df_rsumm.drop(dropped,axis=1,inplace=True)
df_rsumm['year'].value_counts()
df_2015 = df_rsumm[df_rsumm['year'] == 2015]

df_2016 = df_rsumm[df_rsumm['year'] == 2016]

df_2017 = df_rsumm[df_rsumm['year'] == 2017]

df_2018 = df_rsumm[df_rsumm['year'] == 2018]
dropped = ['year','day','id','date','listing_id']

df_2015.drop(dropped,axis=1,inplace=True)

df_2016.drop(dropped,axis=1,inplace=True)

df_2017.drop(dropped,axis=1,inplace=True)

df_2018.drop(dropped,axis=1,inplace=True)
df_2015["count"] = df_2015.groupby("month")["reviewer_id"].transform('count')

df_2016["count"] = df_2016.groupby("month")["reviewer_id"].transform('count')

df_2017["count"] = df_2017.groupby("month")["reviewer_id"].transform('count')

df_2018["count"] = df_2018.groupby("month")["reviewer_id"].transform('count')
dropped = ['reviewer_id']

df_2015.drop(dropped,axis=1,inplace=True)

df_2016.drop(dropped,axis=1,inplace=True)

df_2017.drop(dropped,axis=1,inplace=True)

df_2018.drop(dropped,axis=1,inplace=True)

df_2015 = df_2015.drop_duplicates()

df_2016 = df_2016.drop_duplicates()

df_2017 = df_2017.drop_duplicates()

df_2018 = df_2018.drop_duplicates()

df_2015=df_2015.reset_index(drop=True)

df_2016=df_2016.reset_index(drop=True)

df_2017=df_2017.reset_index(drop=True)

df_2018=df_2018.reset_index(drop=True)
fig = plt.figure(figsize=(20, 6))

ax0 = fig.add_subplot(2, 2, 1)

ax1 = fig.add_subplot(2, 2, 2)

ax2 = fig.add_subplot(2, 2, 3)

ax3 = fig.add_subplot(2, 2, 4)



df_2018.plot(kind='line', color='blue', x='month',y='count',marker='o',ax=ax0) # add to subplot 1

ax0.set_title('Visitors trend in 2018')

ax0.set_xlabel('Month')

ax0.set_ylabel('Number of visitors')



df_2017.plot(kind='line', color='red', x='month',y='count',marker='o',ax=ax1) # add to subplot 2

ax1.set_title('Visitors trend in 2017')

ax1.set_xlabel('Month')

ax1.set_ylabel('Number of visitors')



df_2016.plot(kind='line', color='cyan', x='month',y='count',marker='o',ax=ax2) # add to subplot 3

ax2.set_title('Visitors trend in 2016')

ax2.set_xlabel('Month')

ax2.set_ylabel('Number of visitors')



df_2015.plot(kind='line', color='green', x='month',y='count',marker='o',ax=ax3) # add to subplot 4

ax3.set_title('Visitors trend in 2015')

ax3.set_xlabel('Month')

ax3.set_ylabel('Number of visitors')



plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5)

plt.show()
dropped = ['listing_id','id','reviewer_id','reviewer_name','day','date']

df_rsumm_orig.drop(dropped,axis=1,inplace=True)
df_2015_comments = df_rsumm_orig[df_rsumm_orig['year'] == 2015]

df_2016_comments = df_rsumm_orig[df_rsumm_orig['year'] == 2016]

df_2017_comments = df_rsumm_orig[df_rsumm_orig['year'] == 2017]

df_2018_comments = df_rsumm_orig[df_rsumm_orig['year'] == 2018]
df_2015_sep_comments = df_rsumm_orig[(df_rsumm_orig['year'] == 2015) & (df_rsumm_orig['month'].str.contains("Sep"))]

df_2016_sep_comments = df_rsumm_orig[(df_rsumm_orig['year'] == 2016) & (df_rsumm_orig['month'].str.contains("Sep"))]

df_2017_sep_comments = df_rsumm_orig[(df_rsumm_orig['year'] == 2017) & (df_rsumm_orig['month'].str.contains("Sep"))]

df_2018_sep_comments = df_rsumm_orig[(df_rsumm_orig['year'] == 2018) & (df_rsumm_orig['month'].str.contains("Sep"))]
dropped = ['month']

df_2015_comments.drop(dropped,axis=1,inplace=True)

df_2016_comments.drop(dropped,axis=1,inplace=True)

df_2017_comments.drop(dropped,axis=1,inplace=True)

df_2018_comments.drop(dropped,axis=1,inplace=True)

df_2015_comments=df_2015_comments.reset_index(drop=True)

df_2016_comments=df_2016_comments.reset_index(drop=True)

df_2017_comments=df_2017_comments.reset_index(drop=True)

df_2018_comments=df_2018_comments.reset_index(drop=True)



df_2015_sep_comments.drop(dropped,axis=1,inplace=True)

df_2016_sep_comments.drop(dropped,axis=1,inplace=True)

df_2017_sep_comments.drop(dropped,axis=1,inplace=True)

df_2018_sep_comments.drop(dropped,axis=1,inplace=True)

df_2015_sep_comments=df_2015_sep_comments.reset_index(drop=True)

df_2016_sep_comments=df_2016_sep_comments.reset_index(drop=True)

df_2017_sep_comments=df_2017_sep_comments.reset_index(drop=True)

df_2018_sep_comments=df_2018_sep_comments.reset_index(drop=True)
df_list_year_wise = [df_2015_comments,df_2016_comments,df_2017_comments,df_2018_comments] #list of all frames
df_list_sep = [df_2015_sep_comments,df_2016_sep_comments,df_2017_sep_comments,df_2018_sep_comments] #list of all frames
from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS)
rev_comments_wc = WordCloud(

    background_color='white',

    max_words=100000, #if we dont give this it does for entire rows for that frame

    stopwords = stopwords

)

#instantinate word cloud objects

def show_wclouds(text):

    rev_comments_wc.generate(text)

    return(rev_comments_wc)
fig = plt.figure(figsize=(15, 6))

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4)

for i in range(0,len(df_list_year_wise)):

    ax = fig.add_subplot(2,2,i+1)

    ax.imshow(show_wclouds(str(df_list_year_wise[i]['comments'])),interpolation='bilinear')

    ax.axis('off')

    title="Review Comments trend in "+str(df_list_year_wise[i]['year'].head(1).values[0])

    ax.set_title(title)
fig = plt.figure(figsize=(15, 6))

plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4)

for i in range(0,len(df_list_sep)):

    ax = fig.add_subplot(2,2,i+1)

    ax.imshow(show_wclouds(str(df_list_sep[i]['comments'])),interpolation='bilinear')

    ax.axis('off')

    title="Review Comments trend in SEP of "+str(df_list_sep[i]['year'].head(1).values[0])

    ax.set_title(title)