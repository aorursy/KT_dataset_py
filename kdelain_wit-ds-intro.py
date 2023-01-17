# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: 

# https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra and other math things

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # graphing more complicated things

import matplotlib.pyplot as plt  # more graph things



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) 

# will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
my_array = [[ 0,  1,  2,  3,  4],

       [ 5,  6,  7,  8,  9],

       [10, 11, 12, 13, 14],

       [15, 16, 17, 18, 19],

       [20, 21, 22, 23, 24]]
my_array[1]
my_array[1][3]
listings = pd.read_csv('/kaggle/input/berlin-airbnb-data/listings_summary.csv')
# This gives us the first 5 rows by default. 

# You can enter a number in the parentheses to give you more or less.

# This is a great way to get a quick look at your data -- with some caveats

listings.head()



# You can look at the last rows by using .tail() instead:

#listings.tail()



# Take a minute to look over the information below. 

# Is there anything interesting to you?  

# Is there anything that you have questions about?
listings.columns
# Let's check out a text field. 

# Remember we just wanted to select a column? 

# Super easy with a data frame:

listings['city'].head()
# More useful let's find out how many unique ones there are:

listings['city'].unique()
# Neat. How many listings in each of those cities?

listings['city'].value_counts()
# Let's check out a numerical field. One thing we can do are look at the 

# distribution:

listings['bedrooms'].plot.hist()
#What if I want some statistics?

listings.bedrooms.describe()



# NOTE: Be careful with .describe() because it will try even if it doesn't 

# make sense with the data!
# What happens if we try to describe a text field?



listings.neighbourhood.describe()
# How can I tell if a field is in fact numerical? One way might just be

# to look at the header. 

# If it doesn't show up in the full list if I know the column name 

# I can easily look at it by adding it as an index. Notice that DataFrames

# assume you are looking for a column unless you specify you want a row.

listings['review_scores_rating'].head()
# Write your code here. 

# Write more code here. 

listings['requires_license'].value_counts()
listings[listings['requires_license']=='f']
# Who needs the URL? Let's just drop it, right?

listings.drop('listing_url', axis=1)  # axis=1 means a column
listings.head()
listings.drop('license', axis=1, inplace=True)  # True must be capitalized
listings.head()  # all gone



# You can drop rows too if you want (axis=0) but let's not worry about that now.
columns_of_interest = ['id','host_has_profile_pic','host_since','neighbourhood_cleansed', 'neighbourhood_group_cleansed',

                   'host_is_superhost','description',

                   'latitude', 'longitude','is_location_exact', 'property_type', 'room_type', 'accommodates', 'bathrooms',  

                   'bedrooms', 'bed_type', 'amenities', 'price', 'cleaning_fee',

                   'review_scores_rating','reviews_per_month','number_of_reviews',

                   'review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',

                   'review_scores_communication','review_scores_location','review_scores_value',

                   'security_deposit', 'extra_people', 'guests_included', 'minimum_nights',  

                   'instant_bookable', 'is_business_travel_ready', 'cancellation_policy','availability_365']



df = listings[columns_of_interest].set_index('id')
# Are there any NaNs in there?

df['is_location_exact'].isna().sum()
#Ok cool. There are a bunch of ways to make 't' and 'f' something 

# useful (we'll use 1 and 0) but let's try a map:

df['is_location_exact'] = df['is_location_exact'].map({'f':0,'t':1})
df['is_location_exact'].unique()
# Try one here:

# This will give us a count of how many in each column are missing (null, aka NaN)

df.isnull().sum()
# These three additional columns just need to replace 'f' and 't' with 0 and 1

df['host_is_superhost'] = df['host_is_superhost'].map({'f':0,'t':1})

df['is_business_travel_ready'] = df['is_business_travel_ready'].map({'f':0,'t':1})

df['instant_bookable'] = df['instant_bookable'].map({'f':0,'t':1})
# This column needs to first replace the NaN with 'f' and then replace the 'f' and 't' with 0 and 1.

df['host_has_profile_pic'].fillna('f',inplace=True)

df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'f':0,'t':1})

df['host_has_profile_pic'].value_counts()
# Let's make a quick bar plot of that profile pic column:

sns.countplot(x='host_has_profile_pic',data=df)
# Gotta clean up those prices, remove the $ and , (replace with empty string) 

# and then turn them from strings into numbers:

df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)

df['cleaning_fee'] = df['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype(float)

df['security_deposit'] = df['security_deposit'].str.replace('$', '').str.replace(',', '').astype(float)

df['extra_people'] = df['extra_people'].str.replace('$', '').str.replace(',', '').astype(float)
# Examples, either run this or replace with your own choice

df['cleaning_fee'].fillna(df['cleaning_fee'].median(), inplace=True)

df['security_deposit'].fillna(df['security_deposit'].mean(), inplace=True)

df['extra_people'].fillna(df['extra_people'].min(), inplace=True)
sns.distplot(df['price'], kde=True)
df['price'].isnull().sum()
df['price'].describe()
sns.countplot(x='room_type',data=df)
sns.distplot(df[df['price']<250]['price'], kde=True)
sns.distplot((df[(df['price']<250) & (df['room_type']=='Entire home/apt')]['price']), kde=True)
sns.distplot((df[(df['price']<250) & (df['room_type']=='Private room')]['price']), kde=True)
g = sns.FacetGrid(data=df,col='room_type')

g.map(plt.hist,'price', range=(0,200))
from math import sin, cos, sqrt, atan2, radians



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



df['distance'] = df.apply(haversine_distance_central,axis=1)
sns.jointplot(x='distance',y='price',data=df,kind='scatter', xlim=(0,15), ylim=(0,250))
df['distance'].plot.hist()
sns.heatmap(df.corr(),cmap='coolwarm')