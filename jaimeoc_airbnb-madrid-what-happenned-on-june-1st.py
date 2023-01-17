import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import warnings

from IPython.core.display import display, HTML



warnings.filterwarnings('ignore')

display(HTML("<style>div.output_scroll { height: 44em; }</style>"))

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

%matplotlib inline
airbnb = pd.read_csv("../input/sep_madrid_airbnb.csv")
#Select all relevant columns



df_airbnb = airbnb[['listing_url', 'name', 'host_id','calculated_host_listings_count_entire_homes', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'zipcode',  'latitude', 'longitude', 'property_type', 'room_type', 'price', 'minimum_nights','number_of_reviews','availability_365', 'reviews_per_month']]
# We take a peek at the dataset



df_airbnb.head(3)
#Checking the amount of rows of the given dataset.



len(df_airbnb)
#Checking dtypes



df_airbnb.dtypes
#remove $ sign , commas, and leading and trailing white spaces. After that, convert it to float



df_airbnb['price'] = df_airbnb['price'].str.replace('[$,]','').str.strip().astype(np.float64)

df_airbnb.isnull().sum()
df_airbnb.loc[df_airbnb.reviews_per_month.isnull()].head(5)
df_airbnb.fillna(value = {'reviews_per_month': 0}, inplace = True)
df_airbnb.fillna(value = {'zipcode': 'unknown'}, inplace = True)
#setting figure size for future visualizations



sns.set(rc={'figure.figsize':(10,8)})
top_host = df_airbnb.host_id.value_counts().head(10)
host_counts_plot = top_host.plot(kind='barh', title = 'Hosts with the most accommodations')

host_counts_plot.set_ylabel('Host ID')

host_counts_plot.set_xlabel('Accommodation count')
# We are going check some statistic about the distribution of acommodations per host id



df_airbnb.groupby('host_id').size().describe()
# First of all, let's check the neighbourhoods in Madrid



df_airbnb[['neighbourhood_group_cleansed']].drop_duplicates().rename(columns = {'neighbourhood_group_cleansed': 'neighbourhood'})
# We'll use describe method to ouput all statistics grouped per neighbourhood



neighbour_stats = df_airbnb.groupby("neighbourhood_group_cleansed")["price"].describe().T

neighbour_stats
def extract_top_n(series, n = 1):

    '''

    Return top n values inside a series

    '''

    series_sort = series.sort_values(ascending = False)

    return series_sort[0:n]
# Now we compute the two top n values from our stats table



neighbour_stats.apply(extract_top_n, axis = 1,  n = 2)
pic_1=sns.boxplot(data=df_airbnb, y='neighbourhood_group_cleansed', x='price')

ax = pic_1.axes.set_xlim(0,800)
number_reviews_plot = df_airbnb.groupby('neighbourhood_group_cleansed')['number_of_reviews'].sum().sort_values(ascending = False).plot(kind = 'barh')

number_reviews_plot.set_title('Number of reviews by district')

number_reviews_plot.axes.set_xlim(0,50000)
# Now let's compute accommodations with zero or one reviews



zero_reviews = df_airbnb.groupby('neighbourhood_group_cleansed')['number_of_reviews'].apply(lambda x: ((x <= 1).sum())/len(x)).sort_values(ascending = False).plot(kind = 'barh')

zero_reviews.set_title("Percentage of accommodations with 1 or 0 reviews")
#WFirst of all, we'll  filter out our two neighbourhoods: Vicálvaro and San Blas - Canillejas 



df_event = df_airbnb[df_airbnb.neighbourhood_group_cleansed.isin(['Vicálvaro', 'San Blas - Canillejas'])]



_names_ = []



# Looping the names in each accommodation



for name in df_event.name:

    _names_.append(name)

    

# Funtion for spliting words



def split_name(name):

    '''

    Return a list of words given a sentence

    '''

    splits = str(name).split()

    return splits



#Empty list for counting words.



_names_for_count_ = []



#getting name string from our list and using split function, later appending to list above



for x in _names_:

    for word in split_name(x):

        word = word.lower()

        _names_for_count_.append(word)
#For counting word, wells use Counter class from collections package



from collections import Counter



#let's see top 10 used words by host to name their listing



_top_w = Counter(_names_for_count_).most_common()

_top_10_w = _top_w[0:10]

_top_10_w
#Let's see what we can do with our given longitude and latitude columns



#We'll picture lat and long of the accommodations that contained words referencing the football match that took 

#place on June 1st





df_champions = df_airbnb.loc[df_airbnb.name.str.contains('junio |champion|wanda|estadio (?!Bernabeu)|(?<!Bernabeu) stadium|league|LIVERPOOL|TOTTENHAM|final|metropolitano', case=False, regex=True, na = False)]



pic_2=df_champions.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',

                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.3, figsize=(10,8))

pic_2.legend()
import folium
m = folium.Map(

    location = [40.4168, -3.7038],

    tiles = 'Stamen Terrain',

    zoom_start = 12             

              )

df_champions.apply(lambda x: folium.Circle([x.latitude, x.longitude], 50, fill=True).add_to(m).add_to(m),axis = 1)

folium.Marker([40.4362, -3.5995], 'Wanda').add_to(m)



m
df_airbnb_clean = df_airbnb[~df_airbnb.neighbourhood_group_cleansed.isin(['San Blas - Canillejas', 'Vicálvaro'])]

df_airbnb_clean = df_airbnb_clean[~df_airbnb_clean.name.str.contains('junio |champion|wanda|estadio (?!Bernabeu)|(?<!Bernabeu) stadium|league|LIVERPOOL|TOTTENHAM|final|metropolitano', case=False, regex=True, na = False)]
pic_3=df_airbnb_clean.loc[df_airbnb_clean.price > 500].plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',

                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.3, figsize=(10,8))

pic_3.legend()
pic_4=sns.boxplot(data=df_airbnb_clean, y='neighbourhood_group_cleansed', x='price')

ax = pic_4.axes.set_xlim(0,800)
neighbour_stats = df_airbnb_clean.groupby("neighbourhood_group_cleansed")["price"].describe().T.apply(extract_top_n, axis = 1,  n = 2)

neighbour_stats 