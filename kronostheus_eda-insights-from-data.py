import re

import nltk

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from wordcloud import WordCloud

from PIL import Image
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head()
print("DataFrame shape\n===============\nRows: {}\nColumns: {}".format(data.shape[0], data.shape[1]))
data.dtypes
data.host_id = data.host_id.astype(str)

data.id = data.id.astype(str)
data.dtypes
data.rename(

    columns={

        'id': 'listing_id',                                      # match host_id naming convention

        'name': 'title',                                         # Series also has an attribute 'name'

        'neighbourhood_group': 'borough',                        # more common 

        'room_type': 'listing_type',                             # not all listings are rooms

        'number_of_reviews': 'total_reviews',                    # better naming convention

        'reviews_per_month': 'monthly_reviews',                  # better naming convention

        'calculated_host_listings_count': 'host_listings',       # complicated original name

        'availability_365': 'yearly_availability'                # '_365' not as informative

    },

    inplace=True

)



data.columns
data.isnull().sum()
missing_reviews = data[data.last_review.isnull() & data.monthly_reviews.isnull()]

print("Number of missing values: {}".format(missing_reviews.shape[0]))

missing_reviews.head()
data[data.last_review.isnull()].total_reviews.value_counts()
data.last_review = data.last_review.fillna('Never')

data.monthly_reviews = data.monthly_reviews.fillna(0)
print("Empty 'Last Review': {}".format(data.last_review.isnull().sum()))

print("Empty 'Reviews per/ Month': {}".format(data.monthly_reviews.isnull().sum()))
data[data.title.isnull()]
def fill_title(row):

    return "{} in {}, {}".format(row.listing_type, row.neighbourhood, row.borough)
data.title = data.apply(lambda x: fill_title(x) if str(x.title) == 'nan' else x.title, axis=1)
print("Empty 'Name': {}".format(data.title.isnull().sum()))
data.iloc[18047]
data.host_name =data.host_name.fillna('NA')
data.isnull().any()
data.head(3)
listing_counts = data.host_listings.value_counts()

sum(listing_counts / listing_counts.index) == len(data.host_id.unique())
print("Number of unique hosts: {}".format(len(data.host_id.unique())))
len(data.listing_id.unique()) == data.shape[0]
print("Total Listings: {}".format(data.shape[0]))
borough_dict = {borough: data[data.borough == borough].neighbourhood.unique() 

                for borough in data.borough.unique()}
len(data.neighbourhood.unique()) == sum([len(neighbourhoods) for neighbourhoods in borough_dict.values()])
print('New York has a total of {} neighbourhoods:\n'.format(len(data.neighbourhood.unique())))



for borough, neighbourhoods in borough_dict.items():

    print('\t{} has {} neighbourhoods;'.format(borough, len(neighbourhoods)))
dup_coord = data[data.duplicated(subset=['longitude', 'latitude'], keep=False)]

print('There are {} repeated locations.'.format(dup_coord.shape[0]))
dup_coord.sort_values(['latitude', 'longitude']).head()
fig, ax = plt.subplots(figsize=(6,6))



img_ = Image.open('../input/separatednycboroughs/ny_map_new.png')

img = np.array(img_)



ext = [-74.258, -73.69, 40.49, 40.92]



plt.imshow(img, zorder=0, extent=ext)



plt.title('New York City')

plt.xlabel('Longitude')

plt.ylabel('Latitude')



plt.show()
colors = sns.color_palette()
color_palette = {borough: colors[idx] for idx, borough in enumerate(data.borough.unique())}
def plot_map(x, y, hue, title, size=20, lw=0.5, palette=color_palette, color=None):

    

    """

    Plot common scatterplot while keeping control over certain properties. 

    This code is repeated multiple times and so I created a function to avoid code repetition.

    """

    

    fig, ax = plt.subplots(figsize=(8,8))

    

    # NYC map

    plt.imshow(img, zorder=0, extent=ext)

    

    sns.scatterplot(x=x, y=y, hue=hue, ax=ax, zorder=1, linewidth=lw, s=size, palette=color_palette, color=color)

    

    plt.title(title)

    plt.xlabel('Longitude')

    plt.ylabel('Latitude')

    

    plt.show()
sns.palplot(color_palette.values())

plt.xticks(ticks=np.arange(len(data.borough.unique())), labels=data.borough.unique())

plt.show()
alt_colors = sns.color_palette('Set2')

sns.palplot(alt_colors)
# Sorting data will prevent a lot of effort with colouring

data = data.sort_values('borough')
plot_map(

    x=data.longitude,

    y=data.latitude,

    hue=data.borough.values, 

    title='Listings by Borough'

)
cmaps = {

    'Manhattan': 'Greens',

    'Brooklyn': 'Oranges',

    'Queens': 'Reds',

    'Bronx': 'Blues',

    'Staten Island': 'Purples'

}



fig, ax = plt.subplots(figsize=(8,8))

    

plt.imshow(img, zorder=0, extent=ext)



for borough in data.borough.unique():

    sns.kdeplot(

        data=data[data.borough == borough].longitude, 

        data2=data[data.borough == borough].latitude,

        shade=True, 

        shade_lowest=False,

        cmap=cmaps[borough],

        alpha=0.8,

        ax=ax,

        zorder=1

    )





plt.title('Borough Listing Density')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.show()
"""By grouping by Borough, we'll get a DataFrame containing only that borough's listings,

which we can use to get mean coordinates"""



borough_geo = np.array([[borough_df.longitude.mean(), borough_df.latitude.mean()]

                        for b, borough_df in data.groupby(by='borough')])
plot_map(

    x=borough_geo[:,0],

    y=borough_geo[:,1],

    hue=data.borough.unique(),

    title='Mean Latitude/Longitude of Boroughs',

    size=100

)
# Same as before, with the exception of the inclusion of the borough name to insure color correctness

neighbourhood_geo = np.array([[nb_df.longitude.mean(), nb_df.latitude.mean(), nb_df.borough.unique()[0]] 

                              for n, nb_df in data.groupby(by=['neighbourhood'])], dtype='object')
plot_map(

    x=neighbourhood_geo[:,0],

    y=neighbourhood_geo[:,1], 

    hue=neighbourhood_geo[:,2],

    title='Mean Latitude/Longitude of Neighbourhoods'

)
plot_map(

    x=data.longitude,

    y=data.latitude,

    hue=data.borough.values,

    title='New York City',

    size=5,

    lw=0.1

)
# I make entries as Numpy arrays as they are easier to slice

empty_locs = {

    'Manhattan': np.array([

        (-73.965278, 40.782222, 'Central Park'),

        (-73.921944, 40.796667, 'Randalls and Wards Island'),

        (-73.970000, 40.804167, 'Riverside Park'),

        (-73.930844, 40.868485, 'Fort Washington/Inwood Hill Park'),

        (-74.016111, 40.691389, 'Governors Island')

    ], dtype='object'),

    'Staten Island': np.array([

        (-74.125540, 40.589700, 'Latourette Park'),

        (-74.186345, 40.577306, 'Freshkills Park'),

        (-74.126438, 40.548089, 'Great Kills Park'),

        (-74.185817, 40.626650, 'New York Container Terminal'),

        (-74.234468, 40.542046, 'Clay Pit Ponds State Park')

    ], dtype='object'),

    'Brooklyn': np.array([

        (-73.918732, 40.597653, 'Marine Park'),

        (-73.890567, 40.591291, 'Floyd Bennett Field'),

        (-74.019650, 40.612490, 'Dyker Beach Golf Course'),

        (-73.990430, 40.652862, 'Green-Wood Cemetery'),

        (-73.968404, 40.663462, 'Prospect Park')

    ], dtype='object'),

    'Queens': np.array([

        (-73.825591, 40.617744, 'Jamaica Bay Wildlife Refuge'),

        (-73.783515, 40.650281, 'JFK Airport'),

        (-73.868705, 40.698738, 'Jackie Robinson Parkway'),

        (-73.922234, 40.734461, 'Calvary Cemetery'),

        (-73.874133, 40.776953, 'LaGuardia Airport')

    ], dtype='object'),

    'Bronx': np.array([

        (-73.911185, 40.901846, 'Riverdale Park'),

        (-73.885447, 40.894290, 'Van Cortlandt Park'),

        (-73.877263, 40.856996, 'NYC Botanical Garden/Bronx Zoo'),

        (-73.808715, 40.877160, 'Pelham Bay Park'),

        (-73.884190, 40.791865, 'Rikers Island')

    ], dtype='object')

}
def plot_empty_locations(borough):

    

    """

    Show all listings of input borough and plot five of its most prominent visible empty areas.

    """

    

    fig, ax = plt.subplots(figsize=(8,8))

    

    # NYC map

    ax.imshow(img, zorder=0, extent=ext)

    

    # All listings from borough

    sns.scatterplot(x=data[data.borough == borough].longitude,

                    y=data[data.borough == borough].latitude,

                    ax=ax, 

                    zorder=1,

                    s=5,

                    linewidth=0,

                    edgecolor=None,

                    color='skyblue'

                   )



    # Get numpy array of borough

    notable_locations = empty_locs[borough]

    

    # Slice through arrays

    sns.scatterplot(x=notable_locations[:, 0],

                    y=notable_locations[:, 1],

                    hue=notable_locations[:, 2],

                    ax=ax, 

                    zorder=2,

                    palette='Dark2',

                    s=60

                   )

    

    plt.title("{}'s empty locations".format(borough))

    plt.xlabel('Longitude')

    plt.ylabel('Latitude')

    plt.show()
data.borough.unique()
plot_empty_locations('Brooklyn')
plot_empty_locations('Manhattan')
plot_empty_locations('Queens')
plot_empty_locations('Staten Island')
plot_empty_locations('Bronx')
sns.countplot(data.borough, palette=color_palette)

plt.xlabel(None)

plt.show()
def get_borough_count():

    cnt_df = pd.DataFrame(data.borough.value_counts()).reset_index()

    cnt_df.columns = ['borough', 'listings']

    return cnt_df
def get_borough_info():

    cnt_df = get_borough_count()

    

    frq_df = pd.DataFrame(data.borough.value_counts(normalize=True)).reset_index()

    frq_df.columns = ['borough', 'frequency']

    

    return cnt_df.merge(frq_df, on='borough')
get_borough_info()
def listings_by_borough_size():

    land_area = {'Manhattan': 59.1, 'Bronx': 109, 'Brooklyn': 183.4, 'Queens': 281.1, 'Staten Island': 151.2}

    

    la_df = pd.DataFrame.from_dict(land_area, orient='index').reset_index()

    la_df.columns = ['borough', 'land area (km2)']

    

    cnt_df = get_borough_count()

    

    cnt_df = cnt_df.merge(la_df, on='borough')

    

    cnt_df['listing per km2'] = cnt_df.apply(lambda row: row.listings / land_area[row.borough], axis=1)

    

    return cnt_df
listings_by_borough_size()
print('Total number of neighbourhoods: {}'.format(len(data.neighbourhood.unique())))
sns.countplot(

    y=data.neighbourhood, 

    hue=data.borough, order=data.neighbourhood.value_counts().iloc[:10].index,

    palette=color_palette,

    dodge=False

)

plt.show()
def get_top_locations(df):

    return np.array([

        [

            df[df.neighbourhood == nb].longitude.mean(),

            df[df.neighbourhood == nb].latitude.mean(),

            df[df.neighbourhood == nb].borough.unique()[0]

        ] 

        for nb in df.neighbourhood.value_counts().iloc[:10].index], dtype='object')
nb_locs = get_top_locations(data)
plot_map(

    x=nb_locs[:,0],

    y=nb_locs[:,1], 

    hue=nb_locs[:,2],

    title='Top 10 Neighbourhoods',

    size=50

)
def top_nb_boroughs(borough):

    b_df = data[data.borough == borough]

    

    top_nb = get_top_locations(b_df)

    

    print('{} has a total of {} neighbourhoods.'.format(borough, len(b_df.neighbourhood.unique())))

    

    plot_map(

        x=top_nb[:,0],

        y=top_nb[:,1],

        hue=None,

        color=color_palette[borough],

        title='Top 10 Neighbourhoods of {}'.format(borough),

        size=50

    )

    

    sns.countplot(

        y=b_df.neighbourhood, 

        hue=b_df.borough, 

        order=b_df.neighbourhood.value_counts().iloc[:10].index,

        palette=color_palette,

        dodge=False

    )

    plt.show()
top_nb_boroughs('Manhattan')
top_nb_boroughs('Brooklyn')
top_nb_boroughs('Bronx')
top_nb_boroughs('Queens')
data[data.neighbourhood == 'Flushing'].head()
data[data.neighbourhood == 'Jamaica'].head()
top_nb_boroughs('Staten Island')
data[data.neighbourhood == 'Great Kills'].head()
sns.countplot(data.listing_type, palette=alt_colors)

plt.title('Amount of listing types')

plt.xlabel(None)

plt.show()
fig = plt.figure(figsize=(12,6))

sns.countplot(data.borough, hue=data.listing_type, palette=alt_colors)

plt.title('Listing types within each borough')

plt.xlabel(None)

plt.show()
g = sns.catplot(x="borough", col="listing_type", data=data, kind='count', aspect=0.9, palette=color_palette)

g.set_axis_labels("", "Count")

plt.show()
sns.countplot(data[data.borough != 'Manhattan'].listing_type, palette=alt_colors)

plt.title('Amount of listing types, not considering Manhattan')

plt.xlabel(None)

plt.show()
# I defined this before data visualization

dup_coord.sort_values(['longitude', 'latitude']).head()
dup_coord.duplicated('listing_id').any()
sns.countplot(dup_coord.listing_type, palette=alt_colors)

plt.title('Amount of listing types in duplicated locations')

plt.xlabel(None)

plt.show()
dup_coord.duplicated('host_id').any()
dup_coord[dup_coord.duplicated('host_id', keep=False)]
sns.countplot(dup_coord.borough, palette=color_palette)

plt.title('Amount of listing types by borough')

plt.xlabel(None)

plt.show()
fig, ax = plt.subplots(figsize=(8, 8))

sns.countplot(y=dup_coord.neighbourhood, hue=dup_coord.borough, palette=color_palette, dodge=False)

plt.title('Amount of listing types by neighbourhood')

plt.ylabel(None)

plt.show()
dup_coord[dup_coord.neighbourhood == 'Ridgewood']
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(dup_coord.corr(), annot=True, cmap='inferno', vmin=-1)

plt.show()
sns.distplot(data.yearly_availability)

plt.title('Distribtution of Availability')

plt.show()
print('Minimum Availability: {}\nMaximum Availability: {}'.format(data.yearly_availability.min(),

                                                                  data.yearly_availability.max()))
print('Unique availability values: {}'.format(len(data.yearly_availability.unique())))
sns.boxplot(data.yearly_availability)

plt.show()
print('Percentage of 0 day availability listings: {}%'.format(

    round(

        data[data.yearly_availability == 0].shape[0] / data.shape[0], 4) * 100

    )

)
np.percentile(data.yearly_availability, [50, 75])
sns.boxplot(x=data.minimum_nights)

plt.show()
data.minimum_nights.describe()
for days in (2, 7, 14, 30, 60, 90, 180):

    long_stays = data[data.minimum_nights >= days]

    print("Listings with a minimum of {} days: {}".format(days, long_stays.shape[0]))
long_stay = data[data.minimum_nights > 30]
print('Listings with over 30 days: {}'.format(long_stay.shape[0]))
sns.distplot(long_stay.minimum_nights, kde=False)

plt.title('Distribution of Long-Term Listings')

plt.show()
long_stay.head()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

fig.tight_layout(pad=5.0)



sns.countplot(long_stay.borough, ax=ax1, palette=color_palette)

sns.countplot(data.borough, ax=ax2, palette=color_palette)



ax1.set_title('Long-Term')

ax1.set(xlabel=None)

ax2.set_title('Overall')



plt.xlabel(None)

plt.show()
sns.countplot(long_stay.listing_type, palette=alt_colors)

plt.title('Listing Types of Long-Term Listings')

plt.xlabel(None)

plt.show()
long_stay[long_stay.listing_type == 'Shared room']
long_stay[long_stay.host_id == '272247972']
long_stay[long_stay.listing_type == 'Private room'].sample(frac=1).head(10)
long_stay[long_stay.listing_type == 'Entire home/apt'].sample(frac=1).head(10)
repeat_long = long_stay[long_stay.duplicated(subset='host_id', keep=False)]

print('Long-Term listings with repeated hosts: {}'.format(repeat_long.shape[0]))
print('Number of unique hosts: {}'.format(len(repeat_long.host_id.unique())))
repeat_long[repeat_long.duplicated(subset=['longitude', 'latitude'], keep=False)]
sns.countplot(repeat_long.listing_type, palette=alt_colors)

plt.title('Listing Types of Long-Term Stays with Repeated Hosts')

plt.xlabel(None)

plt.show()
sns.countplot(repeat_long.borough, palette=color_palette)

plt.title('Listing Types of Long-Term Stays with Repeated Hosts')

plt.xlabel(None)

plt.show()
for borough in repeat_long.borough.unique():

    b_df = repeat_long[repeat_long.borough == borough]

    print('Number of unique hosts in {}: {} for a total of {} listings.'.format(borough, len(b_df.host_id.unique()), len(b_df)))
unique_hosts = len(data.host_id.unique())
print('There are a total of {} unique hosts.'.format(unique_hosts))
dup_hosts_df = data[data.duplicated(subset='host_id', keep=False)]
dup_hosts = len(dup_hosts_df.host_id.unique())
print('There are a total of {} hosts within more than one listing.'.format(dup_hosts))
print('The percentage of hosts within more than one listing is {}%.'.format(

    round(dup_hosts / unique_hosts, 4) * 100)

)
fig = plt.figure(figsize=(16,4))

sns.countplot(data.host_id.value_counts())

plt.xlabel("Amount of listings")

plt.title("Value counts of how many listings a host has")

plt.show()
# possible values of number of listings

len(data.host_id.value_counts().unique())
def get_big_boy(idx):

    return data.host_id.value_counts().index[idx]
data[data.host_id == get_big_boy(0)].head()
data[data.host_id == get_big_boy(1)].head()
data[data.host_id == get_big_boy(2)].head()
def get_random_host():

    random_listing = dup_hosts_df.sample(n=1)

    return dup_hosts_df[dup_hosts_df.host_id == int(random_listing.host_id)].head(10).sort_index()
get_random_host()
get_random_host()
get_random_host()
get_random_host()
dup_hosts_df.listing_type.value_counts()
sns.countplot(dup_hosts_df.listing_type)

plt.xlabel(None)

plt.show()
dup_host_locs = [len(host_df.neighbourhood.unique()) for host, host_df in dup_hosts_df.groupby('host_id')]
np.mean(dup_host_locs)
np.median(dup_host_locs)
np.max(dup_host_locs)
np.max([len(host_df.borough.unique()) for host, host_df in dup_hosts_df.groupby('host_id')])
sns.distplot(data.total_reviews, kde=False)

plt.show()
data.total_reviews.describe()
sns.distplot(data[data.total_reviews > 100].total_reviews, kde=False)

plt.show()
data[data.total_reviews > 450]
sns.distplot(data.monthly_reviews, kde=False)

plt.show()
data.monthly_reviews.describe()
data[data.monthly_reviews >= 20]
data[data.host_id == 244361589].sort_values('listing_id')
def has_gender(text):

    return re.search(r'\b(female[s]{0,1}|male[s]{0,1}|girl[s]{0,1}|boy[s]{0,1}|m[ae]n|wom[ae]n)\b', text)
gender_df = pd.DataFrame([data.iloc[idx] for idx, txt in enumerate(data.title) if has_gender(txt.lower())])
print("Listings that have a mention of gender in their title: {}".format(gender_df.shape[0]))
gender_df.sample(n=5)
def is_male(text):

    return re.search(r'\b(male[s]{0,1}|boy[s]{0,1}|m[ae]n)\b', text)
def is_female(text):

    return re.search(r'\b(female[s]{0,1}|girl[s]{0,1}|wom[ae]n)\b', text)
female_listings = [txt for txt in gender_df.title if is_female(txt.lower())]

print("Female Only lisings: {} ({}% of gender specific listings)".format(

    len(female_listings), 

    round(len(female_listings) / len(gender_df), 4) * 100

    )

)
male_listings = [txt for txt in gender_df.title if is_male(txt.lower())]

print("Male Only lisings: {} ({}% of gender specific listings)".format(

    len(male_listings), 

    round(len(male_listings) / len(gender_df), 4) * 100

    )

)
sns.countplot(gender_df.listing_type)

plt.xlabel(None)

plt.show()
gender_df[gender_df.listing_type == 'Entire home/apt'].title.values
title_lengths = np.array([len(txt.split()) for txt in data.title.values])
def describe(lst):

    print('Mean: {}'.format(round(np.mean(lst), 2)))

    print('25%: {}\n50%: {}\n75%: {}'.format(*np.percentile(title_lengths, [25, 50, 75])))

    print('Minimum: {}'.format(np.min(lst)))

    print('Maximum: {}'.format(np.max(lst)))
describe(title_lengths)
sns.boxplot(title_lengths)

plt.title('Potential title length outliers')

plt.show()
dict(zip(*np.unique(title_lengths, return_counts=True)))
print(*pd.concat([data.iloc[idx] for idx in np.where(title_lengths > 13)], axis=1).title.values, sep='\n\n')
print(*pd.concat([data.iloc[idx] for idx in np.where(title_lengths < 2)], axis=1).title.values[:10], sep='\n\n')
wordcloud = WordCloud(

    background_color='white',

    stopwords=nltk.corpus.stopwords.words('english'),

    max_words=100,

    max_font_size=40,

    scale=3,

    colormap='Dark2',

    random_state=42

)
fig = plt.figure(1, figsize=(12, 12))

plt.axis('off')



plt.imshow(wordcloud.generate(" ".join(data.title.values)))

plt.show()
remove_words = ['nyc', 'new york city', 'apt', 'home', 'apartment', 'apt.', 'bedroom', 'hotel', 'room']

remove_words.extend(data.borough.str.lower().unique())

remove_words.extend(data.neighbourhood.str.lower().unique())
all_text = " ".join(data.title.values).lower()

for string in remove_words:

    all_text = re.sub(string, '', all_text)
fig = plt.figure(1, figsize=(12, 12))

plt.axis('off')



plt.imshow(wordcloud.generate(all_text))

plt.show()
sns.distplot(data.price, kde=False)

plt.title('Prices')

plt.ylabel('amount')

plt.show()
data.price.describe()
sns.distplot(data[data.price < 500].price, kde=False)

plt.title('Prices')

plt.ylabel('amount')

plt.show()
sns.boxplot(data[data.price < 500].price)

plt.show()
sns.distplot(data[data.price < 40].price, kde=False)

plt.title('Prices')

plt.ylabel('amount')

plt.show()
data[data.price == 0]
sns.distplot(data[data.price > 1000].price, kde=False)

plt.title('Prices')

plt.ylabel('amount')

plt.show()
data[data.price > 6000]