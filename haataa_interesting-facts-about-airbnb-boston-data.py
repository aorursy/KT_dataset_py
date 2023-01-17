# import packages

import datetime

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

#from mpl_toolkits.basemap import Basemap



%matplotlib inline
# import data

bs_reviews = pd.read_csv("../input/reviews.csv")

bs_reviews.head()
bs_listing = pd.read_csv("../input/listings.csv")

bs_listing.head(2)
# have a closer look at listing data

bs_listing.info()
bs_calendar = pd.read_csv("../input/calendar.csv")

bs_calendar.head()

bs_calendar.info()

# str to datetime

to_datetime = lambda x: datetime.datetime.strptime(str(x),'%Y-%m-%d')

# remove sign

def remove_sign(x,sign):

    if type(x) is str:

        x = float(x.replace(sign,'').replace(',',''))

    return x
# clean calendar data

## change column 'date' data type from 'object' to 'datetime'

bs_calendar.date = bs_calendar.date.apply(to_datetime)

## remove '$' sign from price

bs_calendar.price = bs_calendar.price.apply(remove_sign,sign='$')

bs_calendar.info()
# clean listing data

bs_listing.host_since = bs_listing.host_since.apply(to_datetime)

bs_listing.price = bs_listing.price.apply(remove_sign,sign='$')

bs_listing.host_response_rate = bs_listing.host_response_rate.apply(remove_sign,sign='%')

bs_listing.host_acceptance_rate = bs_listing.host_acceptance_rate.apply(remove_sign,sign='%')
bs_listing.info()
# total number of available home each day

avaliable_count_bs = bs_calendar.groupby('date').apply(lambda x: x.notnull().sum())[['price']]

# change column name

avaliable_count_bs = avaliable_count_bs.rename({"price":"total_available_houses"},axis='columns')
# everyday average prices

bs_calendar_open = bs_calendar[bs_calendar.price.notnull()]

# average house price for boston everyday

average_price_bs = bs_calendar_open.groupby('date').mean()[['price']]

# change column name

average_price_bs = average_price_bs.rename({"price":"average_prices"},axis='columns')
# plot total available houses and average prices in one figure

f, ax = plt.subplots(figsize=(15, 6))

plt1 = sns.lineplot(x = avaliable_count_bs.index,y = 'total_available_houses', 

                  data = avaliable_count_bs,color="r",legend=False)

for tl in ax.get_yticklabels():

    tl.set_color('r')



ax2 = ax.twinx()

plt2 = sns.lineplot(x = average_price_bs.index,y = 'average_prices',

             data=average_price_bs,ax=ax2,linestyle=':', legend=False)
# take a closer look at the price spike before 05/2017

average_price_bs_sub = average_price_bs[average_price_bs.index > '2017-03-1']

average_price_bs_sub[average_price_bs_sub.average_prices == max(average_price_bs_sub.average_prices)]
# average avaliablity for each house

avaliable_days_bs = bs_calendar.groupby('listing_id').apply(lambda x: x.notnull().mean())[['price']]

# change column name

avaliable_days_bs = avaliable_days_bs.rename({"price":"avaliable_ratio"},axis='columns')
#avaliable_days_bs.head()

f, ax = plt.subplots(figsize=(15, 6))

ax = sns.distplot(avaliable_days_bs, kde=False)

ax.set_xlabel('avaliable ratio', fontsize=10)

ax.set_ylabel('count', fontsize=10)

# a large portion of houses are available for a small portion of days
# check general price distribution

bs_calendar_open.describe()
sns.distplot(bs_calendar_open.price, kde=False)

# price large than 1000 is pretty rare
# visualize price change pattern for a particular listing 

list_price_eg = bs_calendar_open[bs_calendar_open.listing_id == 14421304]

sns.lineplot(x="date",y="price", data=list_price_eg)
# host since house count

bs_cumhost = bs_listing.groupby('host_since').count()[['id']]

# change column name

bs_cumhost = bs_cumhost.rename({"id":"house_num"},axis='columns')
# get cumulative house numbers

bs_cumhost['cum_house_num'] = bs_cumhost.house_num.cumsum()

bs_cumhost.head()
f, ax = plt.subplots(figsize=(12, 9))

sns.lineplot(data=bs_cumhost)
# creat year_month column to group district growth into year_month period

bs_listing['year_month'] = bs_listing.host_since.dt.to_period('M')
def cum_listing_for_cat(colname,bs_listing=bs_listing):

    '''

    This function is used to get the cummulative listing on different categorys over year_month

    

    INPUT:

    colname - str, the categorical column you want to cummulate

    bs_listing - data.frame,bs listing data

    

    OUTPUT:

    df - a new dataframe that has the following columns:

            1. year_month

            2. colname,indicate category

            3. cum_listings

    '''

    col_listnum = bs_listing.groupby(['year_month',colname]).count()[['id']]

    col_listnum.reset_index(inplace=True)

    # calculate cumulative listing for each category

    col_listnum = col_listnum.sort_values(by=[colname,'year_month'])

    col_listnum['cumnum'] = col_listnum.groupby(by=[colname])['id'].apply(lambda x: x.cumsum())

    # drop unneeded column

    col_listnum.drop(['id'],inplace=True,axis=1)

    # long to wide

    # notice that for each category there are some year_month without any record

    # need to flattern the data make sure every neighborhood has all the year_month record.

    # fill na with the last non na value

    col_listnum_wide = col_listnum.pivot('year_month', colname)

    col_listnum_wide = col_listnum_wide.fillna(method='ffill')

    col_listnum_wide = col_listnum_wide.fillna(0)

    # rename columns prepare for wide to long

    col_listnum_wide.rename(columns=lambda x: 'col_'+x, inplace=True)

    # reset index

    col_listnum_wide.reset_index(inplace=True)

    # remove first row

    col_listnum_wide_clean = col_listnum_wide['col_cumnum']

    col_listnum_wide_clean['year_month'] = col_listnum_wide['year_month']

    # change this wide data to long again for drawing figures

    col_listnum_wide_long = col_listnum_wide_clean.melt('year_month', var_name='cols',  value_name='vals')

    col_listnum_wide_long = col_listnum_wide_long.rename(columns={"cols": colname, "vals": "cum_listings"})

    return col_listnum_wide_long
neighbor_cum_listing = cum_listing_for_cat('neighbourhood_cleansed',bs_listing=bs_listing)

neighbor_cum_listing.head()
f, ax = plt.subplots(figsize=(20, 9))

# change year_month column from type datetime to str to draw figure

neighbor_cum_listing.year_month = neighbor_cum_listing.year_month.apply(lambda x:str(x))

# make sure legend order is ordered by cum listing values

last_date = neighbor_cum_listing[neighbor_cum_listing.year_month==max(neighbor_cum_listing.year_month)]

hue_order = last_date.sort_values(by=['cum_listings'],ascending=False)['neighbourhood_cleansed']

g=sns.lineplot(x='year_month',y='cum_listings',hue='neighbourhood_cleansed', hue_order =hue_order ,

               data=neighbor_cum_listing)

g=g.set_xticklabels(neighbor_cum_listing.year_month.unique(),rotation=90)
# how different price level homes increase in numbers over the years?

# first see price distribution

bs_listing.price.describe()
# use 25% price value as low price bar; 75% price value as high price bar

def price_level(x,low_bar=85,high_bar=220):

    if x<=low_bar:

        x='Low_Price'

    elif x>=high_bar:

        x='High_Price'

    else:

        x='Medium_Price'

    return x

bs_listing['price_level'] = bs_listing.price.apply(price_level)
price_level_listnum_wide_long = cum_listing_for_cat('price_level',bs_listing=bs_listing)
f, ax = plt.subplots(figsize=(20, 9))

# change year_month column from type datetime to str to draw figure

price_level_listnum_wide_long.year_month = price_level_listnum_wide_long.year_month.apply(lambda x:str(x))

g=sns.lineplot(x='year_month',y='cum_listings',hue='price_level',style='price_level',

               data=price_level_listnum_wide_long)

g=g.set_xticklabels(price_level_listnum_wide_long.year_month.unique(),rotation=90)
# select price and ratings and dropna

bs_price_rate = bs_listing[["id","price","review_scores_rating","number_of_reviews","price_level"]].dropna()

bs_price_rate.head()
f, ax = plt.subplots(figsize=(15, 6))

sns.scatterplot(x='price',y='review_scores_rating',hue='number_of_reviews',alpha=0.5,data=bs_price_rate)
price_level_review_num = bs_price_rate.groupby('price_level').mean()

price_level_review_num
bs_price_rate.describe()
# find the listing with the highest number of reviews

max_id = bs_price_rate[bs_price_rate.number_of_reviews == max(bs_price_rate.number_of_reviews)]

max_id
# info of this listing

bs_listing[bs_listing.id== 66288].T
bs_calendar[bs_calendar.listing_id==66288 ].count()
# keep listings with not null prices

bs_listing_price = bs_listing[bs_listing.price.notnull()]
def plot_price_by_cat(colname,bs_listing=bs_listing,fig_row_size=11,fig_col_size=9):

    price_col = bs_listing_price.groupby(colname).mean()[['price']]

    price_col.reset_index(inplace=True)

    f, ax = plt.subplots(figsize=(fig_row_size, fig_col_size))

    sns.barplot(y=colname,x='price',data=price_col.sort_values(by='price', ascending=False))
# price and neighborhood

plot_price_by_cat('neighbourhood_cleansed',bs_listing=bs_listing_price)
plot_price_by_cat('property_type',bs_listing=bs_listing_price)
plot_price_by_cat('host_response_time',bs_listing=bs_listing_price,fig_row_size=10,fig_col_size=8)
# price and bed_type

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)

price_bed_type = bs_listing_price.groupby('bed_type').mean()[['price']]

price_bed_type.reset_index(inplace=True)

sns.barplot(x='bed_type',y='price',data=price_bed_type.sort_values(by='price', ascending=False),ax=ax1)

# price and bed_type

price_room_type = bs_listing_price.groupby('room_type').mean()[['price']]

price_room_type.reset_index(inplace=True)

sns.barplot(x='room_type',y='price',data=price_room_type.sort_values(by='price', ascending=False),ax=ax2)
# see how numerical values correlate with price

bs_listing_price_num = bs_listing_price.select_dtypes(include=['float64','int'])
bs_listing_price_num.info()
# drop irrelevent colunms

bs_listing_price_num = bs_listing_price_num.drop(['id','scrape_id','host_id','latitude','longitude',

                                                  'jurisdiction_names','neighbourhood_group_cleansed',

                                                  'license','has_availability','neighbourhood_group_cleansed'], axis=1)
corr = bs_listing_price_num.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr,mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})