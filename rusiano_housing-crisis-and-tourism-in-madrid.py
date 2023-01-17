%matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import scipy.stats as sps    



from langdetect import detect

def mydetect(text):

    try:

        return detect(text)

    except:

        return np.nan



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
listings_detail_df = pd.read_csv(

    '/kaggle/input/madrid-airbnb-data/listings_detailed.csv',

    true_values=['t'], false_values=['f'], na_values=[None, 'none'])



listings_detail_df[['price', 'cleaning_fee']] = (

    listings_detail_df[['price', 'cleaning_fee']]

    .apply(lambda col: (

        col

        .str[1:]

        .str.replace(',', '')

        .apply(float)))

    .fillna(0))



listings_detail_df['minimum_cost'] = (

    listings_detail_df['price'] * 

    listings_detail_df['minimum_nights'] + 

    listings_detail_df['cleaning_fee'])



listings_detail_df['minimum_cost_per_night'] = (

    listings_detail_df['minimum_cost'] /

    listings_detail_df['minimum_nights'])



listings_detail_df['minimum_cost_per_night_and_person'] = (

    np.round(

        listings_detail_df['minimum_cost_per_night'] /

        listings_detail_df['accommodates'], 2))



listings_detail_df['n_amenities'] = (

    listings_detail_df['amenities']

    .str[1:-1]

    .str.replace("\"", '')

    .str.split(',')

    .apply(len))



amenities_srs = (

    listings_detail_df

    .set_index('id')

    ['amenities']

    .str[1:-1]

    .str.replace("\"", '')

    .str.split(',', expand=True)

    .stack())



listings_detail_df['accommodates_group'] = (

    listings_detail_df['accommodates']

    .pipe(pd.cut, bins=[1,2,3,5,20], include_lowest=True, right=False, 

          labels=['Single', 'Couple', 'Family', 'Group']))



listings_lite_df = listings_detail_df[[

    'id', 'host_id', 'listing_url', 'room_type', 'neighbourhood_group_cleansed', 

    'price', 'cleaning_fee', 'accommodates', 'accommodates_group',

    'minimum_nights', 'minimum_cost', 

    'minimum_cost_per_night', 'minimum_cost_per_night_and_person',

    'n_amenities', 'review_scores_rating', 

    'latitude', 'longitude', 'is_location_exact']].copy()



# listings_lite_df.head()
fig, ax = plt.subplots(figsize=(6,6))

pie_data = (

    listings_lite_df['room_type']

    .value_counts())

(pie_data

 .plot(kind='pie', 

       autopct=lambda v: (

           '{}'.format(int(v/100*sum(pie_data))) +

           '\n' * int(v > 10) + ' ' * int(v <= 10) +

           '({:.1%})'.format(v/100)),

       explode=(0.01, ) * len(pie_data),

       ax=ax))

ax.set_ylabel('')

ax.set_title('% of Listings per Room Type', weight='bold')

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))



n_listings_per_district = (

    listings_lite_df['neighbourhood_group_cleansed']

     .value_counts())



(n_listings_per_district.iloc[::-1]).plot(kind='barh', ax=ax1)



ax1.grid(axis='x')

ax1.set_title('# Listings per District', weight='bold')

ax1.set_ylabel('District')

ax1.set_xlabel('# Listings')



pie_data = (

    n_listings_per_district

    .rename(index={v: 'Other' for v in n_listings_per_district.index[6:]})

    .groupby(level=0, sort=False)

    .sum())

pie2_data = (

    pie_data

    .groupby(lambda v: v if v in ['Centro', 'Other'] else '', sort=False)

    .sum())



(pie_data

 .plot(kind='pie', 

       explode=(0.02, 0, 0, 0, 0, 0, 0.02),

       ax=ax2))

(pie2_data

 .rename(index=lambda v: '')

 .plot(kind='pie', 

       explode=(0.02, 0, 0.02),

       autopct=lambda v: (

           '{}'.format(int(v/100*sum(pie_data))) +

           '\n' + '({:.1%})'.format(v/100)),

       wedgeprops={'alpha': 0},

       ax=ax2))



ax2.set_title('% of Listings per District', weight='bold')

ax2.set_ylabel('')



plt.show()
fig, ax = plt.subplots(figsize=(10, 6))



barplot_data = (

    listings_lite_df

    .groupby(['neighbourhood_group_cleansed', 'room_type'])

    .size()

    .unstack('room_type')

    .fillna(0)

    .apply(lambda row: row / row.sum(), axis=1)

    .sort_values('Entire home/apt')

    .reindex(columns=listings_lite_df['room_type'].value_counts().index))



barplot_data.plot(kind='barh', width=.75, stacked=True, ax=ax)



ax.set_xticks(np.linspace(0,1, 5))

ax.set_xticklabels(np.linspace(0,1, 5))

ax.grid(axis='x', c='k', ls='--')

ax.set_xlim(0,1)



ax.set_ylabel('District')

ax.set_xlabel('# Listings (%)')

ax.legend(loc=(1.01, 0))



ax.set_title('Room Types Proportion\nDistricts Comparison', weight='bold')



plt.show()
import json

from shapely.geometry import MultiPolygon

with open('/kaggle/input/madrid-airbnb-data/neighbourhoods.geojson') as f:

    geojson = json.loads(f.read())

areas = [{'area': MultiPolygon(feature['geometry']['coordinates'], context_type='geojson').area, 

          **feature['properties']} for feature in geojson['features']]

areas_srs = (

    pd.DataFrame(areas)

    .groupby('neighbourhood_group')

    ['area']

    .sum() * 10**4)



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))



(n_listings_per_district.iloc[::-1]).plot(kind='barh', ax=ax1)



ax1.grid(axis='x')

ax1.set_title('# Listings per District', weight='bold')

ax1.set_ylabel('District')

ax1.set_xlabel('# Listings')



areas_srs.sort_values().plot(kind='barh', ax=ax2)



ax2.set_title('Neighbourhood Area', weight='bold')

ax2.grid(axis='x')

ax2.set_ylabel('')

ax2.set_xlabel('Area (km2)')



n_listings_per_district.divide(areas_srs).sort_values().plot(kind='barh', ax=ax3)



ax3.set_title('Neighbourhood Density', weight='bold')

ax3.set_xlabel('Density (Listings per km2)')

ax3.grid(axis='x')



scatter_data = pd.concat([n_listings_per_district.rename('n_listings'), areas_srs], axis=1, sort=False)



scatter_data.plot(kind='scatter', x='area', y='n_listings', s=50, ax=ax4)

ax4.set_xlabel('Area (km2)')

ax4.set_ylabel('# Listings')

ax4.set_title('Area vs # of Listings', weight='bold')

ax4.grid(which='both')

ax4.text(*scatter_data.loc['Centro'].values[::-1], ' Centro')

ax4.text(*scatter_data.loc['Fuencarral - El Pardo'].values[::-1], ' Fuencarral - El Pardo ', ha='right')



fig.tight_layout()

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5), sharey=True)



hosts_per_neighbourhood = listings_lite_df.groupby('neighbourhood_group_cleansed')['host_id'].nunique()



listings_per_host_per_neighbourhood = (

    (listings_lite_df[

        listings_lite_df['room_type']

        .isin(['Entire home/apt', 'Hotel room'])]

    .groupby('neighbourhood_group_cleansed')

    .size() / hosts_per_neighbourhood)

    .iloc[::-1]

)



multiple_listings_perc_per_neighbourhood = (

    (listings_lite_df

    [listings_lite_df['room_type'].isin(['Entire home/apt', 'Hotel room'])]

    .groupby(['neighbourhood_group_cleansed', 'host_id'])

    .size().ge(5)

    .groupby('neighbourhood_group_cleansed')

    .sum() / hosts_per_neighbourhood)

    .iloc[::-1]

)



(listings_per_host_per_neighbourhood

 .plot(kind='barh', color=[sns.color_palette()[0] 

                           if n != 'Centro' else 'navy' 

                           for n in listings_per_host_per_neighbourhood.index], ax=ax1))



(multiple_listings_perc_per_neighbourhood

 .plot(kind='barh', color=[sns.color_palette()[0] 

                           if n != 'Centro' else 'navy' 

                           for n in multiple_listings_perc_per_neighbourhood.index], ax=ax2))



ax1.grid(axis='x')

ax2.grid(axis='x')

ax1.set_ylabel('Neighbourhood')

ax1.set_xlabel('Listings per User')

ax2.set_xlabel('Users with 5+ Listings (%)')

ax1.axvline(1, c='k', ls='--', lw=.9)



ax1.set_title('# Homes, Apartments & Hotel Rooms per Host\nNeighbourhoods Comparison', weight='bold')

ax2.set_title('% of Hosts with 5+ Homes/Apartments/Hotel Rooms\nNeighbourhoods Comparison', weight='bold')



fig.tight_layout()

plt.show()
fig, ax = plt.subplots(figsize=(6,6))



n_listings_per_user = (

    listings_lite_df

    .groupby('host_id')

    .size())



pie_data = (

    n_listings_per_user

    .pipe(pd.cut, bins=[1,2,3,5,1000], include_lowest=True, right=False,

       labels=['1', '2', '3-4', '5+']).value_counts())



pie_data.plot(kind='pie',

              explode=(0.01, ) * len(pie_data),

              autopct=lambda v: (

                  '{}'.format(int(v/100*sum(pie_data))) +

                  '\n' + '({:.1%})'.format(v/100)),)



ax.set_ylabel('')

ax.set_title('# of Listings per User', weight='bold')



plt.show()
n_listings_per_user.sort_values().iloc[::-1].head(10).to_frame('n_properties')
fig, ax = plt.subplots(figsize=(6,6))



pie_data = (

    listings_lite_df['host_id']

    .map(listings_lite_df.groupby('host_id').size() > 1).value_counts())



pie_data.plot(

    kind='pie', labels=['Yes', 'No'], colors=['darkgreen', 'firebrick'], 

    autopct=lambda v: (

                  '{}'.format(int(v/100*sum(pie_data))) +

                  '\n' + '({:.1%})'.format(v/100)),

    startangle=90, ax=ax)



ax.set_ylabel('')

ax.set_title('Does the owner rent another place?', weight='bold')



plt.show()
fig, ax = plt.subplots(figsize=(6,6))



pie_data = (

    listings_lite_df[

        listings_lite_df['room_type'].isin(['Entire home/apt', 'Hotel room'])]

    .groupby('host_id')

    .size()

    .gt(1)

    .value_counts())



pie_data.plot(

    kind='pie', labels=['Yes', 'No'], colors=['darkgreen', 'firebrick'], 

    autopct=lambda v: (

                  '{}'.format(int(v/100*sum(pie_data))) +

                  '\n' + '({:.1%})'.format(v/100)),

    startangle=90, ax=ax)



ax.set_ylabel('')

ax.set_title('Does the owner of an entire property\nrent another entire property?', weight='bold')



plt.show()
listings_lite_df['name_lang'] = listings_detail_df['name'].apply(mydetect)
plt.figure(figsize=(10,6))

sns.distplot(listings_lite_df['price'])

plt.title('Prices Distribution', weight='bold')

plt.grid()



plt.show()
listings_lite_df = listings_lite_df[listings_lite_df['price'] < 300]

trans_prices = listings_lite_df['price'].pipe(np.log1p)



fig, ax = plt.subplots(figsize=(10,6))

ax2 = ax.twiny().twinx()



sns.distplot(listings_lite_df['price'], ax=ax, label='True Data')

sns.distplot(trans_prices, color=sns.color_palette()[1], ax=ax2, label='Transformed Data\n(np.log1p)')

ax.set_title('Prices Distribution', weight='bold')

ax.grid()

ax.legend(loc=2)

ax2.legend(loc=1)

ax.set_xlabel('Price')

ax.set_ylabel('Density')



fig.tight_layout()

plt.show()
fig, (ax,ax2) = plt.subplots(1, 2, figsize=(12,6), gridspec_kw={'width_ratios': [1, .6]})



sns.distplot(trans_prices[listings_lite_df['name_lang'] == 'en'], ax=ax, label='English Title')

sns.distplot(trans_prices[listings_lite_df['name_lang'] == 'es'], ax=ax, label='Spanish Title')



ax.set_title('Prices Distribution\nLanguages Comparison', weight='bold')

ax.set_xlabel('Price')

ax.set_ylabel('Density')

ax.grid()

ax.legend()



sns.boxplot(

    x='name_lang', y='trans_price',

    data=(listings_lite_df[listings_lite_df['name_lang'].isin(['es', 'en'])]

          .assign(trans_price=lambda df: df['price'].pipe(np.log1p))),

    order=['en', 'es'], ax=ax2)



ax2.set_title('Prices Distribution\nLanguages Comparison', weight='bold')

ax2.set_xlabel('Language')

ax2.set_xticklabels(['English', 'Spanish'])

ax2.set_ylabel('Price (Transf.)')

ax2.grid(axis='y')



fig.tight_layout()

plt.show()
samples = trans_prices.groupby(listings_lite_df['name_lang']).apply(list).loc[['es', 'en']]

samples_sizes = samples.apply(len)



samples_sizes.plot(kind='bar', color=sns.color_palette()[:2][::-1])

plt.title('# of Listings\nper Title Language', weight='bold')

plt.ylabel('# Listings')

plt.xticks([0,1], ['Spanish','English'],rotation=0)

plt.xlabel('Language')

plt.grid(axis='y')



plt.show()
samples_normality_tests = list(map(sps.normaltest, samples))

samples_normality_tests
sps.levene(*samples)
sps.ttest_ind(*samples, equal_var=False)
reviews_df = (

    pd.read_csv('/kaggle/input/madrid-airbnb-data/reviews_detailed.csv', 

                parse_dates=['date'])

    .rename(columns={'id': 'review_id'})

    .sort_values('date'))



reviews_df
reviews_df2 = (

    reviews_df

    .pipe(pd.merge, (listings_lite_df

                     .rename(columns={'id': 'listing_id'})

                     [['listing_id', 'room_type', 'accommodates_group']])))

reviews_df2 = reviews_df2[reviews_df2['date'].dt.year.between(2011,2018)]
reviews_df = reviews_df[

    reviews_df['comments'].notna() &

    (~reviews_df['comments'].fillna('').str.contains('The host cancel*ed ((my)|(this)) reservation')) &

    (~reviews_df['comments'].fillna('').str.contains('The reservation was cancel*ed'))

].copy()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15,5), gridspec_kw={'width_ratios': [1, .6]})



n_reviews_srs = (

    reviews_df

    .groupby('date')

    .size()

    .reindex(index=pd.date_range(*reviews_df['date'].agg(['min', 'max']).values))

    .fillna(0))



n_reviews_srs.plot(ax=ax)

n_reviews_srs.rolling(365).mean().plot(ax=ax, c='k', ls='--')



ax.set_xlabel('Date')

ax.set_ylabel('# Reviews')

ax.set_title('Reviews Trend', weight='bold')

ax.grid(axis='both')



ax.add_artist(plt.Circle(('2016-08-30',200), radius=150, edgecolor='r', facecolor='None', zorder=3))

ax.add_artist(plt.Circle(('2017-08-30',400), radius=150, edgecolor='r', facecolor='None', zorder=3))

ax.add_artist(plt.Circle(('2018-08-30',600), radius=150, edgecolor='r', facecolor='None', zorder=3))



n_reviews_srs.cumsum().plot(ax=ax2)



ax2.set_xlabel('Date')

ax2.set_ylabel('# Reviews (Cumulative)')

ax2.set_title('Total # of Reviews', weight='bold')

ax2.grid(axis='both')



plt.show()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))



n_reviews_per_year = (

    reviews_df

    .groupby(reviews_df['date'].dt.year)

    .size()

    .sort_index())



plot_data = n_reviews_per_year.copy()

plot_data.iloc[-1] = np.nan

plot_data.plot(kind='bar', ax=ax1)

plot_data = n_reviews_per_year.copy()

plot_data.iloc[:-1] = np.nan

plot_data.plot(kind='bar', ax=ax1, alpha=.4)



ax1.set_xlabel('Year')

ax1.set_ylabel('# Reviews')

ax1.set_title('# of Reviews per Year', weight='bold')

ax1.grid(axis='y')



n_reviews_per_month = (

    reviews_df

    .groupby(reviews_df['date'].dt.month)

    .size()

    .sort_index())

n_reviews_per_month_no2019 = (

    reviews_df[reviews_df['date'] < '2019-01-01']

    .groupby(reviews_df['date'].dt.month)

    .size()

    .sort_index())



n_reviews_per_month.plot(kind='bar', ax=ax2, alpha=0.5, label='With 2019')

n_reviews_per_month_no2019.plot(kind='bar', ax=ax2, label='Without 2019')



ax2.set_xlabel('Month')

ax2.set_ylabel('# Reviews')

ax2.set_title('# of Reviews per Month', weight='bold')

ax2.grid(axis='y')

ax2.legend()





fig.tight_layout()

plt.show()
lambda_diff = lambda srs: (srs - srs.shift()) / srs.shift()



fig, ax = plt.subplots(figsize=(10,6))

kwargs = {'marker': 's', 'ax': ax}



n_reviews_per_year.iloc[1:-1].pipe(lambda_diff).plot(label='Reviews', **kwargs)



(reviews_df2

 .groupby('reviewer_id')

 ['date']

 .min()

 .dt.year

 .value_counts()

 .sort_index()

 .pipe(lambda_diff)).plot(label='New Tourists', **kwargs)



(reviews_df2

 .groupby('listing_id')

 ['date']

 .min()

 .dt.year

 .value_counts()

 .sort_index()

 .pipe(lambda_diff)).plot(label='New Listings', **kwargs)



(reviews_df2

.groupby([reviews_df2['date'].dt.year.rename('year'), 'listing_id'])

.size()

.groupby('year')

.mean()

.pipe(lambda_diff)).plot(label='Reviews per Listing', **kwargs)



ax.legend()

ax.grid()

# ax.set_ylim(-2.15, 2.15)

ax.set_xlim(2010.5, 2018.5)

ax.set_xlabel('Year')

ax.set_ylabel('Yearly Delta (%)')

ax.set_title('Yearly Trends Comparison', weight='bold')



plt.show()
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 12))



n_reviews_nthweek_x_year_srs = (

    n_reviews_srs

    .groupby([

     n_reviews_srs.index.to_series().dt.year.rename('Year'),

     n_reviews_srs.index.to_series().dt.weekofyear.rename('n-th Week')])

    .sum())



(n_reviews_nthweek_x_year_srs

 .iloc[1:-1]

 .rolling(4)

 .mean()

 .unstack('Year')).plot(ax=ax)



months_dts = [dt for dt in pd.date_range('2019-01-01', '2019-12-31') if dt.is_month_start]

ax.set_xticks([dt.weekofyear for dt in months_dts])

ax.set_xticklabels([dt.strftime('%b') for dt in months_dts])

ax.set_xticks(range(1,53), minor=True)

ax.set_xlim(1, 52)

ax.set_ylabel('# of Reviews')

ax.set_title('# of Reviews\nYears Comparison', weight='bold')

ax.legend(title='Year', loc=(1.01, 0))



ax.add_artist(plt.Rectangle((months_dts[5].weekofyear, 7000), width=5, height=1200,

                            facecolor='r', alpha=.4, edgecolor='r', lw=4))

ax.text(months_dts[5].weekofyear - 0.5, 7000, '1st June, 2019\nChampions League Final', ha='right')

ax.axvspan(months_dts[-1].weekofyear, months_dts[-1].weekofyear+2, 

           facecolor='r', alpha=.4, edgecolor='r', lw=4)

ax.text(months_dts[-1].weekofyear - 0.5, 7000, 

        '8th December\nFeast of the Immaculate Conception', ha='right')

ax.grid(axis='both')

ax.grid(axis='x', which='minor', ls='--')



n_reviews_month_x_year_df = (

    n_reviews_srs

    .groupby([

         n_reviews_srs.index.to_series().dt.year.rename('Year'),

         n_reviews_srs.index.to_series().dt.month.rename('Month')])

    .sum()

    .unstack('Year'))

perc_year_reviews_x_month_df = (

    n_reviews_month_x_year_df

    .divide(n_reviews_month_x_year_df.sum()))



boxplot_data = (

    perc_year_reviews_x_month_df

    .iloc[:, 1:-1]

    .stack()

    .to_frame('% Year Reviews')

    .reset_index())



sns.boxplot(x='Month', y='% Year Reviews', data=boxplot_data, ax=ax2)



ax2.set_xticklabels([

    pd.to_datetime(label.get_text(), format='%m').strftime('%b')

    for label in ax2.get_xticklabels()])

ax2.set_yticklabels([int(tick*100) for tick in ax2.get_yticks()])

ax2.set_ylabel('Year Reviews (%)')

ax2.set_title('Distribution of % Year Reviews', weight='bold')

ax2.grid(axis='y')



fig.tight_layout()

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))



listings_lite_df['accommodates'].plot(kind='hist', ax=ax1)



ax1.set_xlabel('# People')

ax1.set_ylabel('# Listings')

ax1.set_title('Rooms Capacity\nDistribution', weight='bold')

ax1.grid(axis='y')



pie_data = (

    listings_lite_df['accommodates_group']

    .value_counts()

    .sort_index())



pie_data.plot(

    kind='pie', 

    labels=['Single\n(1 person)', 'Couple\n(2 people)', 'Family\n(3-4 people)', 'Group\n(5+ people)'],

    autopct=lambda v: (

         '{}'.format(int(v/100*sum(pie_data))) +

         '\n' + '({:.1%})'.format(v/100)),

    startangle=90,

    explode=(0.01, ) * len(pie_data),

    ax=ax2)



ax2.set_ylabel('')



fig.tight_layout()

plt.show()
n_reviews_year_month_x_accgroup_df = (

    reviews_df2

    .groupby([

        reviews_df2['date'].dt.year.rename('Year'),

        reviews_df2['date'].dt.month.rename('Month'),

        'accommodates_group'])

    .size()

    .unstack(['accommodates_group'])

    .fillna(0))



fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))



sns.heatmap(n_reviews_year_month_x_accgroup_df.T.apply(lambda row: row / row.sum(), axis=1), cmap='RdYlGn', ax=ax)



ax.set_ylabel('Accommodation Group')

ax.set_xlabel('Year-Month')

ax.set_title('# of Reviews\n(Group Proportion)', weight='bold')



sns.heatmap(n_reviews_year_month_x_accgroup_df.T.apply(lambda col: col / col.sum()), cmap='RdYlGn', ax=ax2)



ax2.set_ylabel('Accommodation Group')

ax2.set_xlabel('Year-Month')

ax2.set_title('# of Reviews\n(Monthly Proportion)', weight='bold')



fig.tight_layout()

plt.show()
n_reviews_year_month_x_accgroup_df = (

    reviews_df2

    .groupby([

        reviews_df2['date'].dt.year.rename('Year'),

        reviews_df2['date'].dt.month.rename('Month'),

        'accommodates_group'])

    .size()

    .unstack(['accommodates_group'])

    .fillna(0))



znorm_perc_reviews_year_month_x_accgroup_df = (

    n_reviews_year_month_x_accgroup_df

    # per each year and month we compute the proportion of reviews in each "accommodation group"

    .divide(n_reviews_month_x_year_df.stack().swaplevel().sort_index(), axis=0)

    # we normalize this number across months of the same year

    .unstack('Year')

    .apply(lambda col: (col - col.mean()) / col.std())

    .stack('Year')

    .swaplevel()

    .sort_index())



fig, ax = plt.subplots(figsize=(12, 6))



sns.boxplot(x='Month', y='z_perc', hue='accommodates_group', 

            data=(znorm_perc_reviews_year_month_x_accgroup_df

                  .stack()

                  .sort_index()

                  .to_frame('z_perc')

                  .reset_index()), ax=ax)

ax.set_ylabel('Proportion (Z-Norm.)')

ax.set_ylim(-3,3)

ax.set_title('Accommodation Groups Proportion\nDistribution', weight='bold')

ax.grid(axis='y')

ax.add_artist(plt.Rectangle((5.51, -2.9), width=1.95, height=2.9*2, 

                            facecolor=sns.color_palette()[3], edgecolor='k', alpha=.4, zorder=0))

ax.add_artist(plt.Rectangle((7.51, -2.9), width=1.95, height=2.9*2, 

                            facecolor=sns.color_palette()[0], edgecolor='k', alpha=.4, zorder=0))



ax.legend(title='Capacity', loc=(1.01, 0))



plt.show()