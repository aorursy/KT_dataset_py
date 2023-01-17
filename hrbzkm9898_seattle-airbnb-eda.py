# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

os.chdir("/kaggle/input/seattle-airbnb-data-preprocessing/")

listing = pd.read_csv("listings.csv", 

                      parse_dates=['host_since', 'first_review', 'last_review'])

calendar = pd.read_csv("calendar.csv", parse_dates=['date'])
calendar
listing
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

fig, ax1 = plt.subplots(figsize=(15,6))

ax2 = ax1.twinx()

calendar_available = calendar.groupby('date').apply(lambda x: 100 * len(x[x['available']=='t']) / len(x)).to_frame().rename(columns={0:'available'})

calendar_avgPrice = calendar.groupby('date').apply(lambda x: x.loc[x['available']=='t', 'price'].mean()).to_frame().rename(columns={0:'avgPrice'})

sns.lineplot(x = calendar_available.index, y = 'available', ax=ax1,

             data = calendar_available,color='r')

sns.lineplot(x = calendar_avgPrice.index, y = 'avgPrice', ax=ax2,

             data = calendar_avgPrice,color='b')

ax1.set_ylabel("% available")

ax2.set_ylabel("average price")

for tl in ax1.get_yticklabels():

    tl.set_color('r')

for tl in ax2.get_yticklabels():

    tl.set_color('b')

ax1.yaxis.label.set_color('r')

ax2.yaxis.label.set_color('b')

sns.despine(left=True)

plt.title("Percentage of Available Listings & Average Listing Price")
# listing[['host_id', 'host_since', 'num_days_as_host']]

host_growth = listing.groupby('host_id').apply(lambda x: x[['host_since', 'num_days_as_host']].iloc[0]).reset_index().sort_values('host_since')

host_growth = host_growth[host_growth['host_since'].notnull()]

host_count = host_growth.groupby('host_since').apply(len).to_frame().rename(columns={0:'host_count'})

host_count['cum_host'] = host_count['host_count'].cumsum()

fig, ax = plt.subplots(figsize=(13, 6))

sns.lineplot(x = host_count.index, y = 'cum_host', ax=ax,

             data = host_count)

ax.set_title("Number of Hosts in the Seattle Area")

sns.despine(left=True)
host_growth.num_days_as_host.plot.hist()

plt.xlabel("# of days since being an Airbnb Host")
district_listing = listing[['id', 'host_id', 'host_since', 'host_listings_count', 'calculated_host_listings_count', 'neighbourhood_group_cleansed']]

district_listing
fig, ax = plt.subplots(figsize=(13, 6))

district_listing.groupby('neighbourhood_group_cleansed').apply(len).sort_values().plot.barh(ax=ax)

sns.despine(left=True)

ax.set_ylabel("Neighborhood")

ax.set_xlabel("# of listings")
from pandas.tseries.offsets import MonthEnd

district_review = listing[['id', 'neighbourhood_group_cleansed', 'number_of_reviews', 'first_review']]

district_review = district_review[district_review['number_of_reviews']!=0].sort_values('first_review')



def get_cum_listing(district_group):

    '''

    Return monthly number of Airbnb listings by district

    '''

    return district_group.groupby(district_group['first_review'] + MonthEnd(0)).apply(len).asfreq('M').fillna(0).cumsum()



district_ts = district_review.groupby('neighbourhood_group_cleansed').apply(get_cum_listing).unstack().T.ffill().fillna(0)

district_ts['total'] = district_ts.sum(axis=1)

district_ts
fig, ax = plt.subplots(figsize=(13,6))

district_ts[['total', 'Other neighborhoods', 'Capitol Hill', 'Downtown', 'Central Area']].plot(ax=ax)

ax.set_ylabel("Number of Airbnb Listings")

ax.set_xlabel("Date")

ax.set_title("Growth in New Airbnb Listings by District")

sns.despine(left=True)
fig, ax = plt.subplots(figsize=(13,13))

sns.heatmap(district_ts.iloc[:, :-1].T, cmap="Blues", ax=ax)

ax.set_xticklabels(district_ts.index.strftime('%Y-%m'));

ax.set_xlabel("Date")

ax.set_ylabel("District")

ax.set_title("Airbnb Listing Growth by District")
price_rating = listing[['id', 'host_id', 'number_of_reviews','reviews_per_month',

                        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 

                        'review_scores_checkin', 'review_scores_communication', 

                        'review_scores_location', 'review_scores_value', 'price']]

price_rating = price_rating[price_rating['number_of_reviews']!=0]

price_rating['review_metric'] = (price_rating['review_scores_rating'] + price_rating[['review_scores_accuracy', 'review_scores_cleanliness', 

                                                                                      'review_scores_checkin', 'review_scores_communication', 

                                                                                      'review_scores_location', 'review_scores_value']].mean(axis=1) * 10) / 2

price_rating = price_rating[['id', 'host_id', 'number_of_reviews','reviews_per_month',

                             'review_metric', 'price']]

price_rating
fig, ax = plt.subplots(figsize=(13,6))

p = ax.scatter(price_rating['review_metric'], price_rating['price'], s=10, c=price_rating['number_of_reviews'], cmap="Reds")

colorbar = fig.colorbar(p)

colorbar.set_label("Number of Reviews")

sns.despine(left=True)

ax.set_xlabel("Review Score")

ax.set_ylabel("Price")

ax.set_title("Price vs Review Score")
numeric_cols = ['room_type', 'accommodates', 'bathrooms', 'beds', 'bed_type', 

                'square_feet', 'guests_included']

cat_cols = ['host_is_superhost', 'instant_bookable',

            'has_Wireless_Internet', 'has_Essentials',

            'has_Wheelchair_Accessible', 'has_Indoor_Fireplace',

            'has_Carbon_Monoxide_Detector', 'has_Cats', 'has_Safety_Card',

            'has_Shampoo', 'has_Hot_Tub', 'has_Other_pets', 'has_Pets_Allowed',

            'has_Gym', 'has_First_Aid_Kit', 'has_BuzzerWireless_Intercom',

            'has_Doorman', 'has_Suitable_for_Events', 'has_Fire_Extinguisher',

            'has_Dogs', 'has_Pets_live_on_this_property', 'has_Breakfast',

            'has_Heating', 'has_Laptop_Friendly_Workspace', 'has_Smoking_Allowed',

            'has_Internet', 'has_Air_Conditioning', 'has_24-Hour_Check-in',

            'has_TV', 'has_Elevator_in_Building', 'has_Pool', 'has_Cable_TV',

            'has_Free_Parking_on_Premises', 'has_Washer', 'has_Iron',

            'has_FamilyKid_Friendly', 'has_Kitchen', 'has_Dryer',

            'has_Smoke_Detector', 'has_Hair_Dryer', 'has_Lock_on_Bedroom_Door',

            'has_Hangers', 'property_type', 'neighbourhood_group_cleansed']
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams["figure.figsize"] = (13,6)

result = seasonal_decompose(calendar_avgPrice, model='multiplicative', freq=7)

fig = result.plot();
result.seasonal.groupby(result.seasonal.index.dayofweek+1).mean().plot()

plt.xlabel("Day of Week")

plt.title("Day of Week Effect (Multiplicative Effect on Price)")

sns.despine(left=True)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

fig.suptitle("Average vs Median Price by District", fontsize=20)

rank = listing.groupby('neighbourhood_group_cleansed')['price'].mean().sort_values(ascending=False).index

sns.barplot(x='price', y='neighbourhood_group_cleansed', data=listing, order=rank, ax=ax1)

ax1.set_ylabel("Neighborhood")

ax1.set_xlabel("Average Price")

rank = listing.groupby('neighbourhood_group_cleansed')['price'].median().sort_values(ascending=False).index

sns.barplot(x='price', y='neighbourhood_group_cleansed', data=listing, estimator=np.median, order=rank, ax=ax2)

ax2.set_ylabel("")

ax2.set_xlabel("Median Price")

sns.despine(left=True)
sns.boxplot(x='price', y='neighbourhood_group_cleansed', data=listing, order=rank)

sns.despine(left=True)
listing[numeric_cols]