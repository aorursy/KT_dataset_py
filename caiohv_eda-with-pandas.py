import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
def print_resume_null_values(df):

    print(df.shape)

    tb_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'Types'})

    tb_info = tb_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0: 'Null Values'}))

    tb_info = tb_info.append(pd.DataFrame(round(df.isnull().sum()/df.shape[0] * 100, 2)).T.rename(index={0: 'Null Values (%)'}))

    return tb_info
print_resume_null_values(df)
df['host_id'] = df['host_id'].astype(int)

df['minimum_nights'] = df['minimum_nights'].astype(int)

df['number_of_reviews'] = df['number_of_reviews'].astype(int)

df['calculated_host_listings_count'] = df['calculated_host_listings_count'].astype(int)

df['availability_365'] = df['availability_365'].astype(int)

df['price'] = df['price'].astype(float)

df['reviews_per_month'] = df['reviews_per_month'].astype(float)

df['last_review'] = pd.to_datetime(df['last_review'])
df['month'] = 0

for index, item in df.loc[df['last_review'].notnull()].iterrows():

  df.loc[df['id'] == item['id'], 'month'] = int(item['last_review'].month)
df['year'] = 0

for index, item in df.loc[df['last_review'].notnull()].iterrows():

  df.loc[df['id'] == item['id'], 'year'] = int(item['last_review'].year)
df.loc[df['reviews_per_month'].isnull(), 'reviews_per_month'] = df['reviews_per_month'].median()

df['reviews_per_month']
df.month.mode()
df.loc[df.month == 0, 'month'] = 6
df.year.mode()
df.loc[df.year == 0, 'year'] = 2019
df.loc[df.price == 0, 'price'] = df.price.median()
df.loc[df.number_of_reviews == 0, 'number_of_reviews'] = df.number_of_reviews.median()
df.loc[df.availability_365 == 0, 'availability_365'] = df.availability_365.median()
df.head()
airbnb_df = df[['neighbourhood_group', 'neighbourhood', 'latitude', 

                'longitude', 'room_type', 'price',

                'minimum_nights', 'number_of_reviews', 'reviews_per_month',

                'calculated_host_listings_count', 'availability_365', 'month', 

                'year']]
airbnb_df = airbnb_df.sort_values(by='year')

airbnb_df.head()
print_resume_null_values(airbnb_df)
plt.figure(figsize=(14, 7))

ax = sns.barplot(x="year", y="price", data=airbnb_df, 

                 estimator=np.median, color="salmon")

ax.axes.set_title("Price between 2011 to 2019 (median)", fontsize=25)

ax.set_xlabel("Year",fontsize=18)

ax.set_ylabel("Price",fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.barplot(x="month", y="price", data=airbnb_df, 

                 estimator=np.median, color="salmon")

ax.axes.set_title("Price between Jan to Dec during 2011 to 2019 (median)", fontsize=25)

ax.set_xlabel("Month",fontsize=18)

ax.set_ylabel("Price",fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.barplot(x='room_type', y='price', data=airbnb_df, estimator=np.median)

ax.axes.set_title("Price x Room type (median)", fontsize=25)

ax.set_xlabel("Room Type", fontsize=18)

ax.set_ylabel("Price", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.violinplot(x='room_type', y='price', data=airbnb_df.loc[airbnb_df.price < 500])

ax.axes.set_title("Density and distribution of prices for each room types", fontsize=25)

ax.set_xlabel("Room Type", fontsize=18)

ax.set_ylabel("Price", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.barplot(x='neighbourhood_group', y='price', data=airbnb_df, estimator=np.median)

ax.axes.set_title("Accumulated price average by Neighbourhood Group", fontsize=25)

ax.set_xlabel("Neighbourhood Group", fontsize=18)

ax.set_ylabel("Price", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.violinplot(x='neighbourhood_group', y='price', data=airbnb_df.loc[airbnb_df.price < 500])

ax.axes.set_title("Density and distribution of price for each Neighbourhood Group", fontsize=25)

ax.set_xlabel("Neighbourhood Group", fontsize=18)

ax.set_ylabel("Price", fontsize=18)
plt.figure(figsize=(26, 10))

ax = sns.barplot(x='neighbourhood', y='price',

                 data=airbnb_df.sort_values(by=['price'], ascending=False).iloc[0:28],

                 estimator=np.median)



ax.axes.set_title("Top 20 Neighbourhoods more expensives", fontsize=25)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.set_xlabel("Neighbourhood Group", fontsize=18)

ax.set_ylabel("Price", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.catplot(x="neighbourhood_group", y="price", 

                 hue='room_type', data=airbnb_df,

                 kind="bar", height=8, aspect=2)

ax.set_axis_labels('', 'Price')
plt.figure(figsize=(14, 7))

ax = sns.violinplot(x="neighbourhood_group", y="number_of_reviews", data=airbnb_df.loc[airbnb_df.number_of_reviews < 200])

ax.axes.set_title("Density and distribution of Number of Reviews for each Neighbourhood Group", fontsize=20)

ax.set_xlabel("Neighbourhood Group", fontsize=18)

ax.set_ylabel("Number of Reviews", fontsize=18)
plt.figure(figsize=(18, 8))

ax = sns.barplot(x="neighbourhood", y="number_of_reviews",

                 data=airbnb_df.sort_values(by=['number_of_reviews'],

                                            ascending=False).iloc[:20])



ax.set_title("Number of Reviews x Neighbourhoods (Most results)", fontsize=25)

ax.set_xlabel("Neighbourhood Group", fontsize=18)

ax.set_ylabel("Number of Reviews", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.barplot(x="room_type", y="number_of_reviews", data=airbnb_df)



ax.set_title("Number of reviews x Room Type", fontsize=25)

ax.set_xlabel("Room Type", fontsize=18)

ax.set_ylabel("Number of reviews", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.scatterplot(x="number_of_reviews", y="price", 

                    size='price', sizes=(20, 200), 

                    data=airbnb_df)



ax.set_xlabel("Number of Reviews", fontsize=18)

ax.set_ylabel("Price", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.barplot(x="neighbourhood_group", y="minimum_nights", hue="room_type", data=airbnb_df)

ax.axes.set_title("Minimum Nights x Neighbourhood Group", fontsize=25)

ax.set_xlabel("Neighbourhood Group", fontsize=18)

ax.set_ylabel("Minimum Nights", fontsize=18)
plt.figure(figsize=(28, 10))

ax = sns.barplot(x="neighbourhood", y="minimum_nights",

                 data=airbnb_df.sort_values(by=['minimum_nights'],

                                            ascending=False).iloc[:28])



ax.set_title("Minimum nights x Neighbourhoods (Most results)", fontsize=25)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.set_xlabel("Neighbourhood Group", fontsize=18)

ax.set_ylabel("Minimum nights", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.barplot(x="room_type", y="minimum_nights", data=airbnb_df)



ax.set_title("Room Type x Minimum Nights", fontsize=25)

ax.set_xlabel("Room Type", fontsize=18)

ax.set_ylabel("Minimum Nights", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.violinplot(x="room_type", y="minimum_nights", data=airbnb_df.loc[airbnb_df.minimum_nights < 100])

ax.set_title("Density and distribution of Room Type for each Minimum Nights", fontsize=25)

ax.set_xlabel("Room Type", fontsize=18)

ax.set_ylabel("Minimum Nights", fontsize=18)
plt.figure(figsize=(14, 7))

ax = sns.scatterplot(x="minimum_nights", y="price",

                     hue='price', size='price',

                     sizes=(20, 200), data=airbnb_df)



ax.set_title("Minimum Nights x Price", fontsize=25)

ax.set_xlabel("Minimum Nights", fontsize=18)

ax.set_ylabel("Price", fontsize=18)
data = airbnb_df.drop(['neighbourhood', 'number_of_reviews', 

                       'reviews_per_month', 'latitude', 'longitude', 

                       'month', 'year'], axis=1)
data.to_csv('airbnb_ny_ready_to_ml.csv', index=False)
import pandas as pd

AB_NYC_2019 = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")