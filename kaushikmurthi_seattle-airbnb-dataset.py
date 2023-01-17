# Load all necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



# Update style for all plots

plt.style.use('fivethirtyeight')
# Import the datasets

listings_df = pd.read_csv('/kaggle/input/seattle/listings.csv')

calendar_df = pd.read_csv('/kaggle/input/seattle/calendar.csv')

reviews_df = pd.read_csv('/kaggle/input/seattle/reviews.csv')
# calendar dataset

calendar_df.info()
calendar_df.sample(5)
# Fill missing price with zero

calendar_df['price'].fillna('$0', inplace=True)

# Remove commas in price

calendar_df['price'] = calendar_df['price'].apply(lambda x:''.join(x.split(',')))

# Convert price to numeric data

calendar_df['price'] = calendar_df['price'].apply(lambda x:float(x.split('$')[1]))
# Convert available to numeric data

calendar_df['available'] = calendar_df['available'].apply(lambda x:1 if x=='t' else 0)

# Split month into a separate column

calendar_df['month'] = calendar_df['date'].apply(lambda x: x.split('-')[1])

calendar_df.sample(5)
# check if the number of listings was the same every day

calendar_df.groupby(['listing_id']).count()['date'].unique()
# Groupby by month sum

calendar_df_month_sum = calendar_df.groupby(['month']).sum()

calendar_df_month_sum.drop(['listing_id'],axis=1,inplace=True)

calendar_df_month_sum.reset_index(inplace=True)

calendar_df_month_sum
plt.figure(figsize=(16,8))

sb.lineplot(x='month',y='available',data=calendar_df_month_sum)

plt.xlabel('Month')

plt.xticks(rotation=90)

plt.ylabel('Availability')

plt.ylim(50000,100000)

plt.title('Month vs Availablity')

plt.show()
# Groupby by month average

calendar_df_month_avg = calendar_df.groupby(['month']).mean()

calendar_df_month_avg.drop(['listing_id'],axis=1,inplace=True)

calendar_df_month_avg.reset_index(inplace=True)

calendar_df_month_avg
plt.figure(figsize=(16,8))

sb.lineplot(x='month',y='price',data=calendar_df_month_avg)

plt.xlabel('Month')

plt.xticks(rotation=90)

plt.ylabel('Average Price($)')

plt.ylim(60,120)

plt.title('Month vs Average Price')

plt.show()
# listings dataset

listings_df.info()
listings_df.sample(5)
# find missing data

listings_df_miss = pd.DataFrame((listings_df.isnull().sum())*100/len(listings_df), columns=['% Missing Values'])

listings_df_miss[listings_df_miss['% Missing Values']>0]
# create missing cols list

missing_cols = ['security_deposit','cleaning_fee','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','reviews_per_month']

# check datatype of missing cols

listings_df[missing_cols].info()
# fill '$0' to security deposit and cleaning fee

listings_df['security_deposit'].fillna('$0',inplace=True)

listings_df['cleaning_fee'].fillna('$0',inplace=True)

# remove commas

listings_df['security_deposit'] = listings_df['security_deposit'].apply(lambda x: ''.join(x.split(',')))

listings_df['cleaning_fee'] = listings_df['cleaning_fee'].apply(lambda x: ''.join(x.split(',')))

# remove $ and convert to float

listings_df['security_deposit'] = listings_df['security_deposit'].apply(lambda x: float(x.split('$')[1]))

listings_df['cleaning_fee'] = listings_df['cleaning_fee'].apply(lambda x: float(x.split('$')[1]))
# fill 0 to other missing cols

for col in missing_cols:

    listings_df[col].fillna(0,inplace=True)

    

# print result

listings_df[missing_cols].sample(10)
# convert amenities to numerical data

listings_df['amenities'] = listings_df['amenities'].apply(lambda x: x[1:-1].split(','))

listings_df['TV'] = 0

listings_df['Internet'] = 0

listings_df['Kitchen'] = 0

listings_df['Free_parking'] = 0

listings_df['Washer_dryer'] = 0

listings_df['AC'] = 0

listings_df['Smoke_detector'] = 0
for i in range(len(listings_df)):

    if 'TV' in listings_df.loc[i,'amenities']:

        listings_df.loc[i,'TV'] = 1

    if 'Internet' in listings_df.loc[i,'amenities']:

        listings_df.loc[i,'Internet'] = 1

    if 'Kitchen' in listings_df.loc[i,'amenities']:

        listings_df.loc[i,'Kitchen'] = 1 

    if '"Free Parking on Premises"' in listings_df.loc[i,'amenities']:

        listings_df.loc[i,'Free_parking'] = 1

    if 'Washer' in listings_df.loc[i,'amenities']:

        listings_df.loc[i,'Washer_dryer'] = 1

    if '"Air Conditioning"' in listings_df.loc[i,'amenities']:

        listings_df.loc[i,'AC'] = 1

    if '"Smoke Detector"' in listings_df.loc[i,'amenities']:

        listings_df.loc[i,'Smoke_detector'] = 1
# clean price, monthly price and weekly price formats

# fill missing values as 0

listings_df['price'].fillna('$0', inplace=True)

listings_df['monthly_price'].fillna('$0', inplace=True)

listings_df['weekly_price'].fillna('$0', inplace=True)

# remove commas from price

listings_df['price'] = listings_df['price'].apply(lambda x: ''.join(x.split(',')))

listings_df['monthly_price'] = listings_df['monthly_price'].apply(lambda x: ''.join(x.split(',')))

listings_df['weekly_price'] = listings_df['weekly_price'].apply(lambda x: ''.join(x.split(',')))

# convert to float

listings_df['price'] = listings_df['price'].apply(lambda x: float(x.split('$')[1]))

listings_df['monthly_price'] = listings_df['monthly_price'].apply(lambda x: float(x.split('$')[1]))

listings_df['weekly_price'] = listings_df['weekly_price'].apply(lambda x: float(x.split('$')[1]))

listings_df[['price','monthly_price','weekly_price']].sample(15)
# find correlation to price

plt.figure(figsize=(16,8))

listings_df.corr()['price'].dropna().sort_values()[:-1].plot(kind='bar', color='red')

plt.show()
listings_df.corr()['cleaning_fee'].dropna().sort_values()
# remove cols

remove_cols = ['accommodates','bedrooms','beds','square_feet','bathrooms','guests_included','weekly_price','monthly_price','cleaning_fee','security_deposit']

listings_df.drop(remove_cols,axis=1,inplace=True)
# Re-run the correlation to price with the remaining features

plt.figure(figsize=(16,8))

listings_df.corr()['price'].dropna().sort_values()[:-1].plot(kind='bar', color='red')

plt.show()
plt.figure(figsize=(16,8))

sb.scatterplot(x='number_of_reviews',y='price',data=listings_df)

plt.show()
# Run the correlation to reviews per month with the remaining features

plt.figure(figsize=(16,8))

listings_df.corr()['reviews_per_month'].dropna().sort_values()[:-1].plot(kind='bar', color='green')

plt.show()
# groupby neighbourhood

listings_df_neigh = listings_df.groupby(['neighbourhood_cleansed']).mean()

listings_df_neigh.reset_index(inplace=True)

listings_df_neigh.sort_values(['price'],inplace=True,ascending=False)
listings_df_neigh.sample(6)
plt.figure(figsize=(20,6))

sb.barplot(x='neighbourhood_cleansed',y='price',data=listings_df_neigh,color='green')

plt.xticks(rotation=90)

plt.xlabel('Neighbourhood')

plt.ylabel('Avg. Price')

plt.show()
# append average price by neighborhood to the original listings dataframe

listings_df['neigh_avg_price'] = listings_df['neighbourhood_cleansed'].apply(lambda x: float(listings_df_neigh[listings_df_neigh['neighbourhood_cleansed']== x]['price'].values))
plt.figure(figsize=(10,10))

sb.scatterplot(x='longitude',y='latitude',data=listings_df,hue='neigh_avg_price',palette='viridis')

plt.legend(loc='upper right',bbox_to_anchor=(1.5, 1.05))

plt.show()