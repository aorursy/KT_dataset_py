# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error



pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 300)
# load dataset

calendar = pd.read_csv('../input/tokyoairbnbdata/calendar.csv')

listing = pd.read_csv('../input/tokyoairbnbdata/listings.csv')

review = pd.read_csv('../input/tokyoairbnbdata/reviews.csv')
calendar.head(3)
# The number of rows

calendar.shape[0]
calendar.listing_id.nunique()
# Remove $ from price and the adjusted_price

calendar["price"] = calendar["price"].str.replace("[$, ]", "").astype(float)

calendar["adjusted_price"] = calendar["adjusted_price"].str.replace("[$, ]", "").astype(float)
listing.info()
print('The maximum price is {}.'.format(calendar['price'].max()))

print('The minimum price is {}.'.format(calendar['price'].min()))

print('The average price is {}.'.format(calendar['price'].mean()))
# The maximam seems outlier. Therefore, we'll drop outliers. 

# One definition of outlier is any data point more than 1.5 interquartile ranges (IQRs) below the first quartile or above the third quartile.

# Computing IQR

Q1 = calendar['price'].quantile(0.25)

Q3 = calendar['price'].quantile(0.75)

IQR = Q3 - Q1



# Filtering Values between Q1-1.5IQR and Q3+1.5IQR

calendar_new = calendar.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 * @IQR)')
plt.hist(calendar_new['price'], bins=20);
listing.head(3)
listing["price"] = listing["price"].str.replace("[$, ]", "").astype(float)
# The maximam seems outlier. Therefore, we'll drop outliers. 

# One definition of outlier is any data point more than 1.5 interquartile ranges (IQRs) below the first quartile or above the third quartile.

# Computing IQR

Q1_l = listing['price'].quantile(0.25)

Q3 = listing['price'].quantile(0.75)

IQR = Q3 - Q1



# Filtering Values between Q1-1.5IQR and Q3+1.5IQR

listing = listing.query('(@Q1 - 1.5 * @IQR) <= price <= (@Q3 + 1.5 * @IQR)')
plt.hist(listing['price'], bins=20);
review.head(3)
review.shape[0]
listing["price"].dtype
review['listing_id'].nunique() 

#It seems listing_id is foreign key for listing table

#It seems some accomodations did not get reviews yet
review['id'].nunique() #It seems id is a unique number in this table
# Explore the calendar.csv and find the way

print('The number of available in the calendar is {}'.format(calendar_new[calendar_new['available'] == 't']['available'].count()))

print('The number of not available in the calendar is {}'.format(calendar_new[calendar_new['available'] == 'f']['available'].count()))
# The duration of calendar data

print('The duration is between {} and {}.'.format(calendar_new['date'].min(), calendar_new['date'].max()))
# Create a column that corresponds t/f value in the available column

calendar['available_t'] = calendar['available'].apply(lambda x: 1 if x == 't' else 0)

calendar['available_f'] = calendar['available'].apply(lambda x: 1 if x == 'f' else 0)

calendar['available_total'] = calendar['available'].apply(lambda x: 1 if x == 'f' else 1)
# Change dtype of date column to datetime

calendar['date'] = pd.to_datetime(calendar['date'])
# set index for availability 

df_m = calendar[['available_t','available_f', 'available_total', 'date']].set_index(calendar['date'])

df_m = df_m.set_index([df_m.index.year, df_m.index.month, df_m.index])

df_m.index.names = ['year', 'month', 'date']

df_sum = df_m.sum(level=['year', 'month'])

df_sum['occupancy rate'] = df_m['available_f'].mean(level=['year', 'month'])

df_sum = df_sum.reset_index()
# Create a date column that has the year and the date, e.g.'20XX-XX'

df_sum['year'] = df_sum['year'].astype('str')

df_sum['month'] = df_sum['month'].astype('str')

df_sum['date'] = df_sum['year'].str.cat(df_sum['month'], sep='-')
# set index for price

df_p = calendar[['price','adjusted_price', 'date', 'minimum_nights']].set_index(calendar['date'])

df_p = df_p.set_index([df_p.index.year, df_p.index.month, df_p.index])

df_p.index.names = ['year', 'month', 'date']

df_ave_p = df_p.mean(level=['year', 'month'])

df_ave_p = df_ave_p.reset_index()
df_ave_p
# Create a date column that has the year and the date, e.g.'20XX-XX'

df_ave_p['year'] = df_ave_p['year'].astype('str')

df_ave_p['month'] = df_ave_p['month'].astype('str')

df_ave_p['date'] = df_ave_p['year'].str.cat(df_ave_p['month'], sep='-')
# Create a figure, omitting Sep 2019 and 2020 since some dates are missing

fig, ax1 = plt.subplots()

plt.xticks(rotation=-70)

ax2 = ax1.twinx()

ax1.bar(df_sum['date'].iloc[1:-1], df_sum['available_total'].iloc[1:-1], color="#9DE0AD")

ax2.plot(df_sum['date'].iloc[1:-1], df_sum['occupancy rate'].iloc[1:-1], color="#FF8C94")

ax1.set_xlabel('year-month', fontsize = 12)

ax1.set_ylabel('The number of accomodations in Tokyo', fontsize = 12)

ax2.set_ylabel('Occupancy rate', fontsize = 12);
plt.plot(df_ave_p['date'].iloc[1:-1], df_ave_p['price'].iloc[1:-1])

plt.xticks(rotation=-70)

plt.xlabel('year-month', fontsize=12)

plt.ylabel('average price of accomodations', fontsize=12);
# numbers of beds -> listing.beds, property type -> listing.property_type,

# room type -> listing.room_type, accommodates -> listing.accommodates,

# Bed type -> listing.bed_type, 

# https://qiita.com/go8/items/90167693f142ebb55a7d (Matplotlib Graph)
# calculate occupancy rate calendar by listing id

df_groupby_ava = calendar.groupby('listing_id')[['available_f', 'available_total']].sum()

df_groupby_ava['occupancy_rate'] = df_groupby_ava.available_f/df_groupby_ava.available_total

df_groupby_ava = df_groupby_ava.reset_index()

# merge this  to listing dataframe

listing_new = pd.merge(listing, df_groupby_ava, how='left', left_on='id', right_on='listing_id')
# Histgram of number of beds 

print('The maximum number of beds is {}'.format(listing.beds.max()))

print('The minimum number of beds is {}'.format(listing.beds.min()))

print('The maximum number of accomodates is {}'.format(listing.accommodates.max()))

print('The minimum number of accomodates is {}'.format(listing.accommodates.min()))

print(listing.beds.value_counts())
plt.figure()

plt.hist(listing.beds, bins=50)

plt.xlabel('the number of beds')

plt.title('Histgarm of beds');
plt.figure()

plt.hist(listing.accommodates, bins=20)

plt.xlabel('the number of accomodates')

plt.title('Histgarm of accomodates');
# property type -> listing.property_type

plt.figure()

plt.xticks(rotation=90)

sns.barplot(x=listing_new.property_type.value_counts().index, y=listing_new.property_type.value_counts()/listing.shape[0], palette = 'viridis')

plt.title("Property Type")



# Create a dataframe and sort by occupacy_rate

property_type_oc = pd.DataFrame(listing_new.groupby('property_type').occupancy_rate.mean())

property_type_oc = property_type_oc.sort_values(by=['occupancy_rate'], ascending=False)

plt.figure()

plt.xticks(rotation=90)

plt.bar(property_type_oc.index, property_type_oc.occupancy_rate, color='#FF8C94')

plt.title("Property Type - Occupancy rate");
# room type -> listing.room_type

plt.figure()

sns.barplot(listing_new.room_type.value_counts().index, listing_new.room_type.value_counts()/listing.shape[0], palette = 'viridis')

plt.title("Room Type")



# Create a dataframe and sort by occupacy_rate

room_type_oc = pd.DataFrame(listing_new.groupby('room_type').occupancy_rate.mean())

room_type_oc = room_type_oc.sort_values(by=['occupancy_rate'], ascending=False)

plt.figure()

plt.bar(room_type_oc.index, room_type_oc.occupancy_rate, color='#FF8C94')

plt.title("Room Type - Occupancy rate");
# bed type -> listing.bed_type



plt.figure()

sns.barplot(listing_new.bed_type.value_counts().index, listing_new.bed_type.value_counts()/listing.shape[0], palette = 'viridis')

plt.title("Bed Type")



# Create a dataframe and sort by occupacy_rate

bed_type_oc = pd.DataFrame(listing_new.groupby('bed_type').occupancy_rate.mean())

bed_type_oc = bed_type_oc.sort_values(by=['occupancy_rate'], ascending=False)

plt.figure()

plt.bar(bed_type_oc.index, bed_type_oc.occupancy_rate, color='#FF8C94')

plt.title("Bed Type - Occupancy rate");
listing.bed_type.value_counts()
# location -> listing.neighbourhood_cleansed



plt.figure(figsize=(14, 4))

plt.xticks(rotation=90)

sns.barplot(listing_new.neighbourhood_cleansed.value_counts().index,listing_new.neighbourhood_cleansed.value_counts()/listing.shape[0], palette = 'viridis')

plt.title("Location")



# Create a dataframe and sort by occupacy_rate

neighbourhood_cleansed_oc = pd.DataFrame(listing_new.groupby('neighbourhood_cleansed').occupancy_rate.mean())

neighbourhood_cleansed_oc = neighbourhood_cleansed_oc.sort_values(by=['occupancy_rate'], ascending=False)

plt.figure(figsize=(14, 4))

plt.xticks(rotation=90)

plt.bar(neighbourhood_cleansed_oc.index, neighbourhood_cleansed_oc.occupancy_rate, color='#FF8C94')

plt.title("Location - Occupancy rate");
listing['price_pp'] = listing.price/listing.accommodates

print('The average price per person is {} yen.'.format(round(listing['price_pp'].mean())))
listing = listing.rename(index=str, columns={"id": "listing_id"})
# We'll merge listing dataframe and 'month' in calendar dataframe

# Create a column that holds month data in calendar dataframe

calendar['month'] = calendar.date.dt.month

calendar_new = calendar.drop(['date','available','adjusted_price','price'], axis=1)

listing_new = listing.drop(['minimum_nights','maximum_nights'], axis=1)



# Merge calendar and listing data onmerge

df_new = pd.merge(calendar_new, listing_new, on = 'listing_id')
# Check missing values

listings_missing_df = df_new.isnull().mean()*100



#filter out only columns, which have missing values

listings_columns_with_nan = listings_missing_df[listings_missing_df > 0]



#plot the results

listings_columns_with_nan.plot.bar(title='Missing values per column, %');
'''Drop columns that have high null ratio:

    thumbnail_url, medium_url, xl_picture_url, host_acceptance_rate, neighbourhood_group_cleansed, 

   square_feet, weekly_price, monthly_price, jurisdiction_names



   Drop other unneeded columns since they are not relevant to the anaysis:

   from calendar dataframe: date, available, adjusted_price, minimum_nights, maximum_nights

   from listing dataframe: host_since, id, listing_url, scrape_id, last_scraped, name, summary, space, 

   description, experiences_offered, neighborhood_overview, notes, 

   transit, access, interaction, house_rules, thumbnail_url, medium_url, 

   picture_url, xl_picture_url, host_id, host_url, host_name, host_location, 

   host_about, host_thumbnail_url, host_picture_url, host_neighbourhood, host_verifications, 

   host_has_profile_pic, host_identity_verified, smart_location, state, city, zipcode, market, country_code, country, 

   latitude, longitude, is_location_exact, amenities, calendar_updated, calendar_last_scraped, 

   first_review, last_review, license, cancellation_policy, require_guest_profile_picture, 

   require_guest_phone_verification, price_pp,calculated_host_listings_count,

   minimum_minimum_nights, maximum_minimum_nights, minimum_maximum_nights, maximum_maximum_nights, minimum_nights_avg_ntm, maximum_nights_avg_ntm,

   calculated_host_listings_count_entire_homes, calculated_host_listings_count_private_rooms, calculated_host_listings_count_shared_rooms'''



columns_to_drop = ['host_since','thumbnail_url','medium_url','xl_picture_url','host_acceptance_rate','neighbourhood_group_cleansed',

'square_feet','weekly_price','monthly_price','jurisdiction_names','listing_id','listing_url','scrape_id','last_scraped','name','summary','space',

'description','experiences_offered','neighborhood_overview','notes','transit','access','interaction','house_rules','thumbnail_url','medium_url',

'picture_url','xl_picture_url','host_id','host_url','host_name','host_location','host_about','host_thumbnail_url',

'host_picture_url','neighbourhood','street','host_neighbourhood','host_verifications','smart_location','state','city',

'zipcode','market','country_code','country','latitude','longitude','is_location_exact','amenities','calendar_updated',

'minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm',

'calendar_last_scraped','first_review','last_review','license','cancellation_policy','require_guest_profile_picture',

'require_guest_phone_verification','price_pp','calculated_host_listings_count','calculated_host_listings_count_entire_homes',

'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms']
df_new_clean = df_new.drop(columns_to_drop, axis=1)
# 1 dropping all the rows that have missing data

df_new_clean.dropna(inplace=True)
# Categorical variables -> host_response_time, host_is_superhost, host_has_profile_pic, host_identity_verified, neighbourhood_cleansed, 

# property_type, room_type, bed_type, has_availability, instant_bookable, is_business_travel_ready, 



# Numerical variables -> host_response_rate, host_listings_count, host_total_listings_count, accommodates, bathrooms, 

# bedrooms, beds, security_deposit, cleaning_fee, guests_included, minimum_nights, maximum_nights, availability_30, 

# availability_60, availability_90, availability_365, number_of_reviews, number_of_reviews_ltm, review_scores_rating, 

# review_scores_accuracy, review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location, 

# review_scores_value, reviews_per_month



# Change 'object' datatype to numeric -> host_response_rate, security_deposit, cleaning_fee, extra_people

df_new_clean["host_response_rate"] = df_new_clean["host_response_rate"].str.replace("%", "")

df_new_clean["security_deposit"] = df_new_clean["security_deposit"].str.replace("[$, ]", "")

df_new_clean["cleaning_fee"] = df_new_clean["cleaning_fee"].str.replace("[$, ]", "")

df_new_clean["extra_people"] = df_new_clean["extra_people"].str.replace("[$, ]", "")

df_new_clean = df_new_clean.astype({'host_response_rate': float, 'security_deposit': float, 'cleaning_fee': float, 'extra_people': float})
# numerical variables to find out the correlations

cols = ['price', 'host_response_rate','host_listings_count','host_total_listings_count','accommodates','bathrooms',

        'bedrooms', 'beds', 'security_deposit','cleaning_fee','guests_included','minimum_nights',

        'maximum_nights','availability_30', 'availability_60','availability_90','availability_365',

        'number_of_reviews','number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',

        'review_scores_cleanliness','review_scores_checkin', 'review_scores_communication', 'review_scores_location',

        'review_scores_value', 'reviews_per_month', 'month']

plt.figure(figsize=(20,20))

sns.heatmap(df_new_clean[cols].corr(), annot=True, fmt='.2f');
#Pull a list of the column names of the categorical variables

cat_df = df_new_clean.select_dtypes(include=['object'])

cat_cols_lst = cat_df.columns
# Convert Categorical variables

def create_dummy_df(df, cat_cols, dummy_na):

    for col in  cat_cols:

        try:

            # for each cat add dummy var, drop original column

            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)

        except:

            continue

    return df



df_new_clean_cat = create_dummy_df(df_new_clean, cat_cols_lst, dummy_na=False)
# Train ML model by only numerical columns, by LinearRegression

X = df_new_clean_cat.drop('price', axis=1)

y = df_new_clean[['price']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=30)
# Train ML model by only numerical columns, with RandomForest

forest = RandomForestRegressor(n_estimators=4, 

                               criterion='mse', 

                               random_state=57, 

                               n_jobs=-1)

forest = forest.fit(X_train, y_train.squeeze())



#calculate scores for the model

y_train_pred = forest.predict(X_train)

y_test_pred = forest.predict(X_test)



print('MSE train: %.3f, test: %.3f' % (

        mean_squared_error(y_train, y_train_pred),

        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (

        r2_score(y_train, y_train_pred),

        r2_score(y_test, y_test_pred)))
#get feature importances from the model

headers = ["name", "score"]

values = sorted(zip(X_train.columns, forest.feature_importances_), key=lambda x: x[1] * -1)

forest_feature_importances = pd.DataFrame(values, columns = headers)

forest_feature_importances = forest_feature_importances.sort_values(by = ['score'], ascending = False)



features = forest_feature_importances['name'][:15]

y_pos = np.arange(len(features))

scores = forest_feature_importances['score'][:15]



#plot feature importances

plt.figure(figsize=(10,5))

plt.bar(y_pos, scores, align='center')

plt.xticks(y_pos, features, rotation='vertical')

plt.ylabel('Score')

plt.xlabel('Features')

plt.title('Feature importances')

 

plt.show()
# create a dataframe that has actual data and predicted data

DFRFtest = pd.DataFrame({'Actual':y_test['price'], 'Prediction': y_test_pred})



# check the first 5 data

DFRFtest.head(5)