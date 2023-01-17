import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

l = pd.read_csv('../input/seattle/listings.csv')



l.columns
l.shape
l.dtypes.value_counts()
l.head(5)
cols = ['number_of_reviews', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy',

       'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 

       'review_scores_location', 'review_scores_value', 'reviews_per_month']

l[cols].describe()
cols = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 

        'review_scores_communication', 'review_scores_location', 'review_scores_value']

df = l[cols]



import seaborn as sns

corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
l['review_scores_rating'].hist()
cols = ['number_of_reviews', 'first_review', 'last_review', 'reviews_per_month']

l[cols].head()
l[['number_of_reviews', 'reviews_per_month']].hist()
df = l[['review_scores_rating', 'reviews_per_month']].copy()



df['cx_score'] = df['review_scores_rating'] / 100 * df['reviews_per_month']



import matplotlib.pyplot as plt

%matplotlib inline



fig = plt.figure(figsize = (18,5))

ax = fig.gca()

df.hist(layout=(1,3), ax=ax)
corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
cols = ['property_type', 'room_type', 'accommodates', 'bathrooms', 'beds', 'bed_type', 'amenities', 'square_feet']

l[cols].head()
l['property_type'].value_counts()
l['room_type'].value_counts()
l['bed_type'].value_counts()
cols = ['accommodates', 'bathrooms', 'beds', 'square_feet']

l[cols].describe()
fig = plt.figure(figsize = (18,5))

ax = fig.gca()

l[cols].hist(layout=(1,4), ax=ax)
l['amenities'].value_counts()
cols = ['property_type', 'room_type', 'accommodates', 'bathrooms', 'beds', 'bed_type', 'amenities', 'square_feet']

l[cols].isnull().sum()
cols = ['street', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state',

       'zipcode', 'market', 'smart_location', 'country_code', 'country', 'latitude', 'longitude',

       'is_location_exact', 'jurisdiction_names']

l[cols].head()
l['neighbourhood'].value_counts()
l['neighbourhood_cleansed'].value_counts()
l['neighbourhood_group_cleansed'].value_counts()
cols = ['city', 'state', 'market', 'country_code', 'country', 'jurisdiction_names']

l[cols].head()
l['zipcode'].value_counts()
fig = plt.figure(figsize = (15,10))

ax = fig.gca()

l.plot.scatter('longitude', 'latitude', ax=ax)
l['is_location_exact'].value_counts()
cols = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'guests_included',

       'extra_people', 'minimum_nights', 'maximum_nights', 'requires_license', 'license',

       'instant_bookable', 'require_guest_profile_picture', 'require_guest_phone_verification',

       'cancellation_policy']

l[cols].head()
l[cols].isnull().sum() / l.shape[0]
cols = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'guests_included',

       'extra_people', 'minimum_nights', 'maximum_nights']

df = l[cols].copy()



# Convert the money columns into floats

dollar_to_float = lambda x: x.replace('[\$,]', '', regex=True).astype(float)

money_cols = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']



# Apply the function to the money cols

df[money_cols] = df[money_cols].apply(dollar_to_float, axis=1)



df.describe()
fig = plt.figure(figsize = (18,7))

ax = fig.gca()

df.hist(ax=ax)
l['cancellation_policy'].value_counts()
cols = ['name', 'summary', 'space', 'description', 'experiences_offered', 'neighborhood_overview',

       'notes', 'transit']

l[cols].head()
l['experiences_offered'].value_counts()
l[cols].isnull().sum() / l.shape[0]
cols = ['host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate',

       'host_acceptance_rate', 'host_is_superhost', 'host_neighbourhood', 'host_listings_count',

       'host_total_listings_count', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified',

       'calculated_host_listings_count']

l[cols].head()
df = l[['host_listings_count', 'host_total_listings_count', 'calculated_host_listings_count']].copy()

df.corr()
fig = plt.figure(figsize = (18,7))

ax = fig.gca()

l.plot.scatter('host_listings_count', 'calculated_host_listings_count', ax=ax)
cols = ['host_response_rate', 'host_acceptance_rate']

df = l[cols].copy()



# Convert the percentage columns into floats

pct_to_float = lambda x: x.str.replace(r'%', r'.0').astype('float') / 100.0



# Apply the function to the rate cols

df = df.apply(pct_to_float, axis=1)



fig = plt.figure(figsize = (12,7))

ax = fig.gca()

df.boxplot(ax=ax)
df.describe()
l['host_acceptance_rate'].value_counts()
l['host_is_superhost'].value_counts()
l['host_has_profile_pic'].value_counts()
l['host_identity_verified'].value_counts()
l['host_response_time'].value_counts()
pd.to_datetime(l['host_since']).hist()
pd.to_datetime(l['host_since']).describe()
l['host_neighbourhood'].value_counts()
len(set(l['host_neighbourhood'].unique()).intersection(set(l['neighbourhood'].unique())))
len(set(l['host_neighbourhood'].unique()).intersection(set(l['neighbourhood_cleansed'].unique())))
l['host_verifications'].unique()
cols = ['calendar_updated', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365']

l[cols].head()
cols = ['availability_30', 'availability_60', 'availability_90', 'availability_365']

l[cols].describe()
l['has_availability'].value_counts()
l['calendar_updated'].unique()
l['calendar_updated'].isnull().sum()
cols = ['id', 'listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',

        'host_id', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped']

l[cols].head()
pd.to_datetime(l['last_scraped']).value_counts()
pd.to_datetime(l['calendar_last_scraped']).value_counts()
l[cols].isnull().sum() / l.shape[0]
df = l.copy()



df['cx_score'] = df['review_scores_rating'] / 100 * df['reviews_per_month']
l['neighbourhood_cleansed'].value_counts()[:10]
fontsize = 10



df_temp = l[['neighbourhood_group_cleansed', 'review_scores_rating']]

df_temp['n'] = np.where(

    df_temp['neighbourhood_group_cleansed'] == 'University District', 

    'University District', 'Other neighbourhoods'

)



fig, axes = plt.subplots(figsize=(14, 8))

sns.violinplot('n','review_scores_rating', data=df_temp, ax = axes)

axes.set_title('Review scores by neighbourhood group')



axes.yaxis.grid(True)

axes.set_xlabel('Neighbourhood group')

axes.set_ylabel('Review score')



plt.show()
fontsize = 10



df_temp = l[['neighbourhood_group_cleansed', 'reviews_per_month']]

df_temp['n'] = np.where(

    df_temp['neighbourhood_group_cleansed'] == 'University District', 

    'University District', 'Other neighbourhoods'

)



fig, axes = plt.subplots(figsize=(14, 8))

sns.violinplot('n','reviews_per_month', data=df_temp, ax = axes)

axes.set_title('Reviews per month by neighbourhood group')



axes.yaxis.grid(True)

axes.set_xlabel('Neighbourhood group')

axes.set_ylabel('Reviews per month')



plt.show()
fontsize = 10



df_temp = l[['neighbourhood_group_cleansed', 'review_scores_rating', 'reviews_per_month']].copy()

df_temp['cx_score'] = df_temp['review_scores_rating'] / 100 * df_temp['reviews_per_month']

df_temp['n'] = np.where(

    df_temp['neighbourhood_group_cleansed'] == 'University District', 

    'University District', 'Other neighbourhoods'

)



fig, axes = plt.subplots(figsize=(14, 8))

sns.violinplot('n','cx_score', data=df_temp, ax = axes)

axes.set_title('CX score by neighbourhood group')



axes.yaxis.grid(True)

axes.set_xlabel('Neighbourhood group')

axes.set_ylabel('CX score')



plt.show()
df = l.copy()

df = df.loc[df['neighbourhood_group_cleansed'] == 'University District']

df['cx_score'] = df['review_scores_rating'] / 100 * df['reviews_per_month']
df = df.drop(['number_of_reviews', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy',

             'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',

             'review_scores_location', 'review_scores_value', 'reviews_per_month'], axis=1)
df = df.dropna(subset=['cx_score'])
def convert_property_type(df):

    """

    Applies transformations to the property_type feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

            new_data.columns - the column names of the dummy variables

    """

    

    # Map the property type to the top 3 values and an Other bucket

    df['property_type'] = df['property_type'].map(

        {'House': 'House', 

         'Apartment': 'Apartment', 

         'Townhouse': 'Townhouse'}

    ).fillna('Other')

    

    # Create the dummy columns and append them to our dataframe

    new_data = pd.get_dummies(df[['property_type']])

    df[new_data.columns] = new_data

    

    # Remove the original categorical column

    df = df.drop(['property_type'], axis=1)

    

    return df, new_data.columns
def convert_room_type(df):

    """

    Applies transformations to the room_type feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

            new_data.columns - the column names of the dummy variables

    """

    # Create the dummy columns and append them to our dataframe

    new_data = pd.get_dummies(df[['room_type']])

    df[new_data.columns] = new_data

    

    # Remove the original categorical column

    df = df.drop(['room_type'], axis=1)

    

    return df, new_data.columns
def convert_bed_type(df):

    """

    Applies transformations to the bed_type feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # We just care about whether the bed is REAL

    df['real_bed'] = df['bed_type'].map({'Real bed': 1}).fillna(0)

    

    # Remove the original categorical column

    df = df.drop(['bed_type'], axis=1)

    

    return df
def convert_amenities(df):

    """

    Applies transformations to the amenities feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Look for presence of the string within the amenities column

    df['amenities_tv'] = df['amenities'].str.contains('TV')

    df['amenities_internet'] = df['amenities'].str.contains('Internet')

    df['amenities_wireless_internet'] = df['amenities'].str.contains('Wireless Internet')

    df['amenities_cable_tv'] = df['amenities'].str.contains('Cable TV')

    df['amenities_kitchen'] = df['amenities'].str.contains('Kitchen')

    df['amenities_elevator_in_building'] = df['amenities'].str.contains('Elevator in Building')

    df['amenities_wheelchair_accessible'] = df['amenities'].str.contains('Wheelchair Accessible')

    df['amenities_smoke_detector'] = df['amenities'].str.contains('Smoke Detector')

    df['amenities_pool'] = df['amenities'].str.contains('Pool')

    df['amenities_free_parking_on_premises'] = df['amenities'].str.contains('Free Parking on Premises')

    df['amenities_air_conditioning'] = df['amenities'].str.contains('Air Conditioning')

    df['amenities_heating'] = df['amenities'].str.contains('Heating')

    df['amenities_pets_live_on_this_property'] = df['amenities'].str.contains('Pets live on this property')

    df['amenities_washer'] = df['amenities'].str.contains('Washer')

    df['amenities_breakfast'] = df['amenities'].str.contains('Breakfast')

    df['amenities_buzzer_wireless_intercom'] = df['amenities'].str.contains('Buzzer/Wireless Intercom')

    df['amenities_pets_allowed'] = df['amenities'].str.contains('Pets Allowed')

    df['amenities_carbon_monoxide_detector'] = df['amenities'].str.contains('Carbon Monoxide Detector')

    df['amenities_gym'] = df['amenities'].str.contains('Gym')

    df['amenities_dryer'] = df['amenities'].str.contains('Dryer')

    df['amenities_indoor_fireplace'] = df['amenities'].str.contains('Indoor Fireplace')

    df['amenities_family_kid_friendly'] = df['amenities'].str.contains('Family/Kid Friendly')

    df['amenities_dogs'] = df['amenities'].str.contains('Dog(s)')

    df['amenities_essentials'] = df['amenities'].str.contains('Essentials')

    df['amenities_cats'] = df['amenities'].str.contains('Cat(s)')

    df['amenities_hot_tub'] = df['amenities'].str.contains('Hot Tub')

    df['amenities_shampoo'] = df['amenities'].str.contains('Shampoo')

    df['amenities_first_aid_kit'] = df['amenities'].str.contains('First Aid Kit')

    df['amenities_smoking_allowed'] = df['amenities'].str.contains('Smoking Allowed')

    df['amenities_fire_extinguisher'] = df['amenities'].str.contains('Fire Extinguisher')

    df['amenities_doorman'] = df['amenities'].str.contains('Doorman')

    df['amenities_washer_dryer'] = df['amenities'].str.contains('Washer / Dryer')

    df['amenities_safety_card'] = df['amenities'].str.contains('Safety Card')

    df['amenities_suitable_for_events'] = df['amenities'].str.contains('Suitable for Events')

    df['amenities_other_pets'] = df['amenities'].str.contains('Other pet(s)')

    df['amenities_24_hour_check_in'] = df['amenities'].str.contains('24-Hour Check-in')

    df['amenities_hangers'] = df['amenities'].str.contains('Hangers')

    df['amenities_laptop_friendly_workspace'] = df['amenities'].str.contains('Laptop Friendly Workspace')

    df['amenities_iron'] = df['amenities'].str.contains('Iron')

    df['amenities_hair_dryer'] = df['amenities'].str.contains('Hair Dryer')

    df['amenities_lock_on_bedroom_door'] = df['amenities'].str.contains('Lock on Bedroom Door')

    

    # Remove the original categorical column

    df = df.drop(['amenities'], axis=1)

    

    return df
df = df.drop(['street', 'city', 'state', 'zipcode', 'market', 'smart_location', 'country_code',

             'country', 'latitude', 'longitude', 'jurisdiction_names'], axis=1)
def convert_neighbourhood_cleansed(df):

    """

    Applies transformations to the neighbourhood_cleansed feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

            new_data.columns - the column names of the dummy variables

    """

    

    # Create dummies on the column

    new_data = pd.get_dummies(df[['neighbourhood_cleansed']])

    df[new_data.columns] = new_data

    

    # We will keep the neighbourhood_cleansed column for future use

    return df, new_data.columns
def convert_neighbourhood_group_cleansed(df):

    """

    Applies transformations to the neighbourhood_group_cleansed feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

            new_data.columns - the column names of the dummy variables

    """

    

    # Create dummies on the column

    new_data = pd.get_dummies(df[['neighbourhood_group_cleansed']])

    df[new_data.columns] = new_data

    

    # We will keep the neighbourhood_cleansed column for future use

    return df, new_data.columns
def convert_is_location_exact(df):

    """

    Input: df - a dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    df['is_location_exact'] = df['is_location_exact'].map({'t':1}).fillna(0)

    

    return df
def convert_price(df):

    """

    Applies transformations to the price feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Convert the money variable into a numeric variable

    df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

    

    return df
def convert_weekly_price(df):

    """

    Applies transformations to the weekly_price feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Convert the money variable into a numeric variable

    df['weekly_price'] = df['weekly_price'].replace('[\$,]', '', regex=True).astype(float)

    

    # Note that this code is assuming that price has already been converted

    # so we will run this code after convert_price().

    df['weekly_price_ratio'] = df['weekly_price'] / df['price']

        

    # Boolean feature to indicate that a weekly price has been set

    df['has_weekly_price'] = ~df['weekly_price'].isnull()

    

    # If there is no weekly price then set the ratio to 7, since

    # this would imply the regular price

    df['weekly_price_ratio'] = df['weekly_price_ratio'].fillna(7)

    df['weekly_price'] = df['weekly_price'].fillna(7*df['price'])

    

    return df
def convert_monthly_price(df):

    """

    Applies transformations to the weekly_price feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Convert the money variable into a numeric variable

    df['monthly_price'] = df['monthly_price'].replace('[\$,]', '', regex=True).astype(float)

    

    # Note that this code is assuming that price has already been converted

    # so we will run this code after convert_price().

    df['monthly_price_ratio'] = df['monthly_price'] / df['price']

        

    # Boolean feature to indicate that a weekly price has been set

    df['has_monthly_price'] = ~df['monthly_price'].isnull()

    

    # If there is no monthly price then set the ratio to 365/12, since 

    # this would imply the regular price.

    df['monthly_price_ratio'] = df['monthly_price_ratio'].fillna(365./12.)

    df['monthly_price'] = df['monthly_price'].fillna(365./12.*df['price'])

    

    return df
def convert_security_deposit(df):

    """

    Applies transformations to the security_deposit feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Convert the money variable into a numeric variable

    df['security_deposit'] = df['security_deposit'].replace('[\$,]', '', regex=True).astype(float)

    

    # Note that this code is assuming that price has already been converted

    # so we will run this code after convert_price().

    df['security_deposit_ratio'] = df['security_deposit'] / df['price']

        

    # Boolean feature to indicate that a weekly price has been set

    df['has_security_deposit'] = ~df['security_deposit'].isnull()

    

    # If there is no security_deposit then set the ratio to zero

    # This assumes that there is no security deposit

    df['security_deposit_ratio'] = df['security_deposit_ratio'].fillna(0)

    df['security_deposit'] = df['security_deposit'].fillna(0)

    

    return df
def convert_cleaning_fee(df):

    """

    Applies transformations to the cleaning_fee feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Convert the money variable into a numeric variable

    df['cleaning_fee'] = df['cleaning_fee'].replace('[\$,]', '', regex=True).astype(float)

    

    # Note that this code is assuming that price has already been converted

    # so we will run this code after convert_price().

    df['cleaning_fee_ratio'] = df['cleaning_fee'] / df['price']

    

    # Convert the money variable into a numeric variable

    df['cleaning_fee'] = df['cleaning_fee'].replace('[\$,]', '', regex=True).astype(float)

    

    # If there is no cleaning_fee then set the ratio to zero

    # This assumes that there is no cleaning fee

    df['cleaning_fee_ratio'] = df['cleaning_fee_ratio'].fillna(0)

    df['cleaning_fee'] = df['cleaning_fee'].fillna(0)

    

    return df
def convert_extra_people(df):

    """

    Applies transformations to the extra_people feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Convert the money variable into a numeric variable

    df['extra_people'] = df['extra_people'].replace('[\$,]', '', regex=True).astype(float)

    

    # Note that this code is assuming that price has already been converted

    # so we will run this code after convert_price().

    df['extra_people_ratio'] = df['extra_people'] / df['price']

    

    # Convert the money variable into a numeric variable

    df['extra_people'] = df['extra_people'].replace('[\$,]', '', regex=True).astype(float)

    

    # If there is no extra_people then set the ratio to zero

    # This assumes that there is no extra people fee

    df['extra_people_ratio'] = df['extra_people_ratio'].fillna(0)

    df['extra_people'] = df['extra_people'].fillna(0)

    

    return df
df = df.drop(['requires_license', 'license'], axis=1)
def convert_instant_bookable(df):

    """

    Applies transformations to the instant_bookable feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    df['instant_bookable'] = df['instant_bookable'].map({'t': 1}).fillna(0)

    

    return df
def convert_require_guest_profile_picture(df):

    """

    Applies transformations to the require_guest_profile_picture feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    df['require_guest_profile_picture'] = df['require_guest_profile_picture'].map({'t': 1}).fillna(0)

    

    return df
def convert_require_guest_phone_verification(df):

    """

    Applies transformations to the require_guest_phone_verification feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    df['require_guest_phone_verification'] = df['require_guest_phone_verification'].map({'t': 1}).fillna(0)

    

    return df
def convert_cancellation_policy(df):

    """

    Applies transformations to the cancellation_policy feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

            new_data.columns - the column names of the dummy variables

    """

    

    # Create dummies on the column

    new_data = pd.get_dummies(df[['cancellation_policy']])

    df[new_data.columns] = new_data

    

    # We will keep the cancellation_policy column for future use

    df = df.drop(['cancellation_policy'], axis=1)

    

    return df, new_data.columns
df = df.drop(['name', 'summary', 'space', 'description', 'experiences_offered', 'neighborhood_overview',

             'notes', 'transit'], axis=1)
df = df.drop(['host_name', 'host_acceptance_rate', 'host_total_listings_count'], axis=1)
def convert_host_about(df):

    """

    Applies transformations to the host_about feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Ascertain whether the value is null

    df['has_host_about'] = ~df['host_about'].isnull()

    

    # Drop the original column

    df = df.drop(['host_about'], axis=1)

    

    return df
def convert_host_since(df):

    """

    Applies transformations to the host_since feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Subtract the dates to get the number of days

    df['days_as_host'] = (pd.to_datetime(df['last_scraped']) - pd.to_datetime(df['host_since'])) / np.timedelta64(1, 'D')

    

    # Drop the original column

    df = df.drop(['host_since', 'last_scraped'], axis=1)

    

    return df
def convert_host_location(df):

    """

    Applies transformations to the host_location feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """



    # Search for 'Seattle' in the host_location field

    df['host_in_seattle'] = df['host_location'].str.contains('Seattle')

    

    # Drop the original column

    df = df.drop(['host_location'], axis=1)

    

    return df
def convert_host_response_time(df):

    """

    Applies transformations to the host_response_time feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """



    # Map the values

    df['host_response_time'] = df['host_response_time'].map(

        {'within an hour': 1, 'within a few hours': 2, 'within a day': 3}

    ).fillna(4)

    

    return df
def convert_host_response_rate(df):

    """

    Applies transformations to the host_response_rate feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Convert to float

    df['host_response_rate'] = df['host_response_rate'].str.replace(r'%', r'.0').astype('float') / 100.0

    

    # Fill missing values with zero

    df['host_response_rate'] = df['host_response_rate'].fillna(0)

    

    return df
def convert_host_neighbourhood(df):

    """

    Applies transformations to the host_neighbourhood feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Lookup against all 3 neighbourhood columns

    df['host_in_neighbourhood'] = np.where(

        df['host_neighbourhood'] == df['neighbourhood'], True, 

        np.where(

            df['host_neighbourhood'] == df['neighbourhood_cleansed'], True,

            np.where(

                df['host_neighbourhood'] == df['neighbourhood_group_cleansed'], True, False

            )

        )

    )

    

    # Remove the original columns

    df = df.drop(['host_neighbourhood', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed'], axis=1)

    

    return df
def convert_host_verifications(df):

    """

    Applies transformations to the host_verifications feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    # Lookup the substring and set boolean value as column

    df['host_verif_email'] = df['host_verifications'].str.contains('email')

    df['host_verif_kba'] = df['host_verifications'].str.contains('kba')

    df['host_verif_phone'] = df['host_verifications'].str.contains('phone')

    df['host_verif_reviews'] = df['host_verifications'].str.contains('reviews')

    df['host_verif_jumio'] = df['host_verifications'].str.contains('jumio')

    df['host_verif_facebook'] = df['host_verifications'].str.contains('facebook')

    df['host_verif_linkedin'] = df['host_verifications'].str.contains('linkedin')

    df['host_verif_google'] = df['host_verifications'].str.contains('google')

    df['host_verif_manual_online'] = df['host_verifications'].str.contains('manual_online')

    df['host_verif_manual_offline'] = df['host_verifications'].str.contains('manual_offline')

    df['host_verif_sent_id'] = df['host_verifications'].str.contains('sent_id')

    df['host_verif_amex'] = df['host_verifications'].str.contains('amex')

    df['host_verif_weibo'] = df['host_verifications'].str.contains('weibo')

    df['host_verif_photographer'] = df['host_verifications'].str.contains('photographer')

    

    # Drop the original column

    df = df.drop(['host_verifications'], axis=1)

    

    return df
def convert_host_is_superhost(df):

    """

    Applies transformations to the host_is_superhost feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1}).fillna(0)

    

    return df
def convert_host_has_profile_pic(df):

    """

    Applies transformations to the host_has_profile_pic feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'t': 1}).fillna(0)

    

    return df
def convert_host_identity_verified(df):

    """

    Applies transformations to the host_is_superhost feature of the dataset.

    

    Input: df - the AirBnb Seattle dataset containing the unprocessed column's data

    Output: df - the modified dataset containing the transformed features

    """

    

    df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1}).fillna(0)

    

    return df
df = df.drop(['calendar_updated', 'has_availability', 'availability_30', 'availability_60',

             'availability_90', 'availability_365'], axis=1)
df = df.set_index('id')
df = df.drop(['listing_url', 'scrape_id', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 

              'host_id', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped'], axis=1)
df.columns
df.shape
df.columns[df.isnull().sum() / df.shape[0] == 1]
df = df.drop(['square_feet'], axis=1)
# dummies

df, cols_neighbourhood_cleansed = convert_neighbourhood_cleansed(df)

df, cols_neighbourhood_group_cleansed = convert_neighbourhood_group_cleansed(df)

df, cols_property_type = convert_property_type(df)

df, cols_cancellation_policy = convert_cancellation_policy(df)

df, cols_room_type = convert_room_type(df)



cols_dummies = []

cols_dummies.extend(cols_neighbourhood_cleansed)

cols_dummies.extend(cols_neighbourhood_group_cleansed)

cols_dummies.extend(cols_property_type)

cols_dummies.extend(cols_cancellation_policy)

cols_dummies.extend(cols_room_type)
cols_dummies
# map columns to new variables

df = convert_host_since(df)

df = convert_host_location(df)

df = convert_host_about(df)

df = convert_host_response_time(df)

df = convert_host_response_rate(df)

df = convert_host_neighbourhood(df)

df = convert_host_verifications(df)

df = convert_host_has_profile_pic(df)

df = convert_host_identity_verified(df)

df = convert_bed_type(df)

df = convert_amenities(df)

df = convert_price(df)

df = convert_weekly_price(df)

df = convert_monthly_price(df)

df = convert_security_deposit(df)

df = convert_cleaning_fee(df)

df = convert_extra_people(df)

df = convert_instant_bookable(df)

df = convert_require_guest_profile_picture(df)

df = convert_require_guest_phone_verification(df)

df = convert_host_is_superhost(df)

df = convert_is_location_exact(df)
df.shape
# Get a correlation matrix

corr = df.corr()



# Look at variables correlating with our response variable

corr_y = corr['cx_score']



# Plot a horizontal bar chart of the features with > 0.4 correlation (either positive or negative)

fontsize = 10

plt.figure(figsize=(15,10))

corr_y[np.abs(corr_y) > 0.4].sort_values(ascending=False).plot.barh()
df = df[corr_y[np.abs(corr_y) > 0.45].index.values]

df2.shape
from sklearn.model_selection import train_test_split



# Train and test datasets

train, test = train_test_split(df, test_size=0.3, random_state=0)



# We separate out the response variable from the other variables

X_train = train.drop(['cx_score'], axis=1)

y_train = train['cx_score']



X_test = test.drop(['cx_score'], axis=1)

y_test = test['cx_score']
from sklearn import linear_model

from sklearn.metrics import r2_score, mean_squared_error



ols = linear_model.LinearRegression()

ols.fit(X_train, y_train)

y_train_preds = ols.predict(X_train)

r2_score(y_train, y_train_preds)
np.sqrt(mean_squared_error(y_train, y_train_preds))
y_test_preds = ols.predict(X_test)

r2_score(y_test, y_test_preds)
np.sqrt(mean_squared_error(y_test, y_test_preds))
from sklearn.linear_model import RidgeCV



reg = RidgeCV(cv=6)

reg.fit(X_train, y_train)

reg.score(X_train, y_train)
y_train_preds = reg.predict(X_train)

np.sqrt(mean_squared_error(y_train, y_train_preds))
reg.score(X_test, y_test)
y_test_preds = reg.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_test_preds))
coefs = pd.DataFrame(reg.coef_, index=X_train.columns)

coefs.columns = ['Coefficient']

coefs.sort_values(by=['Coefficient'], ascending=False).head(5)
coefs.sort_values(by=['Coefficient'], ascending=True).head(5)