# Importing the libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Importing the dataset

df = pd.read_csv('../input/listings_summary.csv')
df.columns
# Dropping not necessary columns. Criteria: intuition and common sense.

df.drop(['listing_url', 'scrape_id', 'last_scraped', 'experiences_offered', 'neighborhood_overview',

        'transit', 'access', 'interaction', 'house_rules',

       'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',

       'host_about', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location',

       'host_acceptance_rate', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_listings_count',

       'host_total_listings_count', 'host_verifications',

       'host_has_profile_pic', 'host_identity_verified', 'street',

       'neighbourhood', 'neighbourhood_cleansed', 'host_is_superhost',

       'city', 'state', 'zipcode', 'market', 'weekly_price', 'monthly_price', 

       'smart_location', 'country_code', 'country','calendar_updated', 'has_availability',

       'availability_30', 'availability_60', 'availability_90', 'instant_bookable',

       'availability_365', 'calendar_last_scraped', 'number_of_reviews', 'is_location_exact',

       'first_review', 'last_review', 'requires_license','maximum_nights',

       'license', 'jurisdiction_names', 'require_guest_profile_picture', 'require_guest_phone_verification',

       'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',

       'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value',

       'calculated_host_listings_count', 'reviews_per_month', 'is_business_travel_ready', 'minimum_nights'],

        axis=1, inplace=True)
# Checking for duplicates

df.duplicated().sum()
# Checking missing values

df.isna().sum()
# Setting 'id' as an index

df = df.set_index('id')
# Dropping a few more columns with extremely high level of NaN

# Except security_deposit and cleaning_fee because here NaN seems to be 0

df.drop(['space', 'notes', 'square_feet', 'host_response_time', 'host_response_rate'], axis=1, inplace=True)
# Taking care of missing values in several columns



# NaNs in bathrooms and bedrooms can be replaced by 1 to be on a safe side

df.bathrooms.fillna(1, inplace=True)

df.bedrooms.fillna(1, inplace=True)

# NaNs in beds can be replaced by value in neighbour column 'accommodates' 

df.beds.fillna(df['accommodates'], inplace=True)
# Communities deployment

plt.figure(figsize=(10,5))

sns.countplot(y=df['neighbourhood_group_cleansed'], order = df.neighbourhood_group_cleansed.value_counts().index)

plt.xlabel('Quantity of listings', fontsize='medium')

plt.ylabel('')

plt.title('Communities deployment', fontsize='large')
# Property type deployment - TOP-10 types

plt.figure(figsize=(15,5))

sns.countplot(df['property_type'], order = df.property_type.value_counts().iloc[:10].index)

plt.xlabel('')

plt.ylabel('Quantity of listings', fontsize='large')

plt.title('Property type')
# Room type deployment

plt.figure(figsize=(5,5))

sns.countplot(df['room_type'], order = df.room_type.value_counts(normalize=True).index)

plt.xlabel('')

plt.ylabel('Quantity of listings', fontsize='large')

plt.title('Room type')
# Cleaning (replace '$') and formating price-related columns 

df.price = df.price.str.replace('$', '').str.replace(',', '').astype(float)

df.security_deposit = df.security_deposit.str.replace('$', '').str.replace(',', '').astype(float)

df.cleaning_fee = df.cleaning_fee.str.replace('$', '').str.replace(',', '').astype(float)

df.extra_people = df.extra_people.str.replace('$', '').str.replace(',', '').astype(float)



# NaNs in security_deposit and cleaning_fee seem to be 0

df.security_deposit.fillna(0, inplace=True)

df.cleaning_fee.fillna(0, inplace=True)
# Checking suspiciously low prices

print(df[(['price', 'name'])][(df.price < 10)])
# Dropping rows with price <8$

df = df.drop(df[df.price < 8].index)
# Checking suspiciously high prices

df['price'].plot(kind='box', xlim=(0, 600), vert=False, figsize=(16,1));
# Dropping rows with extremely high price

df = df.drop(df[df.price > 380].index)
# Extract numbers that may contain info re square of rooms from 'description' columns (contains ' s/m/S/M')

df['room_size'] = df['description'].str.extract("(\d{2,3}\s[smSM])", expand=True)

df['room_size'] = df['room_size'].str.replace("\D", "").astype(float)



rv = len(df) - df['room_size'].isna().sum()

print('Real values in "room_size" column:      ', rv)

print('Real values in "room_size" column (%):  ', round(rv/len(df)*100,1), '%')



# (C) This cell of code was taken from the original research, done by Britta Bettendorf
# Extract numbers that may contain info re square of rooms from 'name' columns (contains ' s/m/S/M')

df['room_size_name'] = df['name'].str.extract("(\d{2,3}\s[smSM])", expand=True)

df['room_size_name'] = df['room_size_name'].str.replace("\D", "").astype(float)



rv = len(df) - df['room_size_name'].isna().sum()

print('Real values in "room_size_name" column:      ', rv)

print('Real values in "room_size_name" column (%):  ', round(rv/len(df)*100,1), '%')



# (C) This cell of code was taken from the original research, done by Britta Bettendorf
df.room_size.fillna(0, inplace=True)
# Updating column 'room_size' with values extracted from column 'name'

df.loc[df['room_size'] == 0, 'room_size'] = df['room_size_name']
# We don't need it any more

df.drop(['room_size_name'], axis=1, inplace=True)
# Checking suspiciously low sizes

print(df[(['room_size', 'name'])][(df.room_size < 10)])
# Dropping rows with suspiciously low sizes

df = df.drop(df[df.room_size < 10].index)
# Checking suspiciously high sizes

df['room_size'].plot(kind='box', vert=False, figsize=(16,1));
print(df[(['room_size', 'name'])][(df.room_size > 250)])
# Dropping values of suspiciously high sizes

df.loc[df['room_size'] > 250, 'room_size'] = ''

df.room_size.replace(to_replace='', value=np.nan, inplace=True)
# We have 'NaN's in our column, 2/3 of all values

df.room_size.isna().sum()
df.isna().sum()
# New df for further regression

df_temp = df[['neighbourhood_group_cleansed', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',

              'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'room_size']]
print(df_temp.shape)

df_temp.head(10).transpose()
# Taking care of categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

categorical_cols = ['neighbourhood_group_cleansed']

df_temp[categorical_cols] = df_temp[categorical_cols].apply(lambda col: labelencoder_X.fit_transform(col.astype(str)))

df_temp.head(10).transpose()
# Arranging datasets by existence of 'room_size' value

train_set = df_temp[df_temp['room_size'].notnull()]

test_set  = df_temp[df_temp['room_size'].isnull()]



# Arranging X-training and X-testing datasets

X_train = train_set.drop('room_size', axis=1)

X_test  = test_set.drop('room_size', axis=1)



# Arranging y-training datasets

y_train = train_set['room_size']
# Regression model

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 300, random_state = 123)

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test) 
# Indroduction of predicted data to the main dataset 'df'

y_pred = pd.DataFrame(y_pred)

y_pred.columns = ['room_size']
temp_id = pd.DataFrame(X_test.index)

temp_id.columns = ['temp_id']



y_pred = pd.concat([y_pred, temp_id], axis=1)

y_pred.set_index(['temp_id'], inplace=True)



df_pred = pd.concat([X_test, y_pred], axis=1)

df_pred.head()
df_pred.shape
train_set.shape
df_temp = pd.DataFrame()

df_temp = pd.concat([df_pred, train_set], axis=0)

print(df_temp.shape)

df_temp.head().transpose()
# Checking again suspiciously low sizes

print(df_temp[(['room_size'])][(df_temp.room_size < 10)])
# Checking again suspiciously high sizes

df_temp['room_size'].plot(kind='box', vert=False, figsize=(16,1));
print(df.shape)

df.head(2).transpose()
print(df_temp.shape)

df_temp.head().transpose()
df = df[['property_type', 'amenities', 'cancellation_policy']]

print(df.shape)

df.isna().sum()
df = pd.concat([df, df_temp], axis=1)

df.head(3).transpose()
print(df.shape)

df.isna().sum()
# Let's explore amenities

pd.set_option('display.max_colwidth', -1)

df.amenities.head(5)
# Let's introduce new column with score of amenities

df['amen_score'] = df['amenities'].str.count(',') +1
# We don't need it any more

df.drop(['amenities'], axis=1, inplace=True)
df.head().transpose()
df.isna().sum()
# Taking care of categorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

categorical_cols = ['property_type','cancellation_policy']

df[categorical_cols] = df[categorical_cols].apply(lambda col: labelencoder_X.fit_transform(col.astype(str)))

df.head(10).transpose()
# Creating DV and IV sets

X = df.drop('price', axis=1)

y = df['price']



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=123)
# Gradient Boosting Regression

from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor(n_estimators = 100, max_depth = 3, min_samples_split = 2, learning_rate = 0.1)

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



# Finding the mean_squared error (MSE)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)



# Finding the r2 score or the variance (R2)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_test, y = y_test, cv = 10)



# Printing metrics

print("RMSE Error:", round(np.sqrt(mse), 2))

print("R2 Score:", round(r2, 4))

print("Mean accuracy:", round(accuracies.mean(), 2))

print("Std deviation:", round(accuracies.std(), 4))
import tensorflow as tf
# Establish feeding type for this version of TF

df_tf = df.apply(lambda x: x.astype('float32'))
# Creating DV and IV sets

X_tf = df_tf.drop('price', axis=1)

y_tf = df_tf['price']



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tf, y_tf, test_size = 0.25, random_state=123)
# Feature columns

property_type = tf.feature_column.numeric_column('property_type')

cancellation_policy = tf.feature_column.numeric_column('cancellation_policy')

neighbourhood_group_cleansed = tf.feature_column.numeric_column('neighbourhood_group_cleansed')

accommodates = tf.feature_column.numeric_column('accommodates')

bathrooms = tf.feature_column.numeric_column('bathrooms')

bedrooms = tf.feature_column.numeric_column('bedrooms')

beds = tf.feature_column.numeric_column('beds')

security_deposit = tf.feature_column.numeric_column('security_deposit')

cleaning_fee = tf.feature_column.numeric_column('cleaning_fee')

guests_included = tf.feature_column.numeric_column('guests_included')

room_size = tf.feature_column.numeric_column('room_size')

amen_score = tf.feature_column.numeric_column('amen_score')
feat_cols = [property_type, cancellation_policy, neighbourhood_group_cleansed, accommodates, bathrooms,

             bedrooms, beds, security_deposit, cleaning_fee, guests_included, room_size, amen_score]
# Input function

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=1000,shuffle=True)
# Creating and training model

model = tf.estimator.DNNRegressor(hidden_units=[12,12,12], feature_columns = feat_cols)
model.train(input_fn=input_func, steps = 10000)
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = list(model.predict(pred_input_func))
y_pred = []

for i in predictions:

    y_pred.append(i['predictions'][0])
from sklearn.metrics import mean_squared_error

tf_mse = mean_squared_error(y_test, y_pred)

print("MSE Error:", round(tf_mse, 2))

print("RMSE Error:", round(np.sqrt(tf_mse), 2))
print("Gradient Boosting Regression RMSE Error: ", round(mse**0.5, 2))

print("TensorFlow DNN Regression RMSE Error:    ", round(tf_mse**0.5, 2))