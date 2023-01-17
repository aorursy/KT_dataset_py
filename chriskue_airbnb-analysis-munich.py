# Import all the libraries which will be needed later

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.utils import shuffle



%matplotlib inline
df_calendar = pd.read_csv("../input/munich-airbnb-data/calendar.csv");

df_listings = pd.read_csv("../input/munich-airbnb-data/listings.csv");

df_reviews = pd.read_csv("../input/munich-airbnb-data/reviews.csv");
# Take a look at a concise summary of the DataFrame 'calendar'

df_calendar.info()
# Show the first 5 rows of the data set

df_calendar.head(5)
# List all features in this data set and show the number of missing values

obj = df_calendar.isnull().sum()

for key,value in obj.iteritems():

    percent = round((value * 100 / df_calendar['listing_id'].index.size),3)

    print(key,", ",value, "(", percent ,"%)")
# Take a look at a concise summary of the DataFrame 'listings'

df_listings.info()
# Show the first 5 rows of the data set

df_listings.head(5)
# List all features in this data set and show the number of missing values

obj = df_listings.isnull().sum()

for key,value in obj.iteritems():

    percent = round((value * 100 / df_listings['id'].index.size),3)

    print(key,", ",value, "(", percent ,"%)")
# Count distinct observations per feature

df_listings.nunique()
# Take a look at a concise summary of the DataFrame 'reviews'

df_reviews.info()
# Show the first 5 rows of the data set

df_reviews.head(5)
# List all features in this data set and show the number of missing values

obj = df_reviews.isnull().sum()

for key,value in obj.iteritems():

    percent = round((value * 100 / df_reviews['id'].index.size),3)

    print(key,", ",value, "(", percent ,"%)")
# Plot the number reviews over time to see any patterns

df_reviews_plot = df_reviews.groupby('date')['id'].count().reset_index()

df_reviews_plot["rolling_mean"] = df_reviews_plot.id.rolling(window=30).mean()

df_reviews_plot['date'] = pd.to_datetime(df_reviews_plot['date'])



plt.figure(figsize=(20, 10));

plt.plot(df_reviews_plot.date, df_reviews_plot.rolling_mean);



plt.title("Number of reviews by date (monthly mean)");

plt.xlabel("time");

plt.ylabel("reviews");

plt.grid()
# Copy the data to a new DataFrame for further clean up

df_calendar_clean = df_calendar.copy(deep=True)
# Clean up the data set "calendar" as the previous analysis pointed out



# Drop "adjusted_price"

df_calendar_clean = df_calendar_clean.drop("adjusted_price", axis = 1)



# Remove missing values

df_calendar_clean.dropna(how='all', inplace=True)



# Convert the data type of feature 'date' from object to DateTime

df_calendar_clean['date'] = pd.to_datetime(df_calendar_clean['date'])



# clean up the format of the 'price' values. Maybe not the best solution - but will do the job

df_calendar_clean['price'] = df_calendar_clean['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)



# Convert the feature 'available' to boolean data type

# (This conversion is actually not necessary for further analysis)

df_calendar_clean['available'] = df_calendar_clean['available'].replace({'t': True, 'f':False})
# Group the data by mean price per date

df_calendar_clean = df_calendar_clean.groupby('date')['price'].mean().reset_index()
# Plot the mean price over time

plt.figure(figsize=(20, 10));

plt.plot(df_calendar_clean.date, df_calendar_clean.price);



plt.title("Mean price by date");

plt.xlabel("date");

plt.ylabel("price");

plt.grid();
# Create a new feature 'month'

df_calendar_clean["month"] = df_calendar_clean["date"].dt.month



# Show a Boxplot to see the price distribution per month

plt.figure(figsize=(20, 10))

boxplot = sns.boxplot(x = 'month',  y = 'price', data 

                      = df_calendar_clean).set_title('Distribution of price per month');
# Create a new feature 'weekday'

df_calendar_clean["weekday"] = df_calendar_clean["date"].dt.weekday_name



# Show a Boxplot to see the price distribution per weekday

plt.figure(figsize=(20, 10))

sns.boxplot(x = 'weekday',  y = 'price', data 

            = df_calendar_clean).set_title('Distribution of price per weekday');
# Show the statistics per weekday

df_calendar_clean.groupby(['weekday'])['price'].describe()
# Show the statistics per per week of the year

df_calendar_clean["week"] = df_calendar_clean["date"].dt.week

df_calendar_clean.groupby(['week'])['price'].describe()
# Show a Boxplot to see the price distribution per week

plt.figure(figsize=(20, 10))

sns.boxplot(x = 'week',  y = 'price', data 

            = df_calendar_clean).set_title('Distribution of price per week');
# Copy the data to a new DataFrame for further clean up

df_listings_clean = df_listings.copy(deep=True)
# Clean up the data set "listings" as the previous analysis pointed out



# Drop features which are not used further 

features_to_drop = ['listing_url', 'picture_url','host_url', 'host_thumbnail_url', 'host_picture_url',

                    'name', 'summary', 'space', 'neighborhood_overview', 'transit', 'interaction', 'description',

                    'host_name', 'host_location', 'host_neighbourhood', 'street', 'last_scraped', 'zipcode',

                    'calendar_last_scraped', 'first_review', 'last_review', 'host_since', 'calendar_updated',

                    'experiences_offered', 'state', 'country', 'country_code', 'city', 'market',

                    'host_total_listings_count', 'smart_location']

df_listings_clean.drop(features_to_drop, axis=1, inplace=True)
# Remove constant features by finding unique values per feature 

df_listings_clean = df_listings_clean[df_listings_clean.nunique().where(df_listings_clean.nunique()!=1).dropna().keys()]



# Drop features with 50% or more missing values

more_than_50 = list(df_listings_clean.columns[df_listings_clean.isnull().mean() > 0.5])

df_listings_clean.drop(more_than_50, axis=1, inplace=True)



# Clean up the format values. Maybe not the best solution - but will do the job.

df_listings_clean['price'] = df_listings_clean['price'].replace('[\$,]', '', regex=True).astype(float)

df_listings_clean['extra_people'] = df_listings_clean['extra_people'].replace({'\$': '', ',': ''}, regex=True).astype(float)

df_listings_clean['cleaning_fee'] = df_listings_clean['cleaning_fee'].replace({'\$': '', ',': ''}, regex=True).astype(float)

        

# Convert rates type from string to float and remove the % sign

df_listings_clean['host_response_rate'] = df_listings_clean['host_response_rate'].str.replace('%', '').astype(float)

df_listings_clean['host_response_rate'] = df_listings_clean['host_response_rate'] * 0.01

    

# Covert boolean data from string data type to boolean

boolean_features = ['instant_bookable', 'require_guest_profile_picture', 

                'require_guest_phone_verification', 'is_location_exact', 'host_is_superhost', 'host_has_profile_pic', 

                'host_identity_verified']

df_listings_clean[boolean_features] = df_listings_clean[boolean_features].replace({'t': True, 'f': False})



## Fill numerical missing data with mean value

numerical_feature = df_listings_clean.select_dtypes(np.number)

numerical_columns = numerical_feature.columns



imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imp_mean = imp_mean.fit(numerical_feature)



df_listings_clean[numerical_columns] = imp_mean.transform(df_listings_clean[numerical_columns])

     

# Remove all remaining missing values  

df_listings_clean.dropna(inplace=True)
# Show price statistic for each neighbourhood  

df_listings_clean.groupby(["neighbourhood_cleansed"])["price"].describe()
# Create new feature 'mean' with the mean price per neighbourhood

df_listings_clean['mean']=df_listings_clean.groupby('neighbourhood_cleansed')['price'].transform(lambda r : r.mean())
df_listings_plot = df_listings_clean

df_listings_plot = df_listings_plot.groupby('neighbourhood_cleansed')[['price']].mean()

df_listings_plot = df_listings_plot.reset_index()

df_listings_plot = df_listings_plot.sort_values(by='price',ascending=False)

df_listings_plot.plot.bar(x='neighbourhood_cleansed', y='price', color='blue', rot=90, figsize = (20,10)).set_title('Mean Price per Neighbourhood');
# Since we also have the geo data (latitude and longitude) of the apartments we can create a map

fig = px.scatter_mapbox(df_listings_clean, color="mean", lat='latitude', lon='longitude',

                        center=dict(lat=48.137154, lon=11.576124), zoom=10,

                        mapbox_style="stamen-terrain",width=1000, height=800);

fig.show()
# Copy the data to a new DataFrame for encoding 

df_listings_encoded = df_listings_clean.copy(deep=True)
# Show the remaining features and the data type

df_listings_encoded.info()
# Remove outliers of the feature 'price" - drop all higher than 90% quantile

outliers = df_listings_encoded["price"].quantile(0.90)

df_listings_encoded = df_listings_encoded[df_listings_encoded["price"] < outliers]
# Encode features for use in machine learing model



# Encode feature 'amenities' and concat the data

df_listings_encoded.amenities = df_listings_encoded.amenities.str.replace('[{""}]', "")

df_amenities = df_listings_encoded.amenities.str.get_dummies(sep = ",")

df_listings_encoded = pd.concat([df_listings_encoded, df_amenities], axis=1) 



# Encode feature 'host_verification' and concat the data

df_listings_encoded.host_verifications = df_listings_encoded.host_verifications.str.replace("['']", "")

df_verification = df_listings_encoded.host_verifications.str.get_dummies(sep = ",")

df_listings_encoded = pd.concat([df_listings_encoded, df_verification], axis=1)

    

# Encode feature 'host_response_time'

dict_response_time = {'within an hour': 1, 'within a few hours': 2, 'within a day': 3, 'a few days or more': 4}

df_listings_encoded['host_response_time'] = df_listings_encoded['host_response_time'].map(dict_response_time)



# Encode the remaining categorical feature 

for categorical_feature in ['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type', 'neighbourhood', 

                            'cancellation_policy']:

    df_listings_encoded = pd.concat([df_listings_encoded, 

                                     pd.get_dummies(df_listings_encoded[categorical_feature])],axis=1)

        

# Drop features

df_listings_encoded.drop(['amenities', 'neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type', 

                          'host_verifications', 'neighbourhood','cancellation_policy','security_deposit',

                          'id', 'host_id', 'mean', 'latitude', 'longitude'],

                         axis=1, inplace=True)
# Last check if there are any missing values in the data set

sum(df_listings_encoded.isnull().sum())
# Shuffle the data to ensure a good distribution

df_listings_encoded = shuffle(df_listings_encoded)



X = df_listings_encoded.drop(['price'], axis=1)

y = df_listings_encoded['price']



# Split the data into random train and test subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
 # Initalize the model

model = RandomForestRegressor(max_depth=15, n_estimators=100, criterion='mse', random_state=42)

# Fit the model on training data

model.fit(X, y)

        

# Predict results

prediction = model.predict(X_test)
# Evaluate the result - compare r squared of the training set with the test set



# Find R^2 on training set

print("Training Set:")

print("R_squared:", round(model.score(X_train, y_train) ,2))



# Find R^2 on testing set

print("\nTest Set:")

print("R_squared:", round(model.score(X_test, y_test), 2))
# Scatter plot of th actual vs predicted data

plt.figure(figsize=(10, 10))

plt.grid()

plt.xlim((0, 200))

plt.ylim((0, 200))

plt.plot([0,200],[0,200], color='#AAAAAA', linestyle='dashed')

plt.scatter(y_test, prediction, alpha=0.5)

coef = np.polyfit(y_test,prediction,1)

poly1d_fn = np.poly1d(coef) 

plt.plot(y_test, poly1d_fn(y_test))

plt.title('Actual vs. Predicted data');

plt.xlabel("Actual values")

plt.ylabel("Predicted values")

plt.show()
# Sort the importance of the features

importances = model.feature_importances_

    

values = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)

feature_importances = pd.DataFrame(values, columns = ["feature", "score"])

feature_importances = feature_importances.sort_values(by = ['score'], ascending = False)



features = feature_importances['feature'][:10]

y_feature = np.arange(len(features))

score = feature_importances['score'][:10]



# Plot the importance of a feature to the price

plt.figure(figsize=(20,10));

plt.bar(y_feature, score, align='center');

plt.xticks(y_feature, features, rotation='vertical');

plt.xlabel('Features');

plt.ylabel('Score');

plt.title('Importance of features (TOP 10)');