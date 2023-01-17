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
df_listings = pd.read_csv("../input/munich-airbnb-data/listings.csv");
df_calendar = pd.read_csv("../input/munich-airbnb-data/calendar.csv");
df_reviews = pd.read_csv("../input/munich-airbnb-data/reviews.csv");
# Take a look at a concise summary of the DataFrame 'calendar'
df_calendar.info()
# Show the first rows of the dataframe
df_calendar.head()
# List all features in this data set and show the number of missing values
obj = df_calendar.isnull().sum()
for key,value in obj.iteritems():
    percent = round((value * 100 / df_calendar['listing_id'].index.size),3)
    print(key,", ",value, "(", percent ,"%)")
# Show the shape of the dataframe
df_calendar.shape
# Take a closer look at a concise summary of the DataFrame 'listings'
df_listings.info()
# Show the first rows of the dataframe
df_listings.iloc[:,100:120].head()
# List all features in this data set and show the number of missing values
obj = df_listings.isnull().sum()
for key,value in obj.iteritems():
    percent = round((value * 100 / df_listings['id'].index.size),3)
    print(key,", ",value, "(", percent ,"%)")
# Show distinct observations per feature and absolute frequency
df_listings["zipcode"].value_counts()
# Show the shape of the dataframe
df_listings.shape
# Count distinct observations per feature
df_listings.nunique()
# Take a look at a concise summary of the DataFrame 'reviews'
df_reviews.info()
# Show the first rows of the dataframe
df_reviews.head()
# List all features in this data set and show the number of missing values
obj = df_reviews.isnull().sum()
for key,value in obj.iteritems():
    percent = round((value * 100 / df_reviews['id'].index.size),3)
    print(key,", ",value, "(", percent ,"%)")
# Show the shape of the dataframe
df_reviews.shape
# Copy the data to a new DataFrame for further clean up
df_listings_clean = df_listings.copy(deep=True)
# Clean up the data set "listings" as the previous analysis pointed out

# Drop features which are not used further 
features_to_drop = ['listing_url', 'picture_url','host_url', 'host_thumbnail_url', 'host_picture_url',
                    'name', 'summary', 'space', 'neighborhood_overview', 'transit', 'interaction', 'description',
                    'host_name', 'host_location', 'host_neighbourhood', 'street', 'last_scraped',
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
# Show mean price for each neighbourhood_cleansed
df_listings_clean.groupby(["neighbourhood_cleansed"])["price"].describe().sort_values("mean", ascending=False)
# Create new feature 'mean' with the mean price per neighbourhood
df_listings_clean['mean'] = df_listings_clean.groupby('neighbourhood_cleansed')['price'].transform(lambda r : r.mean())
# Create plot for mean price per neighbourhood 
df_listings_plot = df_listings_clean
df_listings_plot = df_listings_plot.groupby('neighbourhood_cleansed')[['price']].mean()
df_listings_plot = df_listings_plot.reset_index()
df_listings_plot = df_listings_plot.sort_values(by='price',ascending=False)
df_listings_plot.plot.bar(x='neighbourhood_cleansed', y='price', color='blue', rot=90, figsize = (20,10)).set_title('Mean Price per city district (Neighbourhood)');
# Since we also have the geo data (latitude and longitude) of the apartments we can create a map
fig = px.scatter_mapbox(df_listings_clean, color="mean", lat='latitude', lon='longitude',
                        center=dict(lat=48.137154, lon=11.576124), zoom=10,
                        mapbox_style="open-street-map",width=1000, height=800);
fig.show()
# Create bins for mean price
bins = [0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400]
df_listings_clean['binned'] = pd.cut(df_listings_clean['mean'], bins)
df_listings_clean
df_listings_clean["latitude"].describe()
df_listings_clean["longitude"].describe()
# Create squares for langitude and latitude
step = 0.0025
to_bin = lambda x: np.floor(x / step) * step
df_listings_clean["latbin"] = df_listings_clean.latitude.map(to_bin)
df_listings_clean["longbin"] = df_listings_clean.longitude.map(to_bin)
#groups = df_listings_clean.groupby(("latbin", "longbin"))
# Create df_squares
df_squares = df_listings_clean[['neighbourhood_cleansed','zipcode','latitude','longitude','latbin','longbin','mean']]
df_squares
# Show creates squares on map
fig = px.scatter_mapbox(df_listings_clean, color="mean", lat='latbin', lon='longbin',
                        center=dict(lat=48.137154, lon=11.576124), zoom=10, 
                        mapbox_style="open-street-map",width=1000, height=800);
fig.show()
# Create dataset with octoberfest coordinates  
oct_data = {
        'id': ['Octoberfest',],
        'latitude': ['48.131'],
        'longitude': ['11.550']}
df_oct = pd.DataFrame(oct_data, columns = ['id', 'latitude', 'longitude'])
df_oct
# Show place where octoberfest is located    
fig = px.scatter_mapbox(df_oct, lat='latitude', lon='longitude',
                        center=dict(lat=48.137154, lon=11.576124), zoom=10,
                        mapbox_style="open-street-map",width=1000, height=800);
fig.show()
### Getting distance between two points based on latitude/longitude
from math import sin, cos, sqrt, atan2, radians

# approximate radius of earth in km
R = 6373.0

# places
lat_oct = radians(48.131)
lon_oct = radians(11.550)
lat_room = radians(48.14)
lon_room = radians(11.58)

dlon = lon_room - lon_oct
dlat = lat_room - lat_oct

a = sin(dlat / 2)**2 + cos(lat_oct) * cos(lat_room) * sin(dlon / 2)**2
c = 2 * atan2(sqrt(a), sqrt(1 - a))

distance = R * c * 1000
distance = round(distance, 2)

print("Result:", distance,"meters")
# create copy
df_squares_oct = df_squares.copy()
df_squares_oct
### Getting distance between two points based on latitude/longitude
from math import sin, cos, sqrt, atan2, radians

# approximate radius of earth in km
R = 6373.0

# insert octoberfest coordinates and 
df_squares_oct['lat_oct'] = 48.131
df_squares_oct['long_oct'] = 11.550

# convert to radians
df_squares_oct['latbin_rad'] = np.radians(df_squares_oct['latbin'])
df_squares_oct['longbin_rad'] = np.radians(df_squares_oct['longbin'])
df_squares_oct['lat_oct_rad'] = np.radians(df_squares_oct['lat_oct'])
df_squares_oct['long_oct_rad'] = np.radians(df_squares_oct['long_oct'])

df_squares_oct['dlon'] = df_squares_oct['longbin_rad'] - df_squares_oct['long_oct_rad']
df_squares_oct['dlat'] = df_squares_oct['latbin_rad'] - df_squares_oct['lat_oct_rad']

df_squares_oct['a']  = np.sin(df_squares_oct['dlat'] / 2)**2 + np.cos(df_squares_oct['lat_oct_rad']) * np.cos(df_squares_oct['latbin_rad']) * np.sin(df_squares_oct['dlon'] / 2)**2

df_squares_oct['c'] = 2 * np.arctan2(np.sqrt(df_squares_oct['a']), np.sqrt(1 - df_squares_oct['a']))

df_squares_oct['distance_meter'] = R * df_squares_oct['c'] * 1000
df_squares_oct['distance_meter'] = round(df_squares_oct['distance_meter'], 2)

df_squares_oct
# Show mean price for apartments
fig = px.scatter_mapbox(df_squares_oct, color="mean", lat='latbin', lon='longbin',
                        center=dict(lat=48.137154, lon=11.576124), zoom=10, 
                        mapbox_style="open-street-map",width=1000, height=800);
fig.show()

# Show distance to octoberfest
fig = px.scatter_mapbox(df_squares_oct, color="distance_meter", lat='latbin', lon='longbin',
                        center=dict(lat=48.137154, lon=11.576124), zoom=10, 
                        mapbox_style="open-street-map",width=1000, height=800);
fig.show()
# analyze distance
df_squares_oct['distance_meter'].describe()
# bincount distance using linspace
y = np.linspace(0, 14000, 29) # 500 meter sections
df_squares_oct['distance_bins'] = pd.cut(df_squares_oct['distance_meter'], bins=y)
df_squares_oct
# Analyze price
df_squares_oct['mean'].describe()
# Bincount price using linspace
x = np.linspace(80, 170, 19) # 5 euro sections
df_squares_oct['mean_bins'] = pd.cut(df_squares_oct['mean'], bins=x)
df_squares_oct
### Offline-Quoten: 2cat/1num
# Create heatmap
cat_var_1 = "neighbourhood_cleansed"
cat_var_2 = "distance_bins"
num_var_1 = "mean"

# Create heatmap
plt.figure(figsize=(30, 15))
cat_means = df_squares_oct.groupby([cat_var_1, cat_var_2]).mean()[num_var_1]
cat_means = cat_means.reset_index(name = 'avg')
cat_means = cat_means.pivot(index = cat_var_2, columns = cat_var_1,
                            values = 'avg')
sns.heatmap(cat_means, annot = True, fmt = '.4f', cmap="coolwarm",
           cbar_kws = {'label' : 'mean(avg)'})

plt.title("Fig.A: Compare distance to octoberfest in meters, average price for a room according to different districts", 
          fontsize=12, loc='center', pad=10)
# Relative Häufigkeit von Offline-Nacharbeiten i.A. des Modells
plt.figure(figsize=(20, 10))
sns.lineplot(x="distance_meter", y="mean", data=df_squares_oct, err_style=None)
plt.title("Fig.B: Compare distance to octoberfest and average price for a room", fontsize=12, loc='center', pad=10)
# Relative Häufigkeit von Offline-Nacharbeiten i.A. des Modells
plt.figure(figsize=(20, 15))
sns.lineplot(x="distance_meter", y="mean", hue="neighbourhood_cleansed", data=df_squares_oct, err_style=None)
plt.title("Fig.C: Compare distance to octoberfest and average price for a room with regards to district", fontsize=12, loc='center', pad=10)
# Copy the data to a new DataFrame for encoding 
df_listings_encoded = df_listings_clean.copy(deep=True)
df_listings_encoded
# Filter dataframe according to TOP 5 city districts in munich in terms of distance-price-ratio 
filter = ['Sendling','Sendling-Westpark','Untergiesing-Harlaching','Laim','Neuhausen-Nymphenburg']
df_listings_encoded = df_listings_encoded[df_listings_encoded.neighbourhood_cleansed.isin(filter)]
df_listings_encoded
# control filter activities, correct!
df_listings_encoded.neighbourhood_cleansed.unique()
# Shape of dataset
df_listings_encoded.shape
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

X = df_listings_encoded.drop(['price',"zipcode","binned"], axis=1)
y = df_listings_encoded['price']

# Split the data into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Control shape after train-test-split, correct!
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# Initalize the model
model = RandomForestRegressor(max_depth=15, n_estimators=100, criterion='mse', random_state=42)
model
# Fit the model on training data
model.fit(X, y)
# Predict results
prediction = model.predict(X_test)
prediction
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