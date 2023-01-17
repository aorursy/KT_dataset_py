# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns



import folium

from folium import plugins

from folium.plugins import MarkerCluster



from sklearn import preprocessing



# Importing dataset

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split



import plotly.graph_objs as go



from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from math import sqrt

from sklearn.metrics import r2_score

import numpy as np



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

import xgboost



import plotly as plotly

from plotly import __version__

import plotly.offline as py 

from plotly.offline import init_notebook_mode, plot

init_notebook_mode(connected=True)

from plotly import tools

import plotly.graph_objs as go

import plotly.express as px



print ("Libraries imported")
# Read the dataset

nyc_data_airbnb_raw = pd.read_csv('../input/Airbnb_Newyork_Listings_Detailed.csv')
# Examine the dataset

nyc_data_airbnb_raw.info()
# Get the number of rows and columns of the dataset

nyc_data_airbnb_raw.shape
# Check data tyes of raw data columns

nyc_data_airbnb_raw.dtypes
# View raw data. Top three entires only

nyc_data_airbnb_raw.head(3)
# Copy dataset to save the original dataset which can be used again later

nyc_data_airbnb = nyc_data_airbnb_raw.copy()

print("Dataset copied")
nyc_data_airbnb.shape
# Check unique values for string type columns



for col in nyc_data_airbnb[['experiences_offered', 'street', 'property_type', 'room_type',

                 'bed_type','amenities']]:

    print('Unique values in column: %s' %col)

    print(nyc_data_airbnb[col].unique()), '\n'
# Removing unimportant columns based on observation

drop_columns = ['id','listing_url','scrape_id','last_scraped','summary','space',

                'description','neighborhood_overview','notes','transit','access',

                'interaction','house_rules','thumbnail_url','medium_url','picture_url',

                'xl_picture_url','host_url','host_name','host_since','host_location',

                'host_about','host_thumbnail_url','host_picture_url',

                'host_verifications','host_is_superhost','host_total_listings_count','host_has_profile_pic',

                'host_identity_verified','require_guest_profile_picture','require_guest_phone_verification',

                'is_location_exact', 'extra_people','calendar_updated','has_availability',

                'calendar_last_scraped','first_review','last_review','license','jurisdiction_names',

                'calculated_host_listings_count','calculated_host_listings_count_entire_homes',

                'calculated_host_listings_count_private_rooms','calculated_host_listings_count_shared_rooms',

                'city','state','zipcode','market','smart_location','country_code','country',

                'experiences_offered','street','weekly_price','monthly_price',

                'minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights',

                'maximum_maximum_nights','minimum_nights_avg_ntm','maximum_nights_avg_ntm',

                'review_scores_cleanliness','review_scores_checkin','review_scores_communication',

                'review_scores_location','number_of_reviews_ltm','review_scores_value']

nyc_data_airbnb.drop(columns=drop_columns,inplace=True)



print("Unimportant columns dropped")
# Drop all NaN values for square_feet & host_acceptance_rate

nyc_data_airbnb.drop(columns = {'square_feet'},inplace=True)

nyc_data_airbnb.drop(columns = {'host_acceptance_rate'},inplace=True)



print("NaN columns dropped")
# Format host_response_rate, remove % sign and convert data type tp float for calculations

nyc_data_airbnb['host_response_rate'] = nyc_data_airbnb['host_response_rate'].str.replace("%", "").astype("float")

print("host_response_rate formatted")
# Printing unique values for requires_license, instant_bookable & is_business_travel_ready



for col in nyc_data_airbnb[['requires_license', 'instant_bookable', 'is_business_travel_ready']]:

    print('Unique values in column: %s' %col)

    print(nyc_data_airbnb[col].unique()), '\n'
# Format requires_license, instant_bookable & is_business_travel_ready as required and also change their data types to int

nyc_data_airbnb.loc[nyc_data_airbnb['requires_license'] == 'f',['requires_license']] = '1'

nyc_data_airbnb.loc[nyc_data_airbnb['requires_license'] == 't',['requires_license']] = '2'

nyc_data_airbnb.loc[nyc_data_airbnb['instant_bookable'] == 'f',['instant_bookable']] = '1'

nyc_data_airbnb.loc[nyc_data_airbnb['instant_bookable'] == 't',['instant_bookable']] = '2'

nyc_data_airbnb.loc[nyc_data_airbnb['is_business_travel_ready'] == 'f',['is_business_travel_ready']] = '1'

nyc_data_airbnb.requires_license = nyc_data_airbnb.requires_license.astype(int)

nyc_data_airbnb.instant_bookable = nyc_data_airbnb.instant_bookable.astype(int)

nyc_data_airbnb.is_business_travel_ready = nyc_data_airbnb.is_business_travel_ready.astype(int)

print("requires_license, instant_bookable & is_business_travel_ready formatted")
# Format price, security_deposit & cleaning_fee columns and change their type from string to float

nyc_data_airbnb.price = nyc_data_airbnb.price.str.strip('$')

nyc_data_airbnb.price = nyc_data_airbnb.price.str.replace(',','')

nyc_data_airbnb.price = nyc_data_airbnb.price.astype(float)



nyc_data_airbnb.security_deposit = nyc_data_airbnb.security_deposit.str.strip('$')

nyc_data_airbnb.security_deposit = nyc_data_airbnb.security_deposit.str.replace(',','')

nyc_data_airbnb.security_deposit = nyc_data_airbnb.security_deposit.astype(float)



nyc_data_airbnb.cleaning_fee = nyc_data_airbnb.cleaning_fee.str.strip('$')

nyc_data_airbnb.cleaning_fee = nyc_data_airbnb.cleaning_fee.str.replace(',','')

nyc_data_airbnb.cleaning_fee = nyc_data_airbnb.cleaning_fee.astype(float)



print("Price related columns formatted")
print("Check null values in the dataset")

nyc_data_airbnb.isnull().sum()
# Fill empty rows with the mean value of their respective columns

nyc_data_airbnb['name'].fillna(value="Default Name", inplace=True)

nyc_data_airbnb['bathrooms'].fillna(value=np.mean(nyc_data_airbnb['bathrooms']), inplace=True)

nyc_data_airbnb['bedrooms'].fillna(value=np.mean(nyc_data_airbnb['bedrooms']), inplace=True)

nyc_data_airbnb['beds'].fillna(value=np.mean(nyc_data_airbnb['beds']), inplace=True)

nyc_data_airbnb['review_scores_rating'].fillna(value=np.mean(nyc_data_airbnb['review_scores_rating']), inplace=True)

nyc_data_airbnb['review_scores_accuracy'].fillna(value=np.mean(nyc_data_airbnb['review_scores_accuracy']), inplace=True)

nyc_data_airbnb['reviews_per_month'].fillna(value=np.mean(nyc_data_airbnb['reviews_per_month']), inplace=True)

nyc_data_airbnb['security_deposit'].fillna(value=np.mean(nyc_data_airbnb['security_deposit']), inplace=True)

nyc_data_airbnb['cleaning_fee'].fillna(value=np.mean(nyc_data_airbnb['cleaning_fee']), inplace=True)

nyc_data_airbnb['host_response_rate'].fillna(value=np.mean(nyc_data_airbnb['host_response_rate']), inplace=True)

nyc_data_airbnb['host_listings_count'].fillna(value=np.mean(nyc_data_airbnb['host_listings_count']), inplace=True)

print("Empty rows filled with mean")
print("Check null values in the dataset after replacing with mean value")

nyc_data_airbnb.isnull().sum()
# Relationship between price and room type

plt.figure()

sns.scatterplot(x='room_type', y='price', hue="room_type", data=nyc_data_airbnb)

plt.xlabel("Room Type")

plt.ylabel("Price")

plt.title("Room Type vs Price", size=15, weight='bold')
# Relationship between price and neighbourhood_group_cleansed

plt.figure()

sns.scatterplot(x="neighbourhood_group_cleansed", y="price", hue="neighbourhood_group_cleansed", data=nyc_data_airbnb)

plt.xlabel("Neighbourhood Group")

plt.ylabel("Price")

plt.title("Neighbourhood Group vs Price", size=15, weight='bold')
# Relationship between price, room_type and neighbourhood_group_cleansed

plt.figure()

sns.scatterplot(x="room_type", y="price", hue="neighbourhood_group_cleansed", size="neighbourhood_group_cleansed", sizes=(50, 200),

               data=nyc_data_airbnb)

plt.xlabel("Room Type")

plt.ylabel("Price")

plt.title("Room Type vs Price vs Neighbourhood Group",size=15, weight='bold')
# Relationship between listing count and neighbourhood_group_cleansed

plt.figure()

sns.countplot(nyc_data_airbnb["neighbourhood_group_cleansed"])

plt.xlabel("Neighbourhood Group")

plt.ylabel("Count")

plt.title("Neighbourhood Group vs Listing Count", size=15, weight='bold')
# Relationship between listing count and neighbourhood_group_cleansed

plt.figure()

sns.scatterplot(nyc_data_airbnb.longitude,nyc_data_airbnb.latitude,hue=nyc_data_airbnb.neighbourhood_group_cleansed)

plt.xlabel("Longitute")

plt.ylabel("Latitude")

plt.title("Longitute vs Latitude vs Neighbourhood Group", size=15, weight='bold')
# Graph to display which neighbourhood_group_cleansed is relatively more expensive

nda = nyc_data_airbnb[nyc_data_airbnb.price < 250]

plt.figure()

sns.boxplot(y="price",x='neighbourhood_group_cleansed',data=nda)

plt.xlabel("Neighbourhood Group")

plt.ylabel("Price")

plt.title("Neighbourhood Group Price Distribution < 250")
# Relationship between number of room types with each neighbourhood group

plt.figure()

sns.countplot(x = 'room_type',hue = "neighbourhood_group_cleansed",data = nyc_data_airbnb)

plt.xlabel("Room Type")

plt.ylabel("Count")

plt.title("Room types occupied by the neighbourhood_group_cleansed")
# catplot room type vs price

plt.figure()

sns.catplot(x="room_type", y="price", data=nyc_data_airbnb);

plt.xlabel("Room Type")

plt.ylabel("Price")

plt.title("Room type vs Price")
# Plot first 1000 listings with most number of reviews on the map using folium

mostReviewsNycData = nyc_data_airbnb.sort_values(by=['number_of_reviews'], ascending=False).head(1000)
print('Rooms with the most number of reviews')

Lat = 40.7128

Long = -74.0060



mapdf1=folium.Map([Lat,Long],zoom_start=10,)



mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon,label in zip(mostReviewsNycData.latitude,mostReviewsNycData.longitude,mostReviewsNycData.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)



mapdf1.add_child(mapdf1_rooms_map)



mapdf1
#Find 1000 most expensive rooms

mostExpensiveNyc=nyc_data_airbnb.sort_values(by=['price'],ascending=False).head(1000)
print('Most Expensive rooms')

Long=-73.80

Lat=40.80

mapdf1=folium.Map([Lat,Long],zoom_start=10,)



mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon,label in zip(mostExpensiveNyc.latitude,mostExpensiveNyc.longitude,mostExpensiveNyc.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)

mapdf1.add_child(mapdf1_rooms_map)



mapdf1
#Count of room types

nyc_data_airbnb['room_type'].value_counts()
rmtng2 = nyc_data_airbnb.groupby(['neighbourhood_group_cleansed','neighbourhood'])['price'].agg('mean')
rmtng1 = pd.DataFrame(rmtng2)
rmtng1.reset_index(inplace=True)
Bronx = rmtng1[rmtng1['neighbourhood_group_cleansed']=='Bronx']

Brooklyn = rmtng1[rmtng1['neighbourhood_group_cleansed']=='Brooklyn']

Manhattan = rmtng1[rmtng1['neighbourhood_group_cleansed']=='Manhattan']

Queens = rmtng1[rmtng1['neighbourhood_group_cleansed']=='Queens']

StatenIsland = rmtng1[rmtng1['neighbourhood_group_cleansed']=='Staten Island']
Bronx1=Bronx.sort_values(by=['price'],ascending=False).head(10)

Brooklyn1=Brooklyn.sort_values(by=['price'],ascending=False).head(10)

Manhattan1=Manhattan.sort_values(by=['price'],ascending=False).head(10)

Queens1=Queens.sort_values(by=['price'],ascending=False).head(10)

StatenIsland1=StatenIsland.sort_values(by=['price'],ascending=False).head(10)
trace1=go.Scatter(x=Bronx1['neighbourhood'],y=Bronx1['price'],marker=dict(color="crimson", size=12),mode="markers",name="Bronx",)



trace2=go.Scatter(x=Brooklyn1['neighbourhood'],y=Brooklyn1['price'],marker=dict(color="blue", size=12),mode="markers",name="Brooklyn",)



trace3=go.Scatter(x=Manhattan1['neighbourhood'],y=Manhattan1['price'],marker=dict(color="purple", size=12),mode="markers",name="Manhattan",)



trace4=go.Scatter(x=Queens1['neighbourhood'],y=Queens1['price'],marker=dict(color="black", size=12),mode="markers",name="Queens",)



trace5=go.Scatter(x=StatenIsland1['neighbourhood'],y=StatenIsland1['price'],marker=dict(color="red", size=12),mode="markers",name="StatenIsland",)



data = [trace1,trace2,trace3,trace4,trace5]



titles=['Most Pricey neighbourhoods-Bronx',

        'Most Pricey neighbourhoods-Brooklyn',

        'Most Pricey neighbourhoods-Manhattan',

        'Most Pricey neighbourhoods-Queens',

        'Most Pricey neighbourhoods-StatenIsland']



fig =plotly.subplots.make_subplots(rows=3,cols=2,subplot_titles=titles)





fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,2)

fig.append_trace(trace3,2,1)

fig.append_trace(trace4,2,2)

fig.append_trace(trace5,3,1)





fig['layout'].update(height=1200,width=1000,paper_bgcolor='white')



py.iplot(fig,filename='pricetypeplot')
#100 most reviewed listings in NYC

top_reviewed_listings=nyc_data_airbnb.nlargest(100,'number_of_reviews')
price_avrg=top_reviewed_listings.price.mean()

print('Average price per night for top 100 reviewed lisitings: {}'.format(price_avrg))
#initializing empty list where we are going to put our name strings

_names_=[]

#getting name strings from the column and appending it to the list

for name in nyc_data_airbnb.name:

    _names_.append(name)

#setting a function that will split those name strings into separate words   

def split_name(name):

    spl=str(name).split()

    return spl

#initializing empty list where we are going to have words counted

_names_for_count_=[]

#getting name string from our list and using split function, later appending to list above

for x in _names_:

    for word in split_name(x):

        word=word.lower()

        _names_for_count_.append(word)
#we are going to use counter

from collections import Counter

#let's see top 30 used words by host to name their listing

_top_30_w=Counter(_names_for_count_).most_common()

_top_30_w=_top_30_w[0:30]
#now let's put our findings in dataframe for further visualizations

sub_w=pd.DataFrame(_top_30_w)

sub_w.rename(columns={0:'Words', 1:'Count'}, inplace=True)
#we are going to use barplot for this visualization

viz_5=sns.barplot(x='Words', y='Count', data=sub_w)

viz_5.set_title('Counts of the top 30 used words for listing names')

viz_5.set_ylabel('Count of words')

viz_5.set_xlabel('Words')

viz_5.set_xticklabels(viz_5.get_xticklabels(), rotation=80)
# Finding out top 10 neighbourhoods

nyc_data_airbnb.neighbourhood.value_counts().head(10)
# Categorical features coversion. We have done this at the end as we need the actual categories for various plots.

nyc_data_airbnb.host_neighbourhood = nyc_data_airbnb.host_neighbourhood.astype('category').cat.codes

nyc_data_airbnb.neighbourhood = nyc_data_airbnb.neighbourhood.astype('category').cat.codes

nyc_data_airbnb.neighbourhood_cleansed = nyc_data_airbnb.neighbourhood_cleansed.astype('category').cat.codes

nyc_data_airbnb.property_type = nyc_data_airbnb.property_type.astype('category').cat.codes

nyc_data_airbnb.room_type = nyc_data_airbnb.room_type.astype('category').cat.codes

nyc_data_airbnb.bed_type = nyc_data_airbnb.bed_type.astype('category').cat.codes

nyc_data_airbnb.neighbourhood_group_cleansed = nyc_data_airbnb.neighbourhood_group_cleansed.astype('category').cat.codes

nyc_data_airbnb.cancellation_policy = nyc_data_airbnb.cancellation_policy.astype('category').cat.codes

nyc_data_airbnb.host_response_time = nyc_data_airbnb.host_response_time.astype('category').cat.codes

print("Categorical features converted")
#For joint plots

nyc_data_airbnb_plot = nyc_data_airbnb.copy()
# Remove name, host_id and amenities for joint plots

nyc_data_airbnb_plot.drop(columns = {'name','host_id','amenities'},inplace=True)
# Joint plots: to check the linear relationship of features with the target variable price

cols = nyc_data_airbnb_plot.columns.values

for c in cols:

    if c != "price":

        sns.jointplot(x=c, y="price", data=nyc_data_airbnb_plot, kind = 'reg', height = 5)

plt.show()
# Correlation heatmap applied on training set

corr = nyc_data_airbnb.corr() 

plt.figure(figsize=(18, 16))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True)
# Analysing price for removing outliers

print(np.mean(nyc_data_airbnb['price']))

print(np.max(nyc_data_airbnb['price']))

print(np.min(nyc_data_airbnb['price']))
nyc_data_airbnb = nyc_data_airbnb_raw.copy()
nyc_data_airbnb['amenities'].head()
np.concatenate(nyc_data_airbnb['amenities'].map(lambda amns: amns.split("|")).values)
features = nyc_data_airbnb[['host_listings_count', 'host_total_listings_count', 'accommodates', 

                     'bathrooms', 'bedrooms', 'beds', 'price', 'guests_included', 'number_of_reviews',

                     'review_scores_rating']]
nyc_data_airbnb['amenities'] = nyc_data_airbnb['amenities'].map(

    lambda amns: "|".join([amn.replace("}", "").replace("{", "").replace('"', "")\

                           for amn in amns.split(",")])

)
nyc_data_airbnb['amenities'].map(lambda amns: amns.split("|")).head()
#First amenity is empty so exclude that

amenities = np.unique(np.concatenate(nyc_data_airbnb['amenities'].map(lambda amns: amns.split("|"))))[1:]
amenity_arr = np.array([nyc_data_airbnb['amenities'].map(lambda amns: amn in amns) for amn in amenities])

amenity_arr
features = pd.concat([features, pd.DataFrame(data=amenity_arr.T, columns=amenities)], axis=1)
features.head(3)
features.shape
for tf_feature in ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic',

                   'is_location_exact', 'requires_license', 'instant_bookable',

                   'require_guest_profile_picture', 'require_guest_phone_verification']:

    features[tf_feature] = nyc_data_airbnb[tf_feature].map(lambda s: False if s == "f" else True)
for categorical_feature in ['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type']:

    features = pd.concat([features, pd.get_dummies(nyc_data_airbnb[categorical_feature])], axis=1)
for col in features.columns[features.isnull().any()]:

    print(col)
for col in features.columns[features.isnull().any()]:

    features[col] = features[col].fillna(features[col].median())
features.price = features.price.str.strip('$')

features.price = features.price.str.replace(',','')

features.price = features.price.astype(float)
nyc_data_airbnb['host_is_superhost'].head()
features['price'].sort_values().reset_index(drop=True).plot()
fitters = features.query('price <= 600')

fitters.shape
y = fitters['price']

X = fitters.drop(columns = {'price'}, axis = 1)

X.shape
# Train Test Split

# 80-20 Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
import time

# Train Linear Regression model

lm = LinearRegression()

t0=time.time()

lm.fit(X_train,y_train)

print("Training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

t1=time.time()

pred_linear_x = lm.predict(X_test)

print("Prediction time:", round(time.time()-t1, 3), "s")



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': pred_linear_x.flatten()})

print(error_airbnb.head().to_string())
# Train Ridge Regression model

ridge_x = Ridge(alpha = 0.01, normalize = True)

t0=time.time()

ridge_x.fit(X_train,y_train) 

print("Training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

t1=time.time()

pred_ridge_x = ridge_x.predict(X_test) 

print("Prediction time:", round(time.time()-t1, 3), "s")



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': pred_ridge_x.flatten()})

print(error_airbnb.head().to_string())
# Train Lasso Regression model

Lasso_x = Lasso(alpha = 0.001, normalize =False)

t0=time.time()

Lasso_x.fit(X_train,y_train)

print("Training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

t1=time.time()

pred_Lasso_x = Lasso_x.predict(X_test) 

print("Prediction time:", round(time.time()-t1, 3), "s")



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': pred_Lasso_x.flatten()})

print(error_airbnb.head().to_string())
# Train ElasticNet Regressor model

model_enet_x = ElasticNet(alpha = 0.01, normalize=False)

t0=time.time()

model_enet_x.fit(X_train,y_train) 

print("Training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

t1=time.time()

pred_enet_x= model_enet_x.predict(X_test)

print("Prediction time:", round(time.time()-t1, 3), "s")



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': pred_enet_x.flatten()})

print(error_airbnb.head().to_string())
# Train Random Forest Regression model

randomf = RandomForestRegressor(n_estimators=200)

t0=time.time()

randomf.fit(X_train,y_train)

print("Training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

t1=time.time()

pred_random_x = randomf.predict(X_test)

print("Prediction time:", round(time.time()-t1, 3), "s")



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': pred_random_x.flatten()})

print(error_airbnb.head().to_string())
X_train = X_train.loc[:,~X_train.columns.duplicated()]

X_test = X_test.loc[:,~X_test.columns.duplicated()]
# Train Xgboost Regressor model

xgb = xgboost.XGBRegressor(n_estimators=310,learning_rate=0.1,objective='reg:squarederror')

t0=time.time()

xgb.fit(X_train, y_train)

print("Training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

t1=time.time()

pred_xgb_x = xgb.predict(X_test)

print("Prediction time:", round(time.time()-t1, 3), "s")



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': pred_xgb_x.flatten()})

print(error_airbnb.head().to_string())
# Train Gradient Boosted Regressor model

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)

t0=time.time()

GBoost.fit(X_train,y_train)

print("Training time:", round(time.time()-t0, 3), "s") # the time would be round to 3 decimal in seconds

t1=time.time()

pred_gboost_x = GBoost.predict(X_test)

print("Prediction time:", round(time.time()-t1, 3), "s")



error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': pred_gboost_x.flatten()})

print(error_airbnb.head().to_string())
print('-------------Linear Regression-----------')



print('MAE: %f'% mean_absolute_error(y_test, pred_linear_x))

print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, pred_linear_x)))   

print('R2 %f' % r2_score(y_test, pred_linear_x))



print('---------------Ridge Regression---------------------')



print('MAE: %f'% mean_absolute_error(y_test, pred_ridge_x))

print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, pred_ridge_x)))   

print('R2 %f' % r2_score(y_test, pred_ridge_x))



print('---------------Lasso Regression-----------------------')



print('MAE: %f' % mean_absolute_error(y_test, pred_Lasso_x))

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test, pred_Lasso_x)))

print('R2 %f' % r2_score(y_test, pred_Lasso_x))



print('---------------ElasticNet Regressor-------------------')



print('MAE: %f' % mean_absolute_error(y_test,pred_enet_x))

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,pred_enet_x)))

print('R2 %f' % r2_score(y_test, pred_enet_x))



print('---------------RandomForest Regressor-------------------')



print('MAE: %f' % mean_absolute_error(y_test,pred_random_x))

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,pred_random_x)))

print('R2 %f' % r2_score(y_test, pred_random_x))



print('---------------XG Boost Regressor-------------------')



print('MAE: %f' % mean_absolute_error(y_test,pred_xgb_x))

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,pred_xgb_x)))

print('R2 %f' % r2_score(y_test, pred_xgb_x))



print('---------------GradientBoosted Regressor-------------------')



print('MAE: %f' % mean_absolute_error(y_test,pred_gboost_x))

print('RMSE: %f' % np.sqrt(mean_squared_error(y_test,pred_gboost_x)))

print('R2 %f' % r2_score(y_test, pred_gboost_x))