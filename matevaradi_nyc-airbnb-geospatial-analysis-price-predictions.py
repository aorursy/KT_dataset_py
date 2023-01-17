# Basics:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Geospatial analysis:

import geopandas as gpd 

from geopandas.tools import geocode

import folium # interactive maps

from folium import Circle, Marker,GeoJson,Icon

from folium.plugins import HeatMap, MarkerCluster

import math

from geopy.geocoders import Nominatim

from geopy import distance



# Linear regression:

import statsmodels.api as sm

# Predictions:

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

# Defining colors for the visualizations that match Airbnb's design

color1="#FF5A60"

color2="#565a5d"

color3="#cfd1cc"

color4="#007a87"



# Setting style for visualizations",

sns.set_style("darkgrid")
# Loading data

nyc_data = pd.read_csv("../input/AB_NYC_2019.csv")

nyc_data.head()
print(nyc_data.isnull().sum(axis=0))
cols_without_nans=list(nyc_data.columns[~nyc_data.isna().any()])

# Only keeping columns with no NaNs and reviews_per_month:

nyc_data=nyc_data[cols_without_nans+["reviews_per_month"]]



# Impute the mean for the reviews_per_month column:

nyc_data=nyc_data.fillna(nyc_data.mean())



print("\n")

print("Any NaNs after cleaning?:",nyc_data.isnull().any().any())

nyc_data.describe()
f, axes = plt.subplots(1, 2,figsize=(12,5))

sns.distplot(nyc_data["price"],color=color2,ax=axes[0]).set(title="Price distribution")

sns.distplot(nyc_data[nyc_data["price"]<2000]["price"],color=color1,ax=axes[1]).set(

    title="Price distribution of Airbnb listings below 2000 $ per night")
f, axes = plt.subplots(1, 2,figsize=(12,5))

sns.distplot(nyc_data["number_of_reviews"],color=color1,ax=axes[0]).set(

    title="Distribution of the number of reviews per listing",xlabel="")

sns.distplot(nyc_data.dropna()["reviews_per_month"],color=color2,ax=axes[1]).set(

    title="Distribution of the number of reviews per listing per month",xlabel="")
f, axes = plt.subplots(1, 2,figsize=(12,5))

sns.countplot(y="room_type",data=nyc_data,orient="h",

                color=color1,ax=axes[0]).set(title='Number of listing per room type', ylabel='')



sns.barplot(y=list(nyc_data.groupby("room_type").mean()["price"].index),

            x=list(nyc_data.groupby("room_type").mean()["price"].values),

            ax=axes[1],color=color2,orient="h").set(title='Price per room type')



plt.tight_layout()
f, axes = plt.subplots(1, 2,figsize=(12,5))

sns.countplot(y="neighbourhood_group",data=nyc_data,orient="h",

                color=color1,ax=axes[0]).set(title='Boroughs', ylabel='')



most_frequent_neighborhoods=nyc_data["neighbourhood"].value_counts()[0:10].index.tolist()

print("The most neighborhoods with the most Airbnb listings are:  \n")

for neighborhood in most_frequent_neighborhoods:

    print(neighborhood)



sns.countplot(y="neighbourhood",data=nyc_data[nyc_data["neighbourhood"].isin(most_frequent_neighborhoods)]

            ,orient="h",color=color1,ax=axes[1]).set(title='Neighbourhoods', ylabel='')

plt.tight_layout()
# Defining a color palette

palette ={"Private room": color2, "Entire home/apt": color1, "Shared room": color3}

sns.barplot(x='neighbourhood_group', y='price', hue='room_type',data=nyc_data,palette=palette).set(

    ylabel="Price",xlabel="Borough",title="Average price in the five boroughs according to room type")
TimesSquare=[40.758896,-73.985130] #Latitude and longitude of Times Square



# Creating a map with Time Square as its starting location:

m_1 = folium.Map(location=TimesSquare, tiles='cartodbpositron', zoom_start=12)



# Add markers for Airbnb listings:

mc = MarkerCluster()

for idx, row in nyc_data.iterrows():

    if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):

        mc.add_child(Marker([row['latitude'], row['longitude']]))

m_1.add_child(mc)



# Display the map

m_1
m_2 = folium.Map(location=TimesSquare, tiles='cartodbpositron', zoom_start=11)



# Add a heatmap to the base map

HeatMap(data=nyc_data[['latitude', 'longitude']], radius=10).add_to(m_2)



# Display the map

m_2
# Get the names of top tourist attractions (according to Google)

top_sights=["Empire State Building","Central Park","Times Square","Brooklyn Bridge",

            "The Metropolitan Museum of Art","Museum of Modern Art","Rockefeller Center",

            "The High Line","Grand Central Terminal","One World Trade Center"]



# This object can be used to return latitude and longitude values for a named location

locator = Nominatim(user_agent="myGeocoder")



# We will store the location data for top sights in the top_sights list

top_sight_locations=[]

for sight in top_sights:

    location=locator.geocode(sight)

    top_sight_locations.append((location.latitude,location.longitude))

    

# Let's check if we all locations are correct (i.e. are in New York City) by displaying them on a map

m_3 = folium.Map(location=TimesSquare, tiles='cartodbpositron', zoom_start=12)



for i in range(len(top_sight_locations)):

    loc=top_sight_locations[i]

    Marker([loc[0], loc[1]],popup=top_sights[i],icon=Icon(color="gray")).add_to(m_3)



# Display the map

m_3
def get_distance(loc_from,loc_to):

    """

    Returns the distance between two locations

    Input: 

        loc_from: tuple or list of latitude and longitude coordinates

        loc_to:   tuple or list of latitude and longitude coordinates

        

    Output: 

        dis:      distance (in kilometers)

    """

    dis=distance.distance(loc_from,loc_to).km   

    return dis
# Example - the second row entry of the database is a "Skylit Midtown Castle":

SkylitMidtownCastle=nyc_data.iloc[1]

SkylitMidtownCastle_distances = [get_distance(

                        (SkylitMidtownCastle.latitude,SkylitMidtownCastle.longitude),

                        (sight[0],sight[1])) 

                        for sight in top_sight_locations]

print("Distance from: \n")

for i in range(len(top_sights)):

    print(top_sights[i],": ",round(SkylitMidtownCastle_distances[i],2),"kilometers")
# Create the map

m_4 = folium.Map(location=TimesSquare, tiles='cartodbpositron', zoom_start=12)



# Add the Airbnb listing

Marker([SkylitMidtownCastle.latitude, SkylitMidtownCastle.longitude],

       popup="Skylit Midtown Castle",icon=Icon(color=color1)).add_to(m_4)



# Add top sights and lines between them and the Skylit Midtown Castle

for i in range(len(top_sight_locations)):

    loc=top_sight_locations[i]

    Marker([loc[0], loc[1]],popup=top_sights[i],icon=Icon(color="gray")).add_to(m_4)

    line_from_SkylitMidtownCastle=folium.PolyLine(locations=[

        [SkylitMidtownCastle.latitude, SkylitMidtownCastle.longitude],[loc[0], loc[1]]],

                                                  weight=2,color=color1,

                                                  tooltip=((str(round(SkylitMidtownCastle_distances[i],2)))+" km"))

    m_4.add_children(line_from_SkylitMidtownCastle)

    

# Display map

m_4
for i in range(len(top_sights)): # Loop through all the top sights

    column_name="dist_from_"+top_sights[i].replace(" ","_").lower()

    # Calculate the distances from each Airbnb listing

    distances=nyc_data.apply(lambda x: get_distance((x["latitude"],x["longitude"]),(top_sight_locations[i])),axis=1)

    # Add distances to our dataframe

    nyc_data[column_name]=distances



print(len(top_sights)," new columns added to the dataset.")
# calculate distance to closest_sight

nyc_data["dist_to_nearest_sight"]=nyc_data[[colname for colname in nyc_data.columns 

                                            if colname.startswith("dist_from")]].apply(min,axis=1)
y=np.log(nyc_data.price+0.0001)

# (The + 0.0001 smoothing is needed because some Airbnbs have a price of 0, which we cannot take the logarithm of)
# Creating a dataframe of the predictors with selected variables

X_1=pd.concat([pd.Series([1]*len(nyc_data),name="(cons)"), # for constant term

    nyc_data[["minimum_nights","reviews_per_month","number_of_reviews","availability_365"]],

# Creating dummies from room_type, shared room is the reference

pd.get_dummies(nyc_data['room_type'])[["Entire home/apt","Private room"]]], 

axis=1)



X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y, test_size=0.3, random_state=24)
lm_1 = sm.OLS(y_train_1,X_train_1).fit()

lm_1.summary()
# Creating a dataframe of the predictors with selected variables

X_2=pd.concat([pd.Series([1]*len(nyc_data),name="(cons)"), # for constant term

    nyc_data[["minimum_nights","reviews_per_month","number_of_reviews","availability_365"]],

# Creating dummies from neighborhood, bronx is the reference

pd.get_dummies(nyc_data['neighbourhood_group'],drop_first=True), 

# Creating dummies from room_type, shared room is the reference

pd.get_dummies(nyc_data['room_type'])[["Entire home/apt","Private room"]]], 

axis=1)



X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y, test_size=0.3, random_state=24)
lm_2 = sm.OLS(y_train_2,X_train_2).fit()

lm_2.summary()
base_vars=["minimum_nights","reviews_per_month","number_of_reviews","availability_365"]

distance_vars=[colname for colname in nyc_data.columns if colname.startswith("dist")]



# Creating a dataframe of the predictors with selected variables

X_3=pd.concat([pd.Series([1]*len(nyc_data),name="(cons)"), # for constant term

    nyc_data[base_vars+distance_vars], 

# Creating dummies from room_type, shared room is the reference

pd.get_dummies(nyc_data['room_type'])[["Entire home/apt","Private room"]]], 

axis=1)



X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y, test_size=0.3, random_state=24)
lm_3 = sm.OLS(y_train_3,X_train_3).fit()

lm_3.summary()
base_vars=["minimum_nights","reviews_per_month","number_of_reviews","availability_365"]

distance_vars=[colname for colname in nyc_data.columns if colname.startswith("dist")]



# Creating a dataframe of the predictors with selected variables

X_4=pd.concat([nyc_data[base_vars+distance_vars], 

# Creating dummies from neighborhood, bronx is the reference

pd.get_dummies(nyc_data['neighbourhood_group'],drop_first=True), 

# Creating dummies from room_type, shared room is the reference

pd.get_dummies(nyc_data['room_type'])[["Entire home/apt","Private room"]]], 

axis=1)



X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_4, y, test_size=0.3, random_state=24)
lm = LinearRegression()

rf = RandomForestRegressor(random_state=24)

gb = GradientBoostingRegressor(max_depth=10)



# Train the models

lm.fit(X_train_4, y_train_4)

rf.fit(X_train_4, y_train_4)

gb.fit(X_train_4, y_train_4)



# Make predictions using the testing set

y_hat_lm = lm.predict(X_test_4)

y_hat_rf = rf.predict(X_test_4)

y_hat_gb = gb.predict(X_test_4)





MSE_lm=mean_squared_error(y_hat_lm,y_test_4)

MSE_rf=mean_squared_error(y_hat_rf,y_test_4)

MSE_gb=mean_squared_error(y_hat_gb,y_test_4)



print("Linear Regression Mean Squared Error:",MSE_lm)

print("Random Forest Regression Mean Squared Error:",MSE_rf)

print("Gradient Boosting Regression Mean Squared Error:",MSE_gb)
# Train the models on the dataset without geospatial variables 

# Creating a dataframe of the predictors with selected variables

X_5=pd.concat([nyc_data[base_vars], 

# Creating dummies from neighborhood, bronx is the reference

pd.get_dummies(nyc_data['neighbourhood_group'],drop_first=True), 

# Creating dummies from room_type, shared room is the reference

pd.get_dummies(nyc_data['room_type'])[["Entire home/apt","Private room"]]], 

axis=1)



X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5, y, test_size=0.3, random_state=24)



lm.fit(X_train_5, y_train_5)

rf.fit(X_train_5, y_train_5)

gb.fit(X_train_5, y_train_5)



print("Improvement for Linear Regression:",

      round((1-MSE_lm/mean_squared_error(lm.predict(X_test_5),y_test_5))*100,2),"%")

print("Improvement for Random Forest Regression:",

      round((1-MSE_lm/mean_squared_error(rf.predict(X_test_5),y_test_5))*100,2),"%")

print("Improvement for Gradient Boosting Regression:",

      round((1-MSE_lm/mean_squared_error(gb.predict(X_test_5),y_test_5))*100,2),"%")