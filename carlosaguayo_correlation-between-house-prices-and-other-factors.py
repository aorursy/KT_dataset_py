import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
hospitals = pd.read_csv("../input/usa-hospitals/Hospitals.csv")

hospital_ratings = pd.read_csv("../input/hospital-ratings/Hospital General Information.csv", encoding="ISO-8859-1")

public_schools = pd.read_csv("../input/usa-public-schools/Public_Schools.csv")

county_time_series = pd.read_csv("../input/zecon/County_time_series.csv")

crosswalk = pd.read_csv("../input/zecon/CountyCrossWalk_Zillow.csv")

unemployment = pd.read_csv("../input/2018-unemployment-rate-by-county/GeoFRED_Unemployment_Rate_by_County_Percent.csv")
unemployment.shape
unemployment.head()
# Unemployment rate in Fairfax, VA in 2018

fairfax_unemployment = unemployment.loc[unemployment['Region Code'] == 51059].sort_values(by=['Region Name'])

print ("The unemployment rate in 2018 for Fairfax, VA is {0}".format(fairfax_unemployment.iloc[:,-1].values[0]))



fairfax_unemployment
hospitals.shape
hospitals.head()
hospital_ratings.shape
hospital_ratings.head()
hospital_ratings = hospital_ratings[['Hospital Name', 'Hospital overall rating']]

hospital_ratings.dropna()

hospital_ratings.head()
hospitals_with_ratings = pd.merge(hospitals, hospital_ratings, how='left', left_on='NAME', right_on='Hospital Name')



hospitals_with_ratings = hospitals_with_ratings.loc[hospitals_with_ratings['COUNTYFIPS'] != 'NOT AVAILABLE']



hospitals_with_ratings = hospitals_with_ratings.loc[~pd.isna(hospitals_with_ratings)['Hospital overall rating']]

hospitals_with_ratings = hospitals_with_ratings.loc[hospitals_with_ratings['Hospital overall rating'] != 'Not Available']

hospitals_with_ratings['Hospital overall rating'] = hospitals_with_ratings['Hospital overall rating'].astype("int64")



hospitals_with_ratings.rename(index=str, columns={"COUNTYFIPS": "FIPS", "Hospital overall rating": "AverageHospitalRating"}, inplace=True)



hospitals_with_ratings.head()
# Hospitals in Fairfax, VA

fairfax_hospitals = hospitals_with_ratings.loc[hospitals_with_ratings['FIPS'] == "51059"].sort_values(by=['NAME'])

print ("There are {0} hospitals in Fairfax, VA".format(fairfax_hospitals.shape[0]))



fairfax_hospitals
public_schools.shape
public_schools.head()
# Public schools for Fairfax, VA

# fairfax_schools = public_schools.loc[(public_schools['COUNTYFIPS'] == 51059) & (public_schools['CITY'] == 'FAIRFAX')].sort_values(by=['NAME'])

fairfax_schools = public_schools.loc[public_schools['COUNTYFIPS'] == 51059].sort_values(by=['NAME'])

print ("There are {0} public schools in Fairfax, VA".format(fairfax_schools.shape[0]))



fairfax_schools
# 2018 unemployment per county

unemployment_per_county = unemployment.loc[:,("Region Code", "2018")]

unemployment_per_county.rename(index=str, columns={"Region Code": "FIPS", "2018": "UnemploymentRate"}, inplace=True)

unemployment_per_county["FIPS"] = unemployment_per_county["FIPS"].astype("int64")



print ("The unemployment rate for Fairfax, VA is {0}".format(unemployment_per_county.loc[unemployment_per_county['FIPS']==51059]["UnemploymentRate"][0]))

unemployment_per_county.head()
# Number of Public schools per county

public_schools_per_county = public_schools['COUNTYFIPS'].value_counts().to_frame()

public_schools_per_county.reset_index(level=0, inplace=True)

public_schools_per_county.rename(index=str, columns={"index": "FIPS", "COUNTYFIPS": "NumberOfSchools"}, inplace=True)

public_schools_per_county["FIPS"] = public_schools_per_county["FIPS"].astype("int64")



print ("There are {0} public schools in Fairfax, VA".format(public_schools_per_county.loc[public_schools_per_county['FIPS']==51059]["NumberOfSchools"][0]))

public_schools_per_county.head()
# Compute average hospital rating per county

average_hospital_rating_per_county = hospitals_with_ratings[['FIPS', 'AverageHospitalRating']]



average_hospital_rating_per_county = average_hospital_rating_per_county.dropna()



average_hospital_rating_per_county = average_hospital_rating_per_county.groupby(['FIPS']).mean()

average_hospital_rating_per_county.reset_index(level=0, inplace=True)

average_hospital_rating_per_county["FIPS"] = average_hospital_rating_per_county["FIPS"].astype("int64")



average_hospital_rating_per_county.head()
# Number of Hospitals per county

hospitals_per_county = hospitals['COUNTYFIPS'].value_counts().to_frame()

hospitals_per_county.reset_index(level=0, inplace=True)

hospitals_per_county.rename(index=str, columns={"index": "FIPS", "COUNTYFIPS": "NumberOfHospitals"}, inplace=True)

hospitals_per_county.dropna(inplace=True)

hospitals_per_county= hospitals_per_county[hospitals_per_county["FIPS"] != "NOT AVAILABLE"]

hospitals_per_county["FIPS"] = hospitals_per_county["FIPS"].astype("int64")



print ("There are {0} hospitals in Fairfax, VA".format(hospitals_per_county.loc[hospitals_per_county['FIPS']==51059]["NumberOfHospitals"][0]))

hospitals_per_county.head()
hospitals_per_county = hospitals_per_county.merge(average_hospital_rating_per_county, on="FIPS", how="left")

hospitals_per_county.head()
county_time_series.head()
# Average price for houses in Fairfax, VA

house_prices = county_time_series.groupby("RegionName").mean()

average_price_for_fairfax = house_prices["ZHVI_AllHomes"][51059]

print("The average price for a house in Fairfax, VA is ${:,.2f}".format(average_price_for_fairfax))
crosswalk = crosswalk[['FIPS', 'CountyName', 'StateName']]

crosswalk["FIPS"] = crosswalk["FIPS"].astype("int64")

crosswalk.head()
# Create Team EST dataset

team_est = house_prices["ZHVI_AllHomes"].to_frame()

team_est.reset_index(level=0, inplace=True)

team_est.rename(index=str, columns={"RegionName": "FIPS", "ZHVI_AllHomes": "AverageHousePrice"}, inplace=True)

team_est["FIPS"] = team_est["FIPS"].astype("int64")

team_est.dropna(inplace=True)



team_est.head()
team_est = team_est.merge(crosswalk, on="FIPS")

team_est = team_est.merge(unemployment_per_county, on="FIPS")

team_est = team_est.merge(public_schools_per_county, on="FIPS")

team_est = team_est.merge(hospitals_per_county, on="FIPS")

team_est = team_est[['FIPS', 'CountyName', 'StateName', 'NumberOfSchools', 'NumberOfHospitals', 'AverageHospitalRating', 'UnemploymentRate', 'AverageHousePrice']]

team_est.head()
corrmat = team_est.corr()

plt.subplots(figsize=(6, 6))

sns.heatmap(corrmat.abs(), vmax=.4, square=True)



# corrmat = team_est.corr()

# plt.subplots(figsize=(6,1))

# sns.heatmap(corrmat.abs().values[np.newaxis,-1], vmax=.4)



# corrmat = team_est.corr()

# plt.subplots(figsize=(6,1))

# sns.heatmap(corrmat.abs()["AverageHousePrice"][:,np.newaxis], vmax=.4)
print ("The correlation between a house price and the number of hospitals is: {0}".format(corrmat["AverageHousePrice"]["NumberOfHospitals"]))

print ("The correlation between a house price and the average hospital rating is: {0}".format(corrmat["AverageHousePrice"]["AverageHospitalRating"]))

print ("The correlation between a house price and the number of schools is: {0}".format(corrmat["AverageHousePrice"]["NumberOfSchools"]))

print ("The correlation between a house price and the unemployment rate is: {0}".format(corrmat["AverageHousePrice"]["UnemploymentRate"]))
team_est.to_csv("team_est.csv")
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



xgboost = XGBRegressor(objective ='reg:linear', 

                       colsample_bytree = 0.3, 

                       learning_rate = 0.1,

                       max_depth = 5, 

                       alpha = 10, 

                       random_state=777,

                       n_estimators = 100)



dataset = team_est.values

dataset = dataset[:, 3:]



x, y = dataset[:,:-1], dataset[:, -1]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=777)



xgboost.fit(x_train, y_train)



y_prediction = xgboost.predict(x_test)



r2_score(y_test, y_prediction)
from xgboost import plot_tree

from xgboost import to_graphviz

import matplotlib.pyplot as plt



plot_tree(xgboost)

plt.show()



for i in range(100):

    dot = to_graphviz(xgboost, num_trees=i)

    dot.render("trees{0}".format(i))
print("These are the importance of each feature: {0}".format(xgboost.feature_importances_))
from xgboost import plot_importance

plot_importance(xgboost)

plt.show()
# Generate a few examples

print ("These are the first five entries")

print (x_test[:5])

print ("These are the model prediction for these entries")

print (xgboost.predict(x_test[:5]))

print ("These are the actual prices")

print (y_test[:5])