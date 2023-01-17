import pandas as pd

import time

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline 

from sklearn import preprocessing

import string
pd.set_option('display.max_columns', 300)
pd.set_option('max_colwidth',400)
pd.set_option('display.max_rows', 300)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>")) # using this in your offline notebooks, will get rid of most of the side space
display(HTML("<style>table {float:left}</style>")) # makes the changelog table nicer
train = pd.read_csv('../input/avito-demand-prediction/train.csv')
test = pd.read_csv('../input/avito-demand-prediction/test.csv')
full = pd.concat([train, test], axis=0)
full.info(memory_usage="deep")
%%time
full["city_region"] = full.loc[:, ["city", "region"]].apply(lambda l: " ".join(l), axis=1)
full.sample(5)
print(full.shape)
city_region_unique = full.drop_duplicates(subset="city_region").reset_index(drop=True)
print(city_region_unique.shape)
from geopy import geocoders

# api_key = "" # place your API key here if you want to access the API as often as you like.
# g = geocoders.GoogleV3(api_key=api_key)

g = geocoders.GoogleV3()
# geocode = g.geocode("Самара Самарская область", timeout=10, language="en")
# geocode.raw
# print(geocode.address)
# print(geocode.latitude)
# print(geocode.longitude)
city_region_unique = pd.read_csv("../input/avito-russian-region-cities/avito_region_city_features.csv")
print(city_region_unique.shape)
# city_region_unique["latitude"] = np.nan
# %%time
# print("searching", len(city_region_unique), "entries")

# for index, row in city_region_unique.loc[city_region_unique["latitude"].isnull(), 
#                                          :].iterrows():
    
#     search = city_region_unique.loc[index, "city_region"]
#     try:
#         geocode = g.geocode(search, timeout=10, language="en")

#         city_region_unique.loc[index, 'latitude'] = geocode.latitude
#         city_region_unique.loc[index, 'longitude'] = geocode.longitude
#     except:
#         city_region_unique.loc[index, 'latitude'] = -999
#         city_region_unique.loc[index, 'longitude'] = -999
        
#     time.sleep(.1)
    
#     if index%10==0:
#         print(str(index).ljust(6), end=" ")
#     if (index+1)%120==0:
#         print("")

# print("\n done")
# The library in the Kaggle kernel seems to be miscompiled
# import hdbscan
# city_region_unique["lat_lon_hdbscan_cluster_05_03"] = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3
#                                                                      ).fit_predict(city_region_unique.loc[:, ["latitude", "longitude"]])
print("number of clusters:", len(city_region_unique["lat_lon_hdbscan_cluster_05_03"].unique()))
sns.lmplot(data=city_region_unique, x="longitude", y="latitude", hue="lat_lon_hdbscan_cluster_05_03", 
           size=10, legend=False, fit_reg=False)
# city_region_unique["lat_lon_hdbscan_cluster_10_03"] = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3
#                                                                      ).fit_predict(city_region_unique.loc[:, ["latitude", "longitude"]])
print("number of clusters:", len(city_region_unique["lat_lon_hdbscan_cluster_10_03"].unique()))
sns.lmplot(data=city_region_unique, x="longitude", y="latitude", hue="lat_lon_hdbscan_cluster_10_03", 
           size=10, legend=False, fit_reg=False)
# city_region_unique["lat_lon_hdbscan_cluster_20_03"] = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=3
#                                                                      ).fit_predict(city_region_unique.loc[:, ["latitude", "longitude"]])
print("number of clusters:", len(city_region_unique["lat_lon_hdbscan_cluster_20_03"].unique()))
sns.lmplot(data=city_region_unique, x="longitude", y="latitude", hue="lat_lon_hdbscan_cluster_20_03", 
           size=10, legend=False, fit_reg=False)
city_region_unique["region_id"]      = preprocessing.LabelEncoder().fit_transform(city_region_unique["region"].values)
city_region_unique["city_region_id"] = preprocessing.LabelEncoder().fit_transform(city_region_unique["city_region"].values)
city_region_unique.head()
city_region_unique.to_csv("avito_region_city_features.csv", index=False)
print("before:", full.shape)
full = pd.merge(left=full, right=city_region_unique, how="left", on=["region", "city"])
print("after :", full.shape)
full.loc[:, ["item_id", "user_id", "region", "city", "latitude", "longitude", "lat_lon_hdbscan_cluster_05_03"]].sample(5)