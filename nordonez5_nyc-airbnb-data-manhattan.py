import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns
airbnb_data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

airbnb_data.head(5)
airbnb_data.isna().sum() # Check for NULL/NA values 

airbnb_data.drop(['host_id', 'host_name', 'last_review', 'reviews_per_month'], axis=1, inplace=True)

airbnb_data.fillna({'Name':'NoName'}, inplace=True)
Manhattan_Listings = airbnb_data.loc[airbnb_data['neighbourhood_group'].isin(['Manhattan'])]

Manhattan_Listings.insert(12, 'rounded_prices', round(Manhattan_Listings['price'],-2)) 

#Manhattan_Listings['rounded_prices'] = str(Manhattan_Listings['rounded_prices'])

plt.subplots(figsize=(20,20))

sns.scatterplot(x='longitude', y='latitude', data=Manhattan_Listings, hue='rounded_prices', 

                palette="ch:r=-.2,d=.3_r", sizes = 4)
plt.subplots(figsize=(18,10))

sns.distplot(Manhattan_Listings[Manhattan_Listings.price<800].price)

plt.title('Distribution of Prices Between 0 and 800 for Manhattan')
plt.subplots(figsize=(20,20))

sns.kdeplot(Manhattan_Listings.longitude, Manhattan_Listings.latitude, cmap="Greens", shade=True, shade_lowest=False)