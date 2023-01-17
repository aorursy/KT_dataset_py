import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ab_nyc = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
print(f'Shape of the dataset is {ab_nyc.shape}')
ab_nyc.head()
ab_nyc.info()
ab_nyc.describe()
ab_nyc.isnull().sum()
plt.style.use("fivethirtyeight")

longitude = ab_nyc['longitude']

latitude = ab_nyc['latitude']

neighbourhood_group = ab_nyc['neighbourhood_group']

price = ab_nyc['price']

plt.scatter(longitude,latitude,c=price,cmap="coolwarm",

            edgecolor='black',linewidth=1,alpha=0.65)

cbar = plt.colorbar()

cbar.set_label("Prices")

plt.title("New York City")

plt.xlabel("Longitude")

plt.ylabel("Latitude")



plt.tight_layout()
plt.figure(figsize=(50,30))

fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=2,figsize=(12,10))

ax1[0].hist(ab_nyc['price'],edgecolor='black',bins=50,log=True)

ax1[1].hist(ab_nyc['minimum_nights'],edgecolor='black',bins=50,log=True)

ax2[0].hist(ab_nyc['number_of_reviews'],edgecolor='black',bins=50,log=True)

ax2[1].hist(ab_nyc['availability_365'],edgecolor='black',bins=50)



ax1[0].set_title("Prices")

ax1[0].set_xlabel("Prices")

ax1[0].set_ylabel("Distributions")



ax1[1].set_title("Minimum Nights")

ax1[1].set_xlabel("Minimum Nights")



ax2[0].set_title("Number of Reviews")

ax2[0].set_xlabel("Number of Reviews")

ax2[0].set_ylabel("Distributions")



ax2[1].set_title("Availability 365")

ax2[1].set_xlabel("Availability")



plt.tight_layout()
room_type = ab_nyc['room_type'].value_counts()

neighbourhood_group = ab_nyc['neighbourhood_group'].value_counts()



fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(12,5))



ax1.pie(room_type.values,labels=room_type.index,

       shadow=True,

       autopct='%1.1f%%',

       wedgeprops={'edgecolor':'black'})

ax2.pie(neighbourhood_group.values,labels=neighbourhood_group.index,

       shadow=True,

       autopct='%1.1f%%',

       wedgeprops={'edgecolor':'black'})



ax1.set_title("Room Types")

ax2.set_title("Neighbourhood Groups")

plt.tight_layout()
dataset = ab_nyc.drop(['id','name','host_id','host_name','last_review','reviews_per_month'],axis=1)
dataset.info()
num_of_reviews = pd.qcut(dataset['number_of_reviews'],10,labels=False, duplicates='drop')

minimum_nights = pd.qcut(dataset['minimum_nights'],10,labels=False, duplicates='drop')

availability_365 = np.log10(dataset['availability_365'] + 1)
dataset['number_of_reviews'] = num_of_reviews

dataset['minimum_nights'] = minimum_nights

dataset['availability_365'] = availability_365
fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(12,5))

ax1.hist(num_of_reviews,edgecolor='black',bins=50)

ax2.hist(minimum_nights,edgecolor='black',bins=50)

ax3.hist(availability_365,edgecolor='black',bins=50)



ax1.set_title("Number of Reviews")

ax2.set_title("Minimum Nights")

ax3.set_title("Availablility 365")



plt.tight_layout()
dataset['neighbourhood_group'].value_counts()
dataset['neighbourhood'].value_counts()
neighbourhood = pd.get_dummies(dataset['neighbourhood'],drop_first=True,prefix='ng')

neighbourhood_group = pd.get_dummies(dataset['neighbourhood_group'],drop_first=True,prefix='ng')

room_types = pd.get_dummies(dataset['room_type'],drop_first=True,prefix='rt')

cal_host_lc = pd.get_dummies(dataset['calculated_host_listings_count'],drop_first=True,prefix='chlc')
dataset.drop(['neighbourhood','neighbourhood_group','room_type','calculated_host_listings_count'],axis=1,inplace=True)
dataset = dataset.join([neighbourhood_group,neighbourhood,room_types,cal_host_lc])
print(f'Shape of the prepared dataset is {dataset.shape}')
dataset.head()
X = dataset.drop('price',axis=1)

y = dataset['price']



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
std_sclr = StandardScaler()

std_sclr.fit(X_train)

X_train = std_sclr.transform(X_train)

X_test = std_sclr.transform(X_test)
X_train = pd.DataFrame(X_train,columns=X.columns)

X_test = pd.DataFrame(X_test,columns=X.columns)
xgb = XGBRegressor(n_estimators=100,n_jobs=-1)

xgb.fit(X_train,y_train)

predictions = xgb.predict(X_test)
print(f'Mean Squared Error is {mean_squared_error(predictions,y_test)}')

print(f'Mean Absolute Error is {mean_absolute_error(predictions,y_test)}')

print(f'Root Mean Squared Error is {np.sqrt(mean_squared_error(predictions,y_test))}')