# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



### Import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neighbors import KNeighborsRegressor

import folium

from folium import plugins







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Convert the datasets to a PandasDataframe

listings = pd.read_csv("/kaggle/input/seattle/listings.csv")







print(listings.shape)

print(listings.info())



neigh_price = listings[["neighbourhood","price"]]

listings["price"] = listings["price"].str.replace(',', '')



for col in listings:

    listings["price"] = listings["price"].map(lambda x: x.replace('$',''))
listings['price'] = listings['price'].astype('float')
'''

# pearson's correlation feature selection for numeric input and numeric output

from sklearn.datasets import make_regression

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

# generate dataset

X, y = make_regression(n_samples=100, n_features=100, n_informative=10)

# define feature selection

fs = SelectKBest(score_func=f_regression, k=10)

# apply feature selection

X_selected = fs.fit_transform(X, y)

print(X_selected.shape)

'''
# Cleaning the Data 
listings_num = listings.select_dtypes(include = ['float64', 'int64'])

listings_num.head()


pkmn_type_colors = ['#78C850',  # Grass

                    '#F08030',  # Fire

                    '#6890F0',  # Water

                    '#A8B820',  # Bug

                    '#A8A878',  # Normal

                    '#A040A0',  # Poison

                    '#F8D030',  # Electric

                    '#E0C068',  # Ground

                    '#EE99AC',  # Fairy

                    '#C03028',  # Fighting

                    '#F85888',  # Psychic

                    '#B8A038',  # Rock

                    '#705898',  # Ghost

                    '#98D8D8',  # Ice

                    '#7038F8',  # Dragon

                   ]
neigh_price_grp = listings.groupby("neighbourhood")["price"].mean()

ok = pd.DataFrame(neigh_price_grp).sort_values('price', ascending=False)

#ok.head(20).plot.bar(figsize=(20,10),fontsize = 20,color="g", )



#ok.tail(20).plot.bar(figsize=(20,10),fontsize = 20,color="r")



print(ok.head(20))

print(ok.tail(20))
# Higher houses price neighbourhood

ok.head(20).plot.bar(figsize=(20,10),fontsize = 20,color="g", )
# Cheapest houses price neighbourhood

ok.tail(20).plot.bar(figsize=(20,10),fontsize = 20,color="r")
# Create maps for the two regions
top20 = ok.head(20)

flop20 = ok.tail(20)



top20_loc =  pd.merge(top20, listings, how='right').fillna(0)

flop20_loc = pd.merge(flop20, listings, how='right').fillna(0)





top20_locc = top20_loc[["neighbourhood","price","latitude","longitude","street"]]

flop20_locc = flop20_loc[["neighbourhood","price","latitude","longitude","street"]]



top20_comp = top20_locc.head(20)

flop20_comp = top20_locc.tail(20)

m = folium.Map([47.6062, -122.3321], zoom_start=11)

top20_comp_sub = top20_comp.copy()

for i, row in top20_comp_sub .iterrows():

    folium.Circle([row['latitude'], row['longitude']],

                        radius=10,

                        popup=folium.Popup(row['street']),

                        ).add_to(m)



# Map of higher houses price locations

top20_map = top20_comp_sub [['latitude', 'longitude']].values

# plot heatmap

m.add_child(plugins.HeatMap(top20_map, radius=20))

m
flop20_comp_sub = flop20_comp.copy()

for i, row in flop20_comp_sub .iterrows():

    folium.Circle([row['latitude'], row['longitude']],

                        radius=10,

                        popup=folium.Popup(row['street']),

                        ).add_to(m)
# Map of higher houses price locations

flop20_map = flop20_comp_sub [['latitude', 'longitude']].values

# plot heatmap

m.add_child(plugins.HeatMap(flop20_map, radius=20))

m
# Different properties types prices

property_type_price = listings.groupby("property_type")["price"].mean()



prtype= pd.DataFrame(property_type_price).drop(["Other"])

print(prtype)

prtype.sort_values('price', ascending=False).plot.bar(figsize=(20,10),fontsize = 20,color="teal")



# EDA of differente features of house prices
grouped_property_type = listings.groupby('property_type').agg({'price': ['mean']}).reset_index().set_index('property_type')

proptype=grouped_property_type

proptype
'''

"ax = prtype.plot.bar(rot=10,figsize=(20,10))"



prtype.plot(kind='barh',y='property_type',x='price',color='r')

"ax.legend(fontsize = 20)"

axes = proptype.plot.bar(rot=1, subplots=True)

axes[1].legend(loc=2)

'''
grouped_city = listings.groupby('city').agg({'price': ['mean', 'min', 'max']})

grouped_city
grouped_number_of_reviews = listings.groupby('number_of_reviews').agg({'price': ['mean', 'min', 'max']})

grouped_number_of_reviews

listings_num.hist(figsize=(16, 20), bins=20, xlabelsize=8, ylabelsize=8);
listings_corr = listings.corr()['price']# -1 because the latest row is SalePrice

golden_features_list = listings_corr[abs(listings_corr) > 0.5].sort_values(ascending=False)

print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
# Heatmap of houses features:

import seaborn as sns

plt.figure(figsize=(50,50))

cor = listings.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
corrr = listings_num.corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corrr[(corrr >= 0.5) | (corrr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
cor["price"]
# Heatmap of houses features that have 0.5 correlation or more with the price 



kot = cor[cor>=.5]

plt.figure(figsize=(20,20))

sns.heatmap(kot, cmap="Blues")

important_listings1 = listings[['accommodates','bathrooms','bedrooms','beds','price']]

print(important_listings1.shape)

print(important_listings1.info())

print(important_listings1.describe())



important_listings2 = listings[['accommodates', 'bedrooms', 'bathrooms','minimum_nights','maximum_nights','number_of_reviews','price']]

print(important_listings2.shape)

print(important_listings2.info())

print(important_listings2.describe())
important_listings1.head()

'''

#Using Pearson Correlation

import seaborn as sns

plt.figure(figsize=(12,10))

cor = important_listings.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()

'''
# cleaning the "important_listings1"
null = important_listings1.isnull().sum()

null
null = important_listings2.isnull().sum()

null
important_listings1.dropna(axis=0,how='any',inplace=True)

important_listings2.dropna(axis=0,how='any',inplace=True)

important_listings1.isnull().sum().sum()

important_listings1.shape
important_listings1.head()
'''

dummies = pd.get_dummies(important_listings['property_type']).rename(columns=lambda x: 'property_type' + str(x))

important_listings_new = pd.concat([important_listings, dummies], axis=1)



important_listings_new.shape

'''
'''

important_listings_new = important_listings_new.loc[:,~important_listings_new.columns.duplicated()]

important_listings_new

'''

'''

data = important_listings_new.drop(['property_type'], axis=1)

data.shape

'''
'''

important_listings = important_listings.loc[:,~important_listings.columns.duplicated()]

important_listings

'''
#Using Pearson Correlation

# Heatmap of the price and features correlation

import seaborn as sns

plt.figure(figsize=(12,10))

cor = important_listings1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
cor = important_listings1.corr().drop(['price'])

cor_df = pd.DataFrame(cor["price"]).sort_values('price', ascending=False).plot.bar(figsize=(20,10),fontsize = 20,color="g")

cor_df
#Using Pearson Correlation

import seaborn as sns

plt.figure(figsize=(12,10))

cor = important_listings2.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#cols= important_listings[["accommodates","bathrooms","bedrooms","price","minimum_nights","maximum_nights","number_of_reviews"]]
'''

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

data2 = scaler.fit(important_listings)

data2

'''
# Split the data 

train_df = important_listings1.iloc[0:2847]

test_df = important_listings1.iloc[2847:]

train_df
train_df2 = important_listings2.iloc[0:2847]

test_df2 = important_listings2.iloc[2847:]

train_df2
knn=KNeighborsRegressor(n_neighbors=5,algorithm="brute")



# Predict houses prices using two features
two_features=["accommodates","bathrooms"]



knn.fit(train_df[two_features],train_df["price"])





predictions=knn.predict(test_df[two_features])



from sklearn.metrics import mean_squared_error





two_features_mse=mean_squared_error(test_df["price"],predictions)

two_features_rmse=two_features_mse**(1/2)



print(two_features_mse)

print(two_features_rmse)
# Predict houses prices using three features


three_features = ['accommodates', 'bedrooms', 'bathrooms']

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')



knn.fit(train_df[three_features],train_df["price"])

three_predictions=knn.predict(test_df[three_features])



four_mse=mean_squared_error(test_df["price"],three_predictions)



four_rmse=four_mse**(1/2)



print(four_mse)

print(four_rmse)
# Predict houses prices using the features that have the highest correlations with the price


# First prediction important_listings1

features = ['accommodates', 'bedrooms', 'bathrooms']

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')



knn.fit(train_df[features],train_df["price"])

four_predictions=knn.predict(test_df[features])



mse=mean_squared_error(test_df["price"],four_predictions)



rmse=mse**(1/2)



print(mse)

print(rmse)
# Predict houses prices using the features that we have selected 
# Second prediction important_listings2

features2 = ['accommodates', 'bedrooms', 'bathrooms','minimum_nights','maximum_nights','number_of_reviews']

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')



knn.fit(train_df2[features2],train_df2["price"])

four_predictions=knn.predict(test_df2[features2])



mse2=mean_squared_error(test_df2["price"],four_predictions)



rmse2=mse2**(1/2)



print(mse2)

print(rmse2)
# Comparing the two results



mse12 = [mse,mse2]

rmse12 = [rmse,rmse2]

rmse_ovrl=[rmse,rmse2]

evaluation = pd.DataFrame(

    {"mse12":mse12,

     "rmse12":rmse12

    })

    

    





evaluation
