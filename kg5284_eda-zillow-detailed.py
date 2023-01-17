import sys

print(sys.version)
import numpy as np 

import pandas as pd 

import missingno as msno

from matplotlib import pyplot as plt

import seaborn as sns

from subprocess import check_output
P16_df = pd.read_csv("../input/properties_2016.csv")

train16_df = pd.read_csv("../input/train_2016_v2.csv")


###Data analysis#####################################################

P16_df.head()

col_df = P16_df.columns

#print(col_df)



cnt= P16_df.describe()

#print(cnt)



###to get the cnt of nulls in each column

missing_df = P16_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'P16_missing_count']

#print(missing_df)





###merging 2016 properties and training data of Zillow################

train_df = pd.merge(train16_df, P16_df, on='parcelid', how='left')

#train_df = pd.DataFrame(train_df)

missing_train_df = train_df.isnull().sum(axis=0).sort_values(ascending=True).reset_index()

missing_train_df.columns = ['Column_name','missingcnt_after_merge']

###Visualizing data###################################################



missing_p16_values = P16_df.columns[P16_df.isnull().any()].tolist()

msno.bar(P16_df[missing_p16_values],figsize=(20,8),color='Red',fontsize=12,labels=True,)



fig, ax = plt.subplots(figsize=(12, 8))

sns.barplot(x="missingcnt_after_merge", y="Column_name", data=missing_train_df, color='Sienna', ax=ax)



f,ax = plt.subplots(figsize=(18, 20))

sns.heatmap(train_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)



plt.show()



###Testing Visualization########################



#missing values in Properties_2016 data file

#msno.bar(P16_df[missing_p16_values],figsize=(20,8),color='Red',fontsize=12,labels=True,)



##missing values after merging Properties_2016 and train_2016 data

#fig, ax = plt.subplots(figsize=(12, 25))

#sns.barplot(x="missingcnt_after_merge", y="Column_name", data=missing_train_df, color='Sienna', ax=ax)



#missingValueColumns = train_df.columns[train_df.isnull().any()].tolist()

#msno.bar(train_df[missingValueColumns],figsize=(20,8),color="#34495e",fontsize=12,labels=True)



####Testing pairplot

#g = sns.pairplot(train_df,vars= ["roomcnt","fips","yearbuilt","taxamount"],hue='logerror' ,palette='tab20',size=6)

#g.set(xticklabels=[])



####Testing squarify

#tree=train_df['zipcode'].value_counts().to_frame()

#squarify.plot(sizes=tree['zipcode'].values,label=tree.index,color=sns.color_palette('RdYlGn_r',52))

#plt.rcParams.update({'font.size':20})

#fig=plt.gcf()

#fig.set_size_inches(40,15)

#plt.show()

####testing pairplot

#sns.set()

#cols = ['taxamount','calculatedfinishedsquarefeet' , 'garagecarcnt', 'basementsqft', 'bathroomcnt', 'yearbuilt']

#cols = ['taxamount','calculatedfinishedsquarefeet' , 'garagecarcnt', 'basementsqft', 'bathroomcnt', 'yearbuilt']

#sns.pairplot(train_df[cols], size = 2.5)

data = pd.concat([train_df['taxamount'], train_df['calculatedfinishedsquarefeet']], axis=1)

data.plot.scatter(x='calculatedfinishedsquarefeet', y='taxamount', xlim = (0,10000),ylim=(0,200000));



data = pd.concat([train_df['structuretaxvaluedollarcnt'], train_df['calculatedfinishedsquarefeet']], axis=1)

data.plot.scatter(x='calculatedfinishedsquarefeet', y='structuretaxvaluedollarcnt', xlim = (0,8000),ylim=(0,800000));



data = pd.concat([train_df['landtaxvaluedollarcnt'], train_df['propertylandusetypeid']], axis=1)

data.plot.scatter(x='propertylandusetypeid', y='landtaxvaluedollarcnt', xlim = (0,1000),ylim=(0,800000));



#print(len(train_df['propertylandusetypeid'].unique()))

###Testing for interactive map

import folium

new_df = train_df.loc[train_df['regionidcity']==5534.0]

#print(new_df.iloc[3])

import folium

lat = 40.767937

long = -73.982155

#for i in new_df.index :

map_1 = folium.Map(location=([lat,long]), zoom_start=10) 

folium.Marker([lat,long]).add_to(map_1)

map_1
map_1 = folium.Map(location=[45.5236, -122.6750],

                   tiles='Stamen Toner',

                   zoom_start=13)
new_df = train_df.loc[train_df['regionidcity']==5534.0]

print(new_df.iloc())

new_df.reset_index()

for i in range(new_df.shape[0]):

        long = new_df['longitude'].values[0]

        lat = new_df['latitude'].values[0] 

        folium.Marker([lat, long]).add_to(map_1)

        #pop =  train_df[new_df['parcelid'].values[0]]

        #folium.CircleMarker(location=[lat, long], radius=5,color='#F08080',fill_color='#3186cc').add_to(map_1)

#map_1
###Exploratory Data Analysis(EDA)##################################

#train_df.info()



missing_train_df['missing_train_df_ratio'] = missing_train_df['missingcnt_after_merge'] / train_df.shape[0]

print (missing_train_df.loc[missing_train_df['missing_train_df_ratio'] > .99])



#To determine Number of properties in the each city

#train_df['regionidcity']=train_df['regionidcity'].astype(int)

##in order to obtain exact counts one need to take care of nulls in the dataframe train_df

city_count = train_df.groupby('regionidcity').parcelid.count()

print(city_count)

#there are 177 cities.Below is the number of properties per city.
#unique values in each column

for col in train_df.columns:

    print(col, len(train_df[col].unique()))
#replace null values by mean value of the column

mean_values = train_df.mean(axis=0)

train_df_new = train_df.fillna(mean_values, inplace=True)
plt.figure(figsize=(12,8))

sns.distplot(train_df_new['taxamount'].astype(int));

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(train_df_new.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=train_df_new['taxamount'].values, y=train_df_new['logerror'].values, size=10, color='g')

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Tax Amount', fontsize=12)

plt.show()
cols = ['taxamount','calculatedfinishedsquarefeet','garagecarcnt','bathroomcnt','yearbuilt','regionidcounty']

sns.pairplot(train_df_new[cols], size = 2.5)

#testing correlation matrix

train_df_new.corr().values > 0.2

##Need to delete all the features whose null percentage is above 99%

train_df_new.drop(['decktypeid','finishedsquarefeet6','typeconstructiontypeid','architecturalstyletypeid','fireplaceflag','yardbuildingsqft26','storytypeid','basementsqft','finishedsquarefeet13','buildingclasstypeid'],axis = 1,inplace=True)

#heat map with important features

cols = ['taxamount','calculatedfinishedsquarefeet','garagecarcnt','bathroomcnt','yearbuilt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt']

f,ax = plt.subplots(figsize=(7, 7))

sns.heatmap(train_df_new[cols].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()