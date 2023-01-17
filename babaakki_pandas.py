from __future__ import print_function

import pandas as pd

import matplotlib.pyplot as plt 

import numpy as np
city_names = pd.Series(['New Delhi', 'Mumbai', 'New York', 'London', 'Tokyo', 'Seoul', 'San francisco', 'Sydney', 'Dubai'])

city_names
population_in_M = pd.Series([18.7, 23.6, 22, 8.7, 13.5, 10.3, 8.75, 5.2, 3.38])

population_in_M
cities = pd.DataFrame({'City_names': city_names, 'Population_in_M' : population_in_M})

cities
#Using index parameter to change the index

indexes = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']

cities = pd.DataFrame({'City_names': list(city_names), 'Population_in_M' : list(population_in_M)}, index=indexes)

cities
imdbDF = pd.read_csv('/kaggle/input/imdb1000/imdb_data.csv', sep='\t')

imdbDF = imdbDF.rename(columns={'Imdb Rating':'Rating', 'User Votes':'Votes'})

imdbDF.head()
cities
cities.index
imdbDF.index
cities['Initials'] = ['ND', 'M', 'NY', 'L', 'T', 'SE', 'SF', 'SY', 'D']

cities

cities.reindex(np.random.permutation(cities.index))
#Setting and changing indexes

cities.set_index("Initials")
#Accessing an entire column based on a variable/column name the python dictionary way

cities['Population_in_M']
type(cities['Population_in_M'])
#Naive way

cities.City_names
#Accessing particular cell from a column using index

cities['City_names'][2]
type(cities['City_names'][2])
# Accessing a varuable for it's description

imdbDF.Directors.describe()
imdbDF.Rating.mean()
cities.City_names.describe()
cities.City_names.unique()
cities.City_names.value_counts()
#Accessing a single row from dataframe

cities.iloc[1]
type(cities.iloc[1])
cities.iloc[-1]
imdbDF.iloc[25:29]
type(imdbDF.iloc[0:3])
# Retrieving an entirecolumn using iloc

cities.iloc[:,0]
# Fetching few elements from a column

# Fetching first 10 entries from total_rooms

imdbDF.iloc[:10, 5]
#Accessing a list of entries

cities.iloc[[2, 4,6],0]
cities.loc['II', 'City_names']
imdbDF.loc[2:5, 'Actors']
imdbDF.loc[2:5, ['Rating', 'Votes']]
imdbDF.Rating > 7.5
imdbDF.loc[imdbDF.Votes > 1000000]
# Logical connectors & | ! can be used to combine multiple conditions

imdbDF.loc[(imdbDF.Rating > 9) & (imdbDF.Votes > 500000)]
cities['Population_in_M'] *= 2 
cities
cities['Area_Square_km'] = list(pd.Series([1484, 603.4, 783.8, 1572, 2188, 605.2, 121.4, 1687, 4114]))

cities['population_density'] = cities['Population_in_M'] / cities['Area_Square_km']
cities
population_mean = cities.Population_in_M.mean()

cities['Away_from_mean_population'] = cities.Population_in_M.map(lambda p : p - population_mean)
cities
cities['Country'] = ['India', 'India', 'USA', 'UK', 'Japan', 'South Korea', 'USA', 'Australia', 'UAE']

cities
cities.Country+'--'+cities.City_names
cities.population_density.sum()
imdbDF.Directors.count() ,len(imdbDF.Directors.unique())
imdbDF.Genre.value_counts()
cities.groupby('Country')
cities.groupby('Country').City_names
#Equivalent of count_values()

cities.groupby('Country').Country.count()
#Any summary functions can be used

cities.groupby('Country').Population_in_M.mean()
cities.groupby('Country').apply(lambda df: df.loc[df.Population_in_M.idxmax()])
imdbDF.groupby(['Directors', 'Actors']).apply(lambda df : df.loc[df.Votes.idxmax()])
#Generating a simple summary statistics using agg() function

groupDF = cities.groupby('Country').Population_in_M.agg([len, 'min', 'max', 'mean', 'sum', 'var', 'std'])

groupDF
cities.sort_values(by='Initials')
cities.sort_index()
#using dtype property to get the data type of an element of individual column

cities.Population_in_M.dtype
#using dtypes with dataframe object gives data types for all the columns in the dataframe

cities.dtypes
cities.index.dtype
cities["Indian_Restaurant"] = [1234, 2345, 123, 234, 12, None, 254, 345, 1991]

cities
#pd.isnull(): Checks whether the value is NaN or not

#Complementary function is pd.notnull()

cities.isnull()
#Finding Nan entries specific to a column

cities[pd.isnull(cities.Indian_Restaurant)]
#Changing type for column Indian_restaurant

#cities.Indian_Restaurant.astype('int64')
#Changing NaN with desired type

cities.Indian_Restaurant.fillna(0)
cities
#Modifies the dataframe as well

cities.Indian_Restaurant.fillna(0, inplace=True)
cities
cities.Indian_Restaurant = cities.Indian_Restaurant.astype('int64')
cities
cities.Country = cities.Country.replace('USA', 'America')

cities
#Renaming a column name in a dataframe

cities = cities.rename(columns={'Population_in_M': 'Population', 'Area_Square_km': 'Area'})

cities
#Renaming vaues at index

cities.rename(index={'I':'First', 'IX':'Last'})
df1 = cities.iloc[[1,3,5,7]]

df1
df2 = cities.iloc[[0,2,4,6,8]]

df2
pd.concat([df1, df2])
df3 = cities.loc[:,['City_names', 'Initials']]

df3
df4 = cities.loc[:,['City_names', 'Country']]

df4
df3.join(df4, lsuffix='_df3', rsuffix='_df4')
df3.set_index('City_names').join(df4.set_index('City_names'))