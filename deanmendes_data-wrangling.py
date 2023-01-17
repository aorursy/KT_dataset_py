# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
# csv file for 'Automobile Data Set'

csv_file = '/kaggle/input/auto.csv'

# headers supplied for the data set

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",

         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",

         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",

         "peak-rpm","city-mpg","highway-mpg","price"]
# using pandas to read csv file, assign headers to names

df = pd.read_csv(csv_file, names = headers)

df.head(10)
df.replace('?', np.nan, inplace = True)

df.head()
# identify missing data/values

missing_data = df.isnull().sum().sort_values()

missing_data
# explore d_types, columns etc

df.info()
df.describe().T
avg_normloss = df['normalized-losses'].astype('float').mean()

avg_bore = df['bore'].astype('float').mean()

avg_stroke = df['stroke'].astype('float').mean()

avg_horse = df['horsepower'].astype('float').mean()

avg_rpm = df['peak-rpm'].astype('float').mean()

print(avg_normloss, avg_bore, avg_stroke, avg_horse, avg_rpm)
# replacing

df['normalized-losses'].replace(np.nan, avg_normloss, inplace = True)

df['bore'].replace(np.nan, avg_bore, inplace = True)

df['stroke'].replace(np.nan, avg_stroke, inplace = True)

df['horsepower'].replace(np.nan, avg_horse, inplace = True)

df['peak-rpm'].replace(np.nan, avg_rpm, inplace = True)
# use idxmax() method to identify most occuring value

df['num-of-doors'].value_counts().idxmax()
df['num-of-doors'].replace(np.nan, 'four', inplace = True)
# drop row of missing values

df.dropna(subset = ['price'], axis = 0, inplace = True)



# reset index to account for dropped rows

df.reset_index(drop = True, inplace = True)
# ensure dataframe has no more missing values

missing_check = df.isnull().sum().sort_values()

missing_check
df.dtypes
# convert to floats

df[['bore', 'stroke', 'peak-rpm', 'price']] = df[['bore',

                                                  'stroke',

                                                  'peak-rpm',

                                                  'price']].astype('float')



# convert to integers

df[['normalized-losses', 'horsepower']] = df[['normalized-losses',

                                             'horsepower']].astype('int')
specs = ['length', 'width', 'height']

for s in specs:

    df[s] = df[s]/df[s].max()



# print results   

df[specs].head()
# use numpy.linspace() to create 4 equally spaced numbers between 

# the min and max of horsepowers range

bins = np.linspace(min(df['horsepower']), max(df['horsepower']), 4)

bins
# create group names for bins

bin_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], 

                                 bins,

                                 labels = bin_names, 

                                 include_lowest = True)



df[['horsepower','horsepower-binned']].head(20)
dummy_fuel = pd.get_dummies(df["fuel-type"])

dummy_fuel.head()
# change column name for clarity and ease of access

dummy_fuel.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)

dummy_fuel.head()
# merge data frame "df" and "dummy_variable_1" 

df = pd.concat([df, dummy_fuel], axis=1)



# drop original column "fuel-type" from "df"

df.drop("fuel-type", axis = 1, inplace=True)
# get dummy variable for aspiration

dummy_asp = pd.get_dummies(df['aspiration'])



# rename column for clarity

dummy_asp.rename({'std' : 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

dummy_asp.head()
# merge dummy aspiration variable to data frame

df = pd.concat([df, dummy_asp], axis=1)



# drop original column "fuel-type" from "df"

df.drop("aspiration", axis = 1, inplace=True)

df.head()
# df.to_csv('Cleaned_Auto.csv')