# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
# Analysis of NA values in dataset

df.isna().sum()
# Dropping columns which have high NA values.

df = df.drop(['End_Lat', 'TMC', 'End_Lng', 'Number', 'Wind_Chill(F)', 'Precipitation(in)'

              ,'Airport_Code','Zipcode', 'County', 'Country','Street', 

              'Description', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'], axis = 1) 
# Displaying columns after cleaning up dataframe

df.columns
# Characterstics of data in dataset

df.info()
# Displaying top 5 records in dataset

df.head()
# Top 15 States with most accidents

df.State.value_counts().head(15).plot(kind='barh', figsize=(7,5))
# Count pot of source of accident info.

df.Source.value_counts().head(3).plot(kind='barh', figsize=(7,5))
# Top 15 Cities with most accidents

df.City.value_counts().head(15).plot(kind='barh', figsize=(7,5))
# Relationship between severity and traffic signal

sns.countplot(df['Severity'], hue=df['Traffic_Signal'])
# Relationship between severity and side of road

sns.countplot(df['Severity'], hue=df['Side'])
# Analysis causes of accidents in top 5 most accident prone States

df_state_wise = df[df.State.isin(['CA','TX','FL','SC', 'NC'])]

sns.catplot(x="Severity", hue="Sunrise_Sunset", row="State", data=df_state_wise, kind="count", height=3, aspect=4)
df_CA = df_state_wise[df_state_wise['State']=='CA']
df_CA.head(5)
# Analysis of causes of accidents in CA

# Inference: Traffic signal and junctions contribute to most of accidents. Also, mostly accidents happens at day time.

print(df_CA.Sunrise_Sunset.value_counts())

print(df_CA.Traffic_Signal.value_counts())

print(df_CA.Turning_Loop.value_counts())

print(df_CA.Traffic_Calming.value_counts())

print(df_CA.Stop.value_counts())

print(df_CA.Roundabout.value_counts())

print(df_CA.Junction.value_counts())

print(df_CA.Station.value_counts())