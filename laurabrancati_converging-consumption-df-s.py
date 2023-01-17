# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
water = pd.read_csv("/kaggle/input/water-eda/water_cleaned.csv", index_col = "timestamp", parse_dates = True)

chilled = pd.read_csv("/kaggle/input/chilled-eda/chilled_water_cleaned.csv", index_col = "timestamp", parse_dates = True)

hot = pd.read_csv("/kaggle/input/hot-eda/hot_water_cleaned.csv", index_col = "timestamp", parse_dates = True)

elec = pd.read_csv("/kaggle/input/electricity-eda-genome/electricity_cleaned_new.csv", index_col = "timestamp", parse_dates = True)

gas = pd.read_csv("/kaggle/input/gas-eda-genome/gas_cleaned_new.csv", index_col = "timestamp", parse_dates = True)
df_list = [water, chilled, hot, elec, gas]
water.head(3)
chilled.head(3)
hot.head(3)
elec = elec.drop("Unnamed: 0", axis =1)

gas = gas.drop("Unnamed: 0", axis =1)
gas.head(2)
#resampling to make sure that all df's have weekly timestamps

elec = elec.resample("W").mean()

water = water.resample("W").mean()

chilled = chilled.resample("W").mean()

hot = hot.resample("W").mean()

gas = gas.resample("W").mean()
water.head(3)
elec.shape
gas.shape
water.shape
#separate by site, find sums, concat SUMS 
#possible animal names

animals = ["Panther", "Moose", "Eagle", "Cockatoo", "Fox", "Peacock", "Bull",\

           "Hog", "Crow", "Bobcat", "Robin", "Bear", "Lamb", "Rat", "Gator", "Wolf", "Shrew", "Swan","Mouse"]
#list should be list of dataframes

def summing_per_site(df, animal_list):

    df_copy = df.copy()

    for animal in animal_list:

        col_list = []

        animal_in_df = False

        new_df = pd.DataFrame()

        for col in df.columns:

            if animal in col:

                animal_in_df = True

                col_list.append(col)

                new_df[col] = df_copy[col]

        if animal_in_df:

            df_copy[animal] = new_df.sum(axis = 1)

            df_copy = df_copy.drop(col_list, axis = 1)

    return df_copy  

                    
water = summing_per_site(water, animals)

water.head()
#elec has a column with two animal names so chagning the name of that

elec = elec.rename(columns={"Lamb_education_Robin": "Lamb_education_Robi"})
elec = summing_per_site(elec, animals)

elec.head()
chilled = summing_per_site(chilled, animals)

chilled.head()
hot = summing_per_site(hot, animals)

hot.head()
gas = gas.rename(columns={"Lamb_education_Robin": "Lamb_education_Robi"})

gas = summing_per_site(gas, animals)

gas.head()
df_names = ["water", "chilled", "hot", "elec", "gas"]

df_list = [water, chilled, hot, elec, gas]
for df in df_list:

    print(df.columns)
def creating_site_df(df_list, animal, names):

    name_index = 0

    animal_df = pd.DataFrame()

    for df in df_list:

        if animal in df.columns:

            animal_df[names[name_index]] = df[animal]

            name_index += 1

        else:

            name_index += 1

    return animal_df
panther = creating_site_df(df_list, "Panther", df_names)

panther.head()
moose = creating_site_df(df_list, "Moose", df_names)

moose.head()
eagle = creating_site_df(df_list, "Eagle", df_names)

eagle.head()
cockatoo = creating_site_df(df_list, "Cockatoo", df_names)

cockatoo.head()
fox = creating_site_df(df_list, "Fox", df_names)

fox.head()
peacock = creating_site_df(df_list, "Peacock", df_names)

peacock.head()
bull = creating_site_df(df_list, "Bull", df_names)

bull.head()
hog = creating_site_df(df_list, "Hog", df_names)

hog.head()
crow = creating_site_df(df_list, "Crow", df_names)

crow.head()
bobcat = creating_site_df(df_list, "Bobcat", df_names)

moose.head()
robin = creating_site_df(df_list, "Robin", df_names)

robin.head()
bear = creating_site_df(df_list, "Bear", df_names)

bear.head()
lamb = creating_site_df(df_list, "Lamb", df_names)

lamb.head()
rat = creating_site_df(df_list, "Rat", df_names)

rat.head()
gator = creating_site_df(df_list, "Gator", df_names)

gator.head()
wolf = creating_site_df(df_list, "Wolf", df_names)

wolf.head()
shrew = creating_site_df(df_list, "Shrew", df_names)

shrew.head()
swan = creating_site_df(df_list, "Swan", df_names)

swan.head()
mouse = creating_site_df(df_list, "Mouse", df_names)

mouse.head()
#animals = ["Panther", "Moose", "Eagle", "Cockatoo", "Fox", "Peacock", "Bull", "Hog", "Crow", "Bobcat", "Robin", "Bear", "Lamb", "Rat", "Gator", "Wolf", "Shrew", "Swan","Mouse"]
#saving all of the dataframes as separate csv's 

panther.to_csv("panther_sums.csv")

moose.to_csv("moose_sums.csv")

eagle.to_csv("eagle_sums.csv")

cockatoo.to_csv("cockatoo_sums.csv")

fox.to_csv("fox_sums.csv")

peacock.to_csv("peacock_sums.csv")

bull.to_csv("bull_sums.csv")

hog.to_csv("hog_sums.csv")

crow.to_csv("crow_sums.csv")

bobcat.to_csv("bobcat_sums.csv")

robin.to_csv("robin_sums.csv")

bear.to_csv("bear_sums.csv")

lamb.to_csv("lamb_sums.csv")

rat.to_csv("rat_sums.csv")

gator.to_csv("gator_sums.csv")

wolf.to_csv("wolf_sums.csv")

shrew.to_csv("shrew_sums.csv")

#nothing in swan

mouse.to_csv("mouse_sums.csv")