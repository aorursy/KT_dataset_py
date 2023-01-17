import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import fuzzywuzzy

from fuzzywuzzy import process

import chardet



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# look at the first 100000 bytes to guess the character encoding

with open("/kaggle/input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", "rb") as rawdata:

    result = chardet.detect(rawdata.read(100000))

    

# check what the chaacter encoding might be

print(result)
df = pd.read_csv("/kaggle/input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", encoding = "Windows-1252")
df.columns
df.head()
# get all the unique values in the 'City' column

cities = df['City'].unique()



# sort then in alphabetically and then take a closer look

cities.sort()

cities
df['City'].nunique()
# convert to lower case

df['City'] = df['City'].str.lower()



# removing trailing white spaces

df['City'] = df['City'].str.strip()
# get all the unique values in the 'City' column after converting in lower case and removing white spaces

cities = df['City'].unique()



# sort then in alphabetically and then take a closer look

cities.sort()

cities
df['City'].nunique()
# get all the unique values in the `Province` column

provinces = df['Province'].unique()



# sort them alphabetically and then take a closer look

provinces.sort()

provinces
df['Province'].nunique()
# convert to lower case

df['Province']  = df['Province'].str.lower()



# removing trailing white spaces

df['Province']  = df['Province'].str.strip()
# get all the unique values in the `Province` column after converting in lower case and removing white spaces

provinces = df['Province'].unique()



# sort them alphabetically and then take a closer look

provinces.sort()

provinces
df['Province'].nunique()
# get all the unique values in the 'City' column

cities = df['City'].unique()



# sort then in alphabetically and then take a closer look

cities.sort()

cities
# get the top ten closest matches to "d.i khan"

matches = fuzzywuzzy.process.extract("d.i khan", cities, limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)



# take a look at them

matches
# function to replace rows in the provided column of the provided dataframe that match the provided string above the provided ratio with the provided string

def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):

    

    # get the list of unique strings

    strings = df[column].unique()

    

    # get the top 10 closest matches to our input string

    matches = fuzzywuzzy.process.extract(string_to_match, strings, limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)

    

    # only get matches with a ratio > 90

    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    

    # get the rows of all the close matches in our dataframe

    rows_with_matches = df[column].isin(close_matches)

    

    # replace all rows with close matches with the input matches

    df.loc[rows_with_matches, column] = string_to_match

    

    # let us know the function's done

    print("All done!")
# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"

replace_matches_in_column(df = df, column = "City", string_to_match = "d.i khan")
# get all the unique values in the 'City' column

cities = df['City'].unique()



# sort them alphabetically and then take a closer look

cities.sort()

cities
# It looks like 'kuram agency' and 'kurram agency' should be the same city.

replace_matches_in_column(df = df, column = "City", string_to_match = 'kuram agency')
# get all the unique values in the 'City' column

cities = df['City'].unique()



# sort them alphabetically and then take a closer look

cities.sort()

cities
# province column 



# get all the unique values in the `Province` column after converting in lower case and removing white spaces

provinces = df['Province'].unique()



# sort them alphabetically and then take a closer look

provinces.sort()

provinces
replace_matches_in_column(df = df, column = "Province", string_to_match = "baluchistan")
# get all the unique values in the `Province` column after converting in lower case and removing white spaces

provinces = df['Province'].unique()



# sort them alphabetically and then take a closer look

provinces.sort()

provinces