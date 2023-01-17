# modules we'll use
import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

# set seed for reproducibility
np.random.seed(0)
# look at the first ten thousand bytes to guess the character encoding
with open("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)
# read in our dat
suicide_attacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding='Windows-1252')
#suicide_attacks.info()
suicide_attacks.head(5)
print(suicide_attacks.shape)
print(suicide_attacks.isnull().sum())
#earthquakes.isnull().sum(axis=1)
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# convert to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
# Your turn! Take a look at all the unique values in the "Province" column. 
#provinces = suicide_attacks['Province'].unique()

# sort them alphabetically and then take a closer look
#provinces.sort()
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
# remove trailing white spaces
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()

# Then convert the column to lowercase and remove any trailing white spaces
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# get the top 10 closest matches to "d.i khan"
matches = fuzzywuzzy.process.extract("d.i khan", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    #print("All done!")
# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kuram agency")
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# check what the character encoding might be  and retunr the dataframe
def find_encoding(file_path):
    with open(file_path, 'rb') as rawdata:
        f_encoding = chardet.detect(rawdata.read(100000))
    return pd.read_csv(file_path, encoding=f_encoding['encoding'])

#this function  takes a column and convert that column to lowercase,remove trailing white spaces and removes inconssistant Data Entry 
def process_column(df,column):
    # convert to lower case
    df[column] = df[column].str.lower()
    # remove trailing white spaces
    df[column] = df[column].str.strip()
    column_array = df[column].tolist()
    column_sorted = sorted(column_array)
    #print(column_sorted)
    for i in range(len(column_sorted)):
        replace_matches_in_column(df, column, string_to_match=column_sorted[i])
    return df

# Check encoding and read file into a data frame
file = "../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv"
suicide_attacks_n = find_encoding(file)
#Remove data inconssitencies in column 'City'
suicide_attacks_n = process_column(df = suicide_attacks_n,column='City')
#
cities = suicide_attacks_n['City'].unique()
# sort them alphabetically and then take a closer look
cities.sort()
cities

