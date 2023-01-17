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
# read in our data in the right encoding
suicide_attacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding='Windows-1252')
suicide_attacks.head()
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
# Then convert the column to lowercase and remove any trailing white spaces

# get all the unique values in the 'Province' column
province = suicide_attacks['Province'].unique()

# sort them alphabetically and then take a closer look
province.sort()
province
# convert to lower case
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
# remove trailing white spaces
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()
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
    print("All done!")
# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# get the top 10 closest matches to "kurram agency"
matches = fuzzywuzzy.process.extract("kurram agency", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.

# use the function to replace close matches to "kurram agency" with "kurram agency"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kurram agency")

# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
suicide_attacks.to_csv('suicide_attacks_inconsistency_cleaned.csv')
# read in our data
PakistanSuicideAttacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv")
# look at the first ten thousand bytes to guess the character encoding
with open("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv", 'rb') as rawdata:
    Result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(Result)
# Reading in data with the correct encoding
PakistanSuicideAttacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv", encoding = 'Windows-1252')
PakistanSuicideAttacks.head()
# get all the unique values in the 'City' column
Cities = PakistanSuicideAttacks['City'].unique()

# sort them alphabetically and then take a closer look
Cities.sort()
Cities
# convert to lower case
PakistanSuicideAttacks['City'] = PakistanSuicideAttacks['City'].str.lower()
# remove trailing white spaces
PakistanSuicideAttacks['City'] = PakistanSuicideAttacks['City'].str.strip()
# get all the unique values in the 'City' column
Cities = PakistanSuicideAttacks['City'].unique()

# sort them alphabetically and then take a closer look
Cities.sort()
Cities
# we need to replace with d.i khan and kurram agency

# get the top 10 closest matches to "d.i khan"
Matches_1 = fuzzywuzzy.process.extract("d.i khan", Cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
Matches_1

# get the top 10 closest matches to "d.i khan"
Matches_2 = fuzzywuzzy.process.extract("kurram agency", Cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
Matches_2
# use the function we wrote previously to replace close matches to "d.i khan" and "kurram agency"
replace_matches_in_column(df=PakistanSuicideAttacks, column='City', string_to_match="d.i khan")
# use the function we wrote previously to replace close matches to "d.i khan" and "kurram agency"
replace_matches_in_column(df=PakistanSuicideAttacks, column='City', string_to_match="kurram agency")
# get all the unique values in the 'City' column
Cities = PakistanSuicideAttacks['City'].unique()

# sort them alphabetically and then take a closer look
Cities.sort()
Cities
PakistanSuicideAttacks.to_csv('PakistanSuicideAttacks_inconsistency_cleaned.csv')