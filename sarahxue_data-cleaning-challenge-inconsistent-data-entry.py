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
Province = suicide_attacks['Province'].unique()
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()
Province = suicide_attacks['Province'].unique()
Province.sort()
Province
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
# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.
match_kuram = fuzzywuzzy.process.extract('kuram agency', cities, limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)
match_kuram
# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_match_kuram_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest match_kuram to our input string
    match_kuram = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get match_kuram with a ratio > 90
    close_match_kuram = [match_kuram[0] for match_kuram in match_kuram if match_kuram[1] >= min_ratio]

    # get the rows of all the close match_kuram in our dataframe
    rows_with_match_kuram = df[column].isin(close_match_kuram)

    # replace all rows with close match_kuram with the input match_kuram
    df.loc[rows_with_match_kuram, column] = string_to_match
    
    # let us know the function's done
    print("All done!")
# use the function we just wrote to replace close match_kuram to "kuram agency" with "kurram agency"
replace_match_kuram_in_column(df=suicide_attacks, column='City', string_to_match="kuram agency")
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities