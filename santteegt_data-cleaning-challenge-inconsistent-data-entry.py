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
provinces = suicide_attacks['Province'].unique()
provinces.sort()
provinces
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
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
# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.
matches = fuzzywuzzy.process.extract("kuram agency", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kuram agency")
cities = suicide_attacks['City'].unique()
cities.sort()
cities
column_name = 'Location Category'
suicide_attacks[column_name].fillna('', inplace=True)
col_unique = suicide_attacks[column_name].unique()
col_unique.sort()
col_unique
correct_text = "Foreign"
matches = fuzzywuzzy.process.extract(correct_text, col_unique, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match=correct_text, min_ratio=88)
suicide_attacks[column_name].fillna('', inplace=True)
col_unique = suicide_attacks[column_name].unique()
col_unique.sort()
col_unique

def analyse_column(df, column_name):
    df[column_name].fillna('', inplace=True)
    col_unique = df[column_name].unique()
    col_unique.sort()
    print(col_unique)
    return col_unique
def get_fuzzy_scores(values, correct_text):
    matches = fuzzywuzzy.process.extract(correct_text, values, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    print(matches)

column_name = 'Location Sensitivity'
col_values = analyse_column(suicide_attacks, column_name)
correct_text = "Low"
get_fuzzy_scores(col_values, correct_text)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match=correct_text, min_ratio=100)
_ = analyse_column(suicide_attacks, column_name)

column_name = 'Open/Closed Space'
col_values = analyse_column(suicide_attacks, column_name)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match='Closed', min_ratio=90)
_ = analyse_column(suicide_attacks, column_name)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match='Open', min_ratio=90)
_ = analyse_column(suicide_attacks, column_name)

column_name = 'Target Type'
col_values = analyse_column(suicide_attacks, column_name)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match='Civilian', min_ratio=90)
_ = analyse_column(suicide_attacks, column_name)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match='Foreigner', min_ratio=90)
_ = analyse_column(suicide_attacks, column_name)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match='Police', min_ratio=90)
_ = analyse_column(suicide_attacks, column_name)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match='Religious', min_ratio=90)
_ = analyse_column(suicide_attacks, column_name)

column_name = 'Targeted Sect if any'
col_values = analyse_column(suicide_attacks, column_name)
replace_matches_in_column(df=suicide_attacks, column=column_name, string_to_match='Shiite', min_ratio=90)
_ = analyse_column(suicide_attacks, column_name)
