# modules we'll use

import pandas as pd

import numpy as np



# helpful modules

import fuzzywuzzy 

#approximate string matching (often colloquially referred to as fuzzy string searching) 

#is the technique of finding strings that match a pattern approximately (rather than exactly)

from fuzzywuzzy import process

import chardet #Universal Character Encoding Detector



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



# get all the unique values in the 'Province' column

provinces = suicide_attacks['Province'].unique()



# sort alphabetically

provinces.sort()



# convert to lowercase

suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()

provinces = suicide_attacks['Province'].unique()



# remove trailing white spaces

suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()



provinces = suicide_attacks['Province'].unique()

provinces.sort()

provinces
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

    # df = dataframe

    

    # get a list of unique strings

    strings = df[column].unique()

    

    # get the top 10 closest matches to our input string

    matches = fuzzywuzzy.process.extract(string_to_match, strings, 

                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)



    # only get matches with a ratio > 90

    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    #print("close_matches")

    #print(close_matches)

    #print("\n")



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



# function to replace rows in the provided column of the provided dataframe

# that match the provided string above the provided ratio with the provided string

# same as replace_matches_in_column function; for practice

def replace_matches_in_column_2(dataframe, column, string_to_match, min_ratio):

    

    # get a list of unique strings

    strings = dataframe[column].unique()

    

    # get the top 10 closest matches to our string_to_match

    matches = fuzzywuzzy.process.extract(string_to_match, strings, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)



    # only get matches with a ratio >= 90

    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    

    #get the rows of all the close matches in our dataframe

    rows_with_matches = dataframe[column].isin(close_matches)



    # replace all rows with close matches with the input matches

    dataframe.loc[rows_with_matches, column] = string_to_match

    

    # let us know the function's done

    print("All done!")

    

replace_matches_in_column_2(dataframe=suicide_attacks, column='City', string_to_match='kuram agency', min_ratio=90)



cities = suicide_attacks['City'].unique()

cities.sort()

cities