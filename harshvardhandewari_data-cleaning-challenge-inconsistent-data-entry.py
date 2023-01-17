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

cities= suicide_attacks['City'].unique()
cities.sort()
cities
# Your turn! Take a look at all the unique values in the "Province" column. 
provinces=suicide_attacks['Province'].unique()
provinces.sort()
print(provinces)
# Then convert the column to lowercase and remove any trailing white spaces
suicide_attacks['Province']=suicide_attacks['Province'].str.lower()
suicide_attacks['Province']=suicide_attacks['Province'].str.strip()
provinces=suicide_attacks['Province'].unique()
provinces.sort()
print(provinces)
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# get the top 10 closest matches to "d.i khan"
matches= fuzzywuzzy.process.extract("mohmand agency", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

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
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kuram agency")
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="mohmand agency")
cities = suicide_attacks['City'].unique()
cities.sort()
print(cities)
print("----------------------------------------------------------------------------------------")
# be the same city. Correct the dataframe so that they are.
replace_matches_in_column(df=suicide_attacks, column='Province', string_to_match="baluchistan")
provinces=suicide_attacks['Province'].unique()
provinces.sort()

print(provinces)
print("----------------------------------------------------------------------------------------")

replace_matches_in_column(df=suicide_attacks, column='Targeted Sect if any', string_to_match="Shiite")
sects=suicide_attacks['Targeted Sect if any'].unique()
#sects.sort()
print(sects)

def print_unique(column, df=suicide_attacks):
    df[column]=df[column].str.strip()
    df[column]=df[column].str.lower()
    uniques=df[column].unique()
    #uniques.sort()
    #I am not able to sort uniques from some of the columns, 
    #get the error : "'<' not supported between instances of 'str' and 'float'"
    #please help
    print(uniques)


print("----------------------------------------------------------------------------------------")
replace_matches_in_column(df=suicide_attacks, column='Target Type', string_to_match="Civilian")
replace_matches_in_column(df=suicide_attacks, column='Target Type', string_to_match="Police")
replace_matches_in_column(df=suicide_attacks, column='Target Type', string_to_match="Foreigner")
replace_matches_in_column(df=suicide_attacks, column='Target Type', string_to_match="Religious")

print_unique('Target Type')
print("----------------------------------------------------------------------------------------")
#print_unique('Location')
print_unique('Location Category')
print_unique('Location Sensitivity')

print("----------------------------------------------------------------------------------------")
suicide_attacks.head()
