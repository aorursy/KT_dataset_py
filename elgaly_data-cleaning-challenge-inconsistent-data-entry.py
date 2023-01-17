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

suicide_attacks.City.sample(6)
suicide_attacks.describe()
na_suicide_attacks=suicide_attacks.isnull().sum()
na_suicide_attacks
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
providence=suicide_attacks['Province'].unique()
providence.sort()
providence
suicide_attacks['Province']=suicide_attacks['Province'].str.lower()
suicide_attacks['Province']=suicide_attacks['Province'].str.strip()
fix_provi=suicide_attacks['Province'].unique()
fix_provi.sort()
fix_provi
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
matches = fuzzywuzzy.process.extract("kurram agency", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
# take a look at them
matches
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kurram agency")
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
matches = fuzzywuzzy.process.extract("Baluchistan", fix_provi, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches
replace_matches_in_column(df=suicide_attacks, column='Province', string_to_match="baluchistan")
fix_provi=suicide_attacks['Province'].unique()
fix_provi.sort()
fix_provi
suicide_attacks_var6 = pd.read_csv("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv",encoding='Windows-1252')
suicide_attacks_var6.head()
#look for inconsistent in City an Province
cities_var6 = suicide_attacks_var6['City'].unique()
cities_var6.sort()
cities_var6
provi_var6 = suicide_attacks_var6['Province'].unique()
provi_var6.sort()
provi_var6
suicide_attacks_var6['City'] = suicide_attacks_var6['City'].str.lower()
suicide_attacks_var6['City'] = suicide_attacks_var6['City'].str.strip()

suicide_attacks_var6['Province'] = suicide_attacks_var6['Province'].str.lower()
suicide_attacks_var6['Province'] = suicide_attacks_var6['Province'].str.strip()
cities_var6 = suicide_attacks_var6['City'].unique()
cities_var6.sort()
cities_var6
matches1 = fuzzywuzzy.process.extract("kurram agency", cities_var6, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches2 = fuzzywuzzy.process.extract("d.i khan", cities_var6, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
print(matches1,"d.i khan: ",matches2)
replace_matches_in_column(df=suicide_attacks_var6, column='City', string_to_match="kurram agency")
replace_matches_in_column(df=suicide_attacks_var6, column='City', string_to_match="d.i khan")
cities_var6 = suicide_attacks_var6['City'].unique()
cities_var6.sort()
cities_var6
provi_var6 = suicide_attacks_var6['Province'].unique()
provi_var6.sort()
provi_var6
matches = fuzzywuzzy.process.extract("Baluchistan", provi_var6, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
# take a look at them
matches
replace_matches_in_column(df=suicide_attacks_var6, column='Province', string_to_match="baluchistan")
provi_var6 = suicide_attacks_var6['Province'].unique()
provi_var6.sort()
provi_var6