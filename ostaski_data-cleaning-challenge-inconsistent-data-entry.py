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
# array(['AJK', 'Balochistan', 'Baluchistan', 'Capital', 'FATA', 'Fata',
#      'KPK', 'Punjab', 'Sindh'], dtype=object)
unique, counts = np.unique(provinces, return_counts=True) # I like this way better
dict(zip(unique, counts))
#{'AJK': 1, 'Balochistan': 1, 'Baluchistan': 1, 'Capital': 1, 'FATA': 1, 'Fata': 1, 'KPK': 1, 'Punjab': 1, 'Sindh': 1}
# looks like Balochistan / Baluchistan should be Balochistan and FATA / Fata should be FATA (str.lower() will take care of the second pair)

suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()
provinces = suicide_attacks['Province'].unique()
unique, counts = np.unique(provinces, return_counts=True)
dict(zip(unique, counts))
# {'ajk': 1, 'balochistan': 1, 'baluchistan': 1, 'capital': 1, 'fata': 1, 'kpk': 1, 'punjab': 1, 'sindh': 1}

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
matches
# [('kurram agency', 100), ('kuram agency', 96), ('bajaur agency', 69), ('khyber agency', 69), ('orakzai agency', 67),
# ('mohmand agency', 59), ('mosal kor, mohmand agency', 59), ('ghallanai, mohmand agency', 49), ('gujrat', 42), ('d.i khan', 38)]

replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kurram agency")
cities = suicide_attacks['City'].unique()
cities
# array(['islamabad', 'karachi', 'quetta', 'rawalpindi', 'north waziristan', 'kohat', 'attock', 'sialkot', 'lahore', 'swat', 'hangu', 'bannu',
# 'lasbela', 'malakand', 'peshawar', 'd.i khan', 'lakki marwat', 'tank', 'gujrat', 'charsadda', 'kurram agency', 'shangla', 'bajaur agency',
# 'south waziristan', 'haripur', 'sargodha', 'nowshehra', 'mohmand agency', 'dara adam khel', 'khyber agency', 'mardan', 'bhakkar',
# 'orakzai agency', 'buner', 'pishin', 'chakwal', 'upper dir', 'muzaffarabad', 'totalai', 'multan', 'lower dir', 'sudhanoti', 'poonch',
# 'mansehra', 'karak', 'swabi', 'shikarpur', 'sukkur', 'chaman', 'khanewal', 'fateh jang', 'taftan', 'tirah valley', 'wagah', 'zhob', 'taunsa',
# 'jacobabad', 'shabqadar-charsadda', 'khuzdar', 'ghallanai, mohmand agency', 'hayatabad', 'mosal kor, mohmand agency', 'sehwan town', 'tangi',
# 'charsadda district'], dtype=object)
