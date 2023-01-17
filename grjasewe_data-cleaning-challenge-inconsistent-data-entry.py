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
# much better to use the result value instead of hard-coding!
suicide_attacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding=result['encoding'])
suicide_attacks.sample(5)
# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
# convert to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
# and let's see what that did
cities = suicide_attacks['City'].unique()
cities
# Your turn! Take a look at all the unique values in the "Province" column. 
# Then convert the column to lowercase and remove any trailing white spaces
# convert to lower case
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
# remove trailing white spaces
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()
# and let's see what that did
provinces = suicide_attacks['Province'].unique()
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

# get a list of unique strings
strings = suicide_attacks['City'].unique()
    
# get the top 10 closest matches to our input string
matches = fuzzywuzzy.process.extract("kurram agency", strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kurram agency")

# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities
#ok let's try
#which columns do we have?
list(suicide_attacks)
#what is this location column that I see?
# some locations are float and some are string, so to be able to sort, we need all one type (str)
suicide_attacks['Location'] = suicide_attacks['Location'].astype(str)
locations = suicide_attacks['Location'].unique()
# sort them alphabetically and then take a closer look
locations.sort()
# although we don't necessarily have any fuzzywuzzy needs, we could benefit from removing leading spaces 
# remove extra white spaces
suicide_attacks['Location'] = suicide_attacks['Location'].str.strip()
locations = suicide_attacks['Location'].unique()
locations.sort()
# Wait, how about checkpost vs check-post.  And then check point vs checkpoint.  Well fuzzywuzzy will not really help us, so let us go with something else
# Let us say checkpost is the correct spelling and a '-' or ' ' should be remove
suicide_attacks['Location'] = suicide_attacks['Location'].replace(to_replace=r'[Cc]heck[ \-]?[Pp]ost(.*)', value=r'checkpost\1', regex=True, inplace=False)
# And then checkpoint, same
suicide_attacks['Location'] = suicide_attacks['Location'].replace(to_replace=r'[Cc]heck[ \-]?[Pp]oint(.*)', value=r'checkpoint\1', regex=True, inplace=False)
locations = suicide_attacks['Location'].unique()
locations.sort()
locations