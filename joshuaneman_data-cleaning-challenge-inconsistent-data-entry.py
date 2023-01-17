# modules we'll use
import pandas as pd
import numpy as np
import chardet

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
#Inspect data
suicide_attacks.head()
print(suicide_attacks.info())
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
province = suicide_attacks['Province'].unique()

# sort alphabetically and inspect
province.sort()
province
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
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kurram agency")

#verify all unique values in 'City' column
cities = suicide_attacks['City'].unique()

#Sort query alphabetically
cities.sort()
cities
print(suicide_attacks.columns)
#Inspect unique values for each column -- one at a time
verify = suicide_attacks['Location Category'].astype(str).unique()
verify.sort()
verify
#Replace Location Category "foreign" with "Foreign"
all_loc_cat = suicide_attacks['Location Category']
all_lower = all_loc_cat.str.lower()
all_capitalize = all_lower.str.capitalize()
suicide_attacks['Location Category'] = all_capitalize
#Verify replacement
print(suicide_attacks['Location Category'].unique())
#Continue to inspect unique values for each column -- one at a time
verify = suicide_attacks['Location Sensitivity'].astype(str).unique()
verify.sort()
verify
#Replace Location Sensitivity "low" with "Low"
all_loc_cat = suicide_attacks['Location Sensitivity']
all_lower = all_loc_cat.str.lower()
all_capitalize = all_lower.str.capitalize()
suicide_attacks['Location Sensitivity'] = all_capitalize
#Verify replacement
print(suicide_attacks['Location Sensitivity'].unique())
#Continue to inspect unique values for each column -- one at a time
verify = suicide_attacks['Open/Closed Space'].astype(str).unique()
verify.sort()
verify
#Replace lower case with capitalize
all_loc_cat = suicide_attacks['Open/Closed Space']
all_lower = all_loc_cat.str.lower()
all_capitalize = all_lower.str.capitalize()
suicide_attacks['Open/Closed Space'] = all_capitalize

#Remove trailing white spaces
suicide_attacks['Open/Closed Space'].str.strip()

#Verify replacement
print(suicide_attacks['Open/Closed Space'].unique())
find_true = suicide_attacks['Open/Closed Space'] == "Open "

#Locate index of records still contaiing trailing white space after 'Open '
suicide_attacks['find_true'] = find_true
idx = suicide_attacks.index[suicide_attacks['find_true']]
print(idx)

#Failed attempt to remove remaining trailing white space in "Open/Closed Space" 
#suicide_attacks["Open/Closed Space"].replace("Open ", value="Open")
#print(suicide_attacks['Open/Closed Space'].unique())

#Second attempt to remove remaining trailling white space in "Open/Closed Space"
suicide_attacks["Open/Closed Space"].replace(["Open "], "Open", inplace=True)
print(suicide_attacks['Open/Closed Space'].unique())

#Continue to inspect unique values for each column -- one at a time
verify = suicide_attacks['Target Type'].astype(str).unique()
verify.sort()
verify
#Replace lowercase with Title (first letter of each word capitalized)
all_target_type = suicide_attacks['Target Type']
all_lower = all_target_type.str.lower()
all_title = all_lower.str.title()
suicide_attacks['Target Type'] = all_title
#Verify replacement
print(suicide_attacks['Target Type'].unique())
#Replace lowercase with Title (first letter of each word capitalized)
all_target_sect = suicide_attacks['Targeted Sect if any']
all_lower_sect = all_target_sect.str.lower()
all_title_sect = all_lower_sect.str.title()
suicide_attacks['Targeted Sect if any'] = all_title_sect
#Verify replacement
print(suicide_attacks['Targeted Sect if any'].unique())
#Create new, single column for all Temperature readings
temp_c = suicide_attacks['Temperature(C)'].round(1)
temp_f = suicide_attacks['Temperature(F)'].round(1)
suicide_attacks['Consolidated Temp'] = temp_c.map(str) + " / " + temp_f.map(str)
print(suicide_attacks['Consolidated Temp'])


    