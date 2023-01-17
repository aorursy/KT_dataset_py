import numpy as np
import pandas as pd
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv", low_memory=False)
np.random.seed(0)
sf_permits.shape
sf_permits.isnull().sum()[:10]
sf_permits_prop_missing = (sf_permits.isnull().sum().sum()/np.product(sf_permits.shape))*100
print('Percentage of missing values in sf_permits: {}'.format(sf_permits_prop_missing))
print('Number of columns with no missing values: {}'.format(sf_permits.dropna(axis=1, inplace=False).shape[1]))
sf_permits_sample = sf_permits.sample(n=7)
sf_permits_sample.fillna(method='bfill', axis=0).fillna(0)  #fill remaining missing values with 0
ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv')
ks.head()
import matplotlib.pyplot as plt

goal_original = ks['goal']
goal_scaled = (goal_original - goal_original.min())/(goal_original.max() - goal_original.min())
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].hist(goal_original, ec='black')
ax[0].set_title('Original Data')
ax[1].hist(goal_scaled, ec='black')
ax[1].set_title('Scaled Data')
from scipy.stats import boxcox
import seaborn as sns

msk = ks.pledged>0
positive_pledges = ks[msk].pledged
normalized_pledges = boxcox(x=positive_pledges)[0]
fig, ax = plt.subplots(1,2, figsize=(12,5))
sns.distplot(a=positive_pledges, hist=True, kde=True, ax=ax[0])
ax[0].set_title('Original Positive Pledges')
sns.distplot(a=normalized_pledges, hist=True, kde=True, ax=ax[1])
ax[1].set_title('Normalized Positive Pledges')
quakes = pd.read_csv('../input/earthquake-database/database.csv')

# Check type of date column
quakes['Date'].dtype
quakes.Date.head()
quakes.loc[3378,'Date']
quakes['date_parsed'] = pd.to_datetime(quakes.Date, infer_datetime_format=True) 
quakes.date_parsed.head()
day_of_month = quakes['date_parsed'].dt.day

# Plot the days
plt.hist(day_of_month, bins=31, ec='black')
before = "This is an interesting text: 你好"

after = before.encode(encoding='UTF-8', errors='replace')
after.decode('UTF-8') # No issue here
after = before.encode(encoding='ASCII', errors='replace')
after.decode('ASCII') # Lose information
import chardet

with open('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))  # Read first 10000 bytes.
result
# See if this is the right encoding
police_killings = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding='ascii')
with open('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
police_killings = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding='Windows-1252')
suicide_attacks = pd.read_csv('../input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv', encoding='Windows-1252')
cities = suicide_attacks['City'].unique()
cities.sort()
cities
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
suicide_attacks['City'] = suicide_attacks['City'].str.strip()
provinces = suicide_attacks['Province'].unique()
provinces.sort()
provinces
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()
import fuzzywuzzy
from fuzzywuzzy import process

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
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kuram agency")
cities = suicide_attacks['City'].unique()
cities.sort()
cities