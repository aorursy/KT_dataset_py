import pandas as pd

%matplotlib inline



df = pd.read_csv("../input/us-minimum-wage-by-state-from-1968-to-2017/Minimum Wage Data.csv", encoding="latin")

df.head()
min_wage_df = pd.DataFrame()



for name, group in df.groupby('State'):

    if min_wage_df.empty:

        min_wage_df = group.set_index('Year')[['Low.2018']].rename(columns={'Low.2018': name})

    else:

        min_wage_df = min_wage_df.join(group.set_index('Year')[['Low.2018']].rename(columns={'Low.2018': name}))

        

min_wage_df.head()
min_wage_df.describe()
min_wage_df.corr().head()
min_wage_df.corr()
df[df['Low.2018']==0]['State'].unique()
import numpy as np



min_wage_corr = min_wage_df.replace(0, np.NaN).dropna(axis=1).corr()

min_wage_corr
import matplotlib.pyplot as plt

plt.matshow(min_wage_corr)
labels = [c[:2] for c in min_wage_corr.columns]



# Figure is the base canvas in the plot

fig = plt.figure(figsize=(12, 12))



# Add one subplot with 2 dimensions

ax = fig.add_subplot(111)



# Fill the data from correlation table and change the color map to be a good one

# green = 1.0, red = 0.0

ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)



# Set labels to states

ax.set_yticklabels(labels)

ax.set_xticklabels(labels)



# Ask to display all labels than what is the asthetics

ax.set_xticks(np.arange(len(labels)))

ax.set_yticks(np.arange(len(labels)))



# Show

plt.show()
import requests



web = requests.get("https://www.infoplease.com/state-abbreviations-and-state-postal-codes")

dfs = pd.read_html(web.text)
# Checking the data frames downloaded

for df in dfs:

    print(df.head())
postal_code_df = dfs[0].copy()

postal_code_df.set_index('State/District', inplace=True)

postal_code_df.head()
postal_code_dict = postal_code_df[["Postal Code"]].to_dict()['Postal Code']

postal_code_dict['Federal (FLSA)'] = "FLSA"

postal_code_dict['Guam'] = 'GU'

postal_code_dict['Puerto Rico'] = 'PR'



postal_code_dict
labels = [postal_code_dict[state] for state in min_wage_corr.columns]
# Figure is the base canvas in the plot

fig = plt.figure(figsize=(12, 12))



# Add one subplot with 2 dimensions

ax = fig.add_subplot(111)



# Fill the data from correlation table and change the color map to be a good one, 

# green = 1.0, red = 0.0

ax.matshow(min_wage_corr, cmap=plt.cm.RdYlGn)



# Set labels to states

ax.set_yticklabels(labels)

ax.set_xticklabels(labels)



# Ask to display all labels than what is the asthetics

ax.set_xticks(np.arange(len(labels)))

ax.set_yticks(np.arange(len(labels)))



# Show

plt.show()
unemp_county = pd.read_csv('../input/unemployment-by-county-us/output.csv')

unemp_county.head()
min_wage_df = min_wage_df.replace(0, np.NaN).dropna(axis=1)

min_wage_df.head()
def get_min_wage(year, state):

    try:

        return min_wage_df.loc[year, state]

    except:

        return np.NaN



# Get the min wage for a state for an year, fail with a NaN if does not exist 

get_min_wage(2012, 'Colorado')
%%time



unemp_county['min_wage'] = list(map(get_min_wage, unemp_county['Year'], unemp_county['State']))
unemp_county.head()
unemp_county[["Rate", "min_wage"]].corr()
unemp_county[["Rate", "min_wage"]].cov()
pres16 = pd.read_csv('../input/2016uspresidentialvotebycounty/pres16results.csv')

pres16.head()
len(unemp_county)
unemp_county_2015 = unemp_county.copy()[(unemp_county['Year'] == 2015) & (unemp_county['Month'] == 'February')]

unemp_county_2015.head()
pres16['st'].unique()
unemp_county_2015['State'] = unemp_county_2015['State'].map(postal_code_dict)
unemp_county_2015.head()
print(len(unemp_county_2015))

print(len(pres16))
pres16.rename(columns={'county': 'County', 'st': 'State'}, inplace=True)
pres16.head()
for df in [unemp_county_2015, pres16]:

    df.set_index(['County', 'State'], inplace=True)
pres16.head()
pres16 = pres16.copy()[pres16['cand'] == 'Donald Trump'][["pct"]]

pres16.dropna(inplace=True)

pres16.head()
all_together = unemp_county_2015.merge(pres16, on=['County', 'State'])

all_together.dropna(inplace=True)
all_together.head()
all_together.drop("Year", axis=1, inplace=True)

all_together.drop("Month", axis=1, inplace=True)

all_together.head()
all_together.corr()
all_together.cov()