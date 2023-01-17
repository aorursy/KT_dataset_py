# Required Libraries for the Exercise
import numpy as np              # linear algebra
import pandas as pd             # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns           # plotting
import matplotlib.pyplot as plt # plotting

from scipy import stats         # Box-Cox Tranformation (Day 2)
from mlxtend.preprocessing \
    import minmax_scaling       # Minimum - Maximum Scaling (Day 2)
import datetime                 # date-time transformations (Day 3)
import chardet                  # character encoding module (Day 4)
import fuzzywuzzy
from fuzzywuzzy \
    import process              # text mining (Day 5)

# Even though I dont think this is necessary, it seems Rachael gives importance to repoducibility.
np.random.seed(23)
# Data Import for this Exercises
sf_permits  = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")
kickstarter = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
gunviolence = pd.read_csv("../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv")
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
suicide_att = pd.read_csv("../input/pakistansuicideattacks/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", encoding = "Windows-1252") # as Rachael suggests.
sf_permits.head(6)
sf_permits.dtypes
na_count  = sf_permits.isnull().sum()
row_count = np.product(sf_permits.shape)
na_perc   = 100 *  na_count.sum() / row_count
na_perc   = na_perc.round(2)
print('Overall percentage of NA values in this dataset is %{0}.'.format(na_perc))
print('There are {0} NA values in Street Number Suffix variable while {1} NA values in Zipcode variable.'.format(na_count[7], na_count[40]))
sf_permits_nonna_rows = sf_permits.dropna()
sf_permits_nonna_cols = sf_permits.dropna(axis = 1)
print('Rows left after dropping rows with at least one NA value: {0} \n'. format(sf_permits_nonna_rows.shape[0]))
print('Columns left after dropping columns with at least one NA value: {0} \n'. format(sf_permits_nonna_cols.shape[1]))
sf_permits_imputated = sf_permits.fillna(method = 'bfill', axis =0).fillna(0)
sf_permits_imputated.head(6)
goal = kickstarter.goal
scaled_goal = minmax_scaling(goal, columns = [0])

fig, ax = plt.subplots(1,2)
sns.distplot(goal, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_goal, ax = ax[1])
ax[1].set_title("Scaled Data")
index_of_pledged = kickstarter.pledged > 0
positive_pledges = kickstarter.pledged.loc[index_of_pledged]
scaled_pledges = stats.boxcox(positive_pledges)[0]

fig, ax = plt.subplots(1,2)
sns.distplot(positives_pledges, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_pledges, ax = ax[1])
ax[1].set_title("Normalized Data")
gunviolence.head()
fig, ax = plt.subplots(1,2)
sns.distplot(gunviolence.n_killed, ax = ax[0])
ax[0].set_title("Dead People by Gun Violence")
sns.distplot(gunviolence.n_injured, ax = ax[1])
ax[1].set_title("Injured People by Gun Violence")
# Figuring out positive values.(Box-Cox only accepts positive values)
killed = gunviolence[gunviolence.n_killed > 0].n_killed
injured = gunviolence[gunviolence.n_injured > 0].n_injured

# Box-Cox Transformation
killed_boxcox = stats.boxcox(killed)[0]
injured_boxcox = stats.boxcox(injured)[0]

# Plot!
fig, ax = plt.subplots(2,2)
sns.distplot(gunviolence.n_killed, ax = ax[0, 0])
ax[0, 0].set_title("Dead")
sns.distplot(killed_boxcox, ax = ax[0, 1])
ax[0, 1].set_title("Dead (Box-Cox)")
sns.distplot(gunviolence.n_injured, ax = ax[1, 0])
ax[1, 0].set_title("Injured")
sns.distplot(injured_boxcox, ax = ax[1, 1])
ax[1, 1].set_title("Injured (Box-Cox)")
print('In Earthquakes dataset, date column is formatted as {0} by default.'.format(earthquakes['Date'].dtype))
earthquakes['Date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)
print('In Earthquakes dataset, parsed date column is formatted as {0} by using infer_datetime_format option since there were more than 1 type.'\
      .format(earthquakes['Date_parsed'].dtype))
earthquakes['month'] = earthquakes['Date_parsed'].dt.month
earthquakes.groupby(['month']).size()
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(1000000))

print(result)
killerpolice = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "Windows-1252")
killerpolice.head()
# Now, let's save.
killerpolice.to_csv("PoliceKillingsUS_utf8.csv")
suicide_att['City'] = suicide_att['City'].str.lower()
suicide_att['City'] = suicide_att['City'].str.strip()
matches = fuzzywuzzy.process.extract("kuram agency", suicide_att['City'], limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)
matches
def column_fixer(df, col, string_to_match, min_ratio = 90):
    unique_vals = df[col].unique()
    matches = fuzzywuzzy.process.extract(string_to_match, unique_vals, limit = 10, scorer = fuzzywuzzy.fuzz.token_sort_ratio)
    matches_above_min = [matches[0] for matches in matches if matches[1] >= min_ratio] # first column names, second column scores.
    matched_rows = df[col].isin(matches_above_min)
    df.loc[matched_rows] = string_to_match
    print("Voila!")
column_fixer(df = suicide_att, col = "City", string_to_match = "kuram agency")
fixed_cities = suicide_att['City'].unique()
fixed_cities.sort()
fixed_cities