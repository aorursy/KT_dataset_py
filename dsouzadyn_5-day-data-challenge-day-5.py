import pandas as pd

import scipy.stats

import seaborn as sns
# Read in our data

digimon_data = pd.read_csv('../input/DigiDB_digimonlist.csv')

# Print the first few rows

digimon_data.head()
digi_type = digimon_data['Type']

digi_stage = digimon_data['Stage']
scipy.stats.chisquare(digi_type.value_counts())
sns.countplot(digi_stage)