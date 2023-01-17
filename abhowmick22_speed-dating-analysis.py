# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# Any results you write to the current directory are saved as output.
data_df = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")
fields = data_df.columns
# Num of fields and some of their names
# print('Number of fields: {0}\n1-10: {1}\n11-20: {2}\n21-30: {3}'.format(len(fields), 
#       fields[0:11], fields[11:21], fields[21:31]))
# Some samples
# print('Example row: {}'.format(data_df.head(1)))
print('Total number of people that participated, assuming person does not appear in more than one wave: {}'.format(len(data_df['iid'].unique())))
print('Total number of dates occurred: {}'.format(len(data_df.index)))
fig, axes = plt.subplots(1, 2, figsize=(17,5))

# The number of dates per person
num_dates_per_male = data_df[data_df.gender == 1].groupby('iid').apply(len)
num_dates_per_female = data_df[data_df.gender == 0].groupby('iid').apply(len)
axes[0].hist(num_dates_per_male, bins=22, alpha=0.5, label='# dates per male')
axes[0].hist(num_dates_per_female, bins=22, alpha=0.5, label='# dates per female')
# axes[0].suptitle('Number of dates per male/female')
axes[0].legend(loc='upper right')

# The number of matches per person
matches = data_df[data_df.match == 1]
matches_male = matches[matches.gender == 1].groupby('iid').apply(len)
matches_female = matches[matches.gender == 0].groupby('iid').apply(len)
axes[1].hist((matches_male / num_dates_per_male).dropna(), alpha=0.5, label='male match percentage')
axes[1].hist((matches_female / num_dates_per_female).dropna(), alpha=0.5, label='female match percentage')
axes[1].legend(loc='upper right')
# axes[1].suptitle('Matches per person by gender')

print('Avg. dates per male: {0:.1f}\t\tAvg. dates per female: {1:.1f}\nAvg. male match percentage: {2:.2f}\tAvg. female match percentage: {3:.2f}'.format(
        num_dates_per_male.mean(), 
        num_dates_per_female.mean(),
        (matches_male / num_dates_per_male).mean() * 100.0,
        (matches_female / num_dates_per_female).mean() * 100.0))
# Preprocessing
def str_to_float(series):
    return series.apply(lambda x: str(x).replace(",", "")).astype('float64')

for trait in ['mn_sat', 'tuition', 'income']:
    data_df[trait] = str_to_float(data_df[trait])
data_df['pid'] = data_df['pid'].fillna(-1.0).astype('int64')  # Invalid PID as -1

# Compute the features for additional traits
def standardize_feature(series):
    return (series - series.mean()) / series.std(ddof=0)
    
data_df['financial'] = standardize_feature(data_df['tuition']) \
                       .add(standardize_feature(data_df['income']), fill_value=0.0)
data_df['liberal'] = data_df['imprace'] + data_df['imprelig']

# Create a dataframe containing information for each person that needs to be looked up
profiles = data_df[['iid', 'mn_sat', 'goal', 'go_out', 'date', 'field_cd', 'financial', 'liberal']]\
           .set_index(keys='iid').drop_duplicates()
profiles = profiles.fillna(profiles.mean())  # Fill NaN values with mean
# Computing trait similarities/differences
data_df['age_diff'] = data_df['age'].sub(data_df['age_o']).abs()  # Age difference

def is_similar_profession(x, profiles):
    if np.isnan(x['field_cd']) or np.isnan(x['pid']) or x['pid'] not in profiles.index:
        return False
    else:
        return x['field_cd'] == profiles.loc[x['pid']]['field_cd']
data_df['sim_profession'] = data_df[['field_cd', 'pid']]\
                            .apply(lambda x: is_similar_profession(x, profiles), axis=1)

def trait_difference(trait):
    trait_other = data_df['pid'].apply(lambda x: profiles.loc[x][trait] if x in profiles.index else None)
    return data_df[trait].sub(trait_other)
    
for trait in ['mn_sat', 'goal', 'go_out', 'date', 'financial', 'liberal']:
    data_df[trait + '_diff'] = trait_difference(trait)
# Visualize the univariate relations and correlations of the features using their kernel density estimates
sns.pairplot(data=data_df, hue='match', vars=['age_diff', 'int_corr', 'samerace', \
                                              'liberal_diff', 'financial_diff', 'mn_sat_diff', \
                                              'goal_diff', 'go_out_diff', 'date_diff'], \
             size=1.5, diag_kind='kde')