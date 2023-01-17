import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from subprocess import check_output

data_df = pd.read_csv('../input/Speed Dating Data.csv', encoding="ISO-8859-1")
fields = data_df.columns

print('Total number of people that participated, assuming person does not appear in more than one wave: {}'.format(len(data_df['iid'].unique())))
print('Total number of dates occurred: {}'.format(len(data_df.index)))
fig, axes = plt.subplots(1, 2, figsize=(10,5))

num_dates_per_male = data_df[data_df.gender == 1].groupby('iid').apply(len)
num_dates_per_female = data_df[data_df.gender == 0].groupby('iid').apply(len)
axes[0].hist(num_dates_per_male, bins=22, alpha=0.5, label='# dates per male')
axes[0].hist(num_dates_per_female, bins=22, alpha=0.5, label='# dates per female')
# axes[0].suptitle('Number of dates per male/female')
axes[0].legend(loc='upper right')

matches = data_df[data_df.match == 1]
matches_male = matches[matches.gender == 1].groupby('iid').apply(len)
matches_female = matches[matches.gender == 0].groupby('iid').apply(len)
axes[1].hist((matches_male / num_dates_per_male).dropna(), alpha=0.5, label='male match percentage')
axes[1].hist((matches_female / num_dates_per_female).dropna(), alpha=0.5, label='female match percentage')
axes[1].legend(loc='upper right')

print('Avg. dates per male: {0:.1f}\t\tAvg. dates per female: {1:.1f}\nAvg. male match percentage: {2:.2f}\tAvg. female match percentage: {3:.2f}'.format(
        num_dates_per_male.mean(), 
        num_dates_per_female.mean(),
        (matches_male / num_dates_per_male).mean() * 100.0,
        (matches_female / num_dates_per_female).mean() * 100.0))