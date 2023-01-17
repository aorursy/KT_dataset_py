# For autoreloading modules
%load_ext autoreload
%autoreload 2
# For notebook plotting
%matplotlib inline

# Standard libraries
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pdpbox import pdp
from plotnine import *
from pandas_summary import DataFrameSummary
from IPython.display import display
from datetime import datetime
KAGGLE_DIR = '../input/'
data = pd.read_csv(KAGGLE_DIR + 'sc2-matches-history.csv')
print('First 5 rows: ')
display(data.head())

print('Last 5 rows: ')
display(data.tail())
data.describe()
display(data[data['player_2'].isnull()])
data.drop(85860, inplace=True)
display(data[data['player_2'].isnull()])
all_data = data[(data['player_1']=='Bly') | (data['player_1']=='Kas')]

display(all_data.head(10))

display(all_data.tail(10))
all_data = all_data[all_data['addon']=='LotV']
all_data.dtypes
all_data['match_date'] =pd.to_datetime(all_data['match_date'],dayfirst=False)
all_data.dtypes
data_df = (all_data.melt('player_1')
       .groupby(['player_1','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1))
data_df['player_1_match_status'].plot(kind='bar', stacked=True)
all_state_pcts = (all_data.melt('player_1_match_status')
       .groupby(['player_1_match_status','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1)).apply(lambda x:
                                                 100 * x / float(x.sum()))
all_state_pcts = all_state_pcts['player_1'].transpose()
all_state_pcts = all_state_pcts[['[winner]','[loser]']]
all_state_pcts.plot(kind='bar', stacked=True)
data_df_5 = (all_data.groupby('player_1').head(5).melt('player_1')
       .groupby(['player_1','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1))
data_df_5['player_1_match_status'].plot(kind='bar', stacked=True)
bly_opposing_race = all_data[(all_data['player_1']=='Bly') & (all_data['player_2_race']=='T')]

display(bly_opposing_race.head(10))
bly_pcts = (bly_opposing_race.melt('player_1_match_status')
       .groupby(['player_1_match_status','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1)).apply(lambda x:
                                                 100 * x / float(x.sum()))
bly_pcts['player_1'].plot.pie(y='Bly', autopct='%1.1f%%',figsize=(7, 7))
bly_pcts_5 = (bly_opposing_race.groupby('player_1').head(5).melt('player_1_match_status')
       .groupby(['player_1_match_status','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1)).apply(lambda x:
                                                 100 * x / float(x.sum()))
bly_pcts_5['player_1'].plot.pie(y='Bly', autopct='%1.1f%%',figsize=(7, 7))
kas_opposing_race = all_data[(all_data['player_1']=='Kas') & (all_data['player_2_race']=='Z')]

display(kas_opposing_race.head(10))
kas_pcts = (kas_opposing_race.melt('player_1_match_status')
       .groupby(['player_1_match_status','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1)).apply(lambda x:
                                                 100 * x / float(x.sum()))
kas_pcts['player_1'].plot.pie(y='Kas', autopct='%1.1f%%',figsize=(7, 7))
kas_pcts_5 = (kas_opposing_race.groupby('player_1').head(5).melt('player_1_match_status')
       .groupby(['player_1_match_status','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1)).apply(lambda x:
                                                 100 * x / float(x.sum()))
kas_pcts_5['player_1'].plot.pie(y='Kas', autopct='%1.1f%%',figsize=(7, 7))
against_data = all_data[(all_data['player_1']=='Bly') &(all_data['player_2']=='Kas')]
df = (against_data.melt('player_1_match_status')
       .groupby(['player_1_match_status','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1))
state_pcts = df['player_1'].groupby(level=0).apply(lambda x:
                                                 100 * x / float(df['player_1'].sum()))
state_pcts.plot.pie(y='Bly', autopct='%1.1f%%',figsize=(7, 7))
df_5 = (against_data.nlargest(5, 'match_date').melt('player_1_match_status')
       .groupby(['player_1_match_status','variable'])['value']
       .value_counts()
       .unstack([1,2], fill_value=0)
       .rename_axis((None, None), 1))
state_pcts_5 = df_5['player_1'].groupby(level=0).apply(lambda x:
                                                 100 * x / float(df_5['player_1'].sum()))
state_pcts_5.plot.pie(y='Bly', autopct='%1.1f%%', figsize=(7, 7))
