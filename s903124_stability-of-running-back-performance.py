import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-expected-yards/train_expected.csv')
data.columns 
data_17 = data[data.GameId.astype(str).str[:4] =='2017']

data_18 = data[data.GameId.astype(str).str[:4] =='2018']
qualify_17 = data_17.groupby('DisplayName').expectedYards.count()[(data_17.groupby('DisplayName').expectedYards.count() > 20)].reset_index().DisplayName

qualify_18 = data_18.groupby('DisplayName').expectedYards.count()[(data_18.groupby('DisplayName').expectedYards.count() > 20)].reset_index().DisplayName
rusher_17 = data_17[data_17.DisplayName.isin(qualify_17)].groupby('DisplayName')['Yards','expectedYards','Dis'].mean().reset_index()

rusher_17['yards_over_expected_17'] = rusher_17.Yards - rusher_17.expectedYards

rusher_18 = data_18[data_18.DisplayName.isin(qualify_18)].groupby('DisplayName')['Yards','expectedYards','Dis'].mean().reset_index()

rusher_18['yards_over_expected_18'] = rusher_18.Yards - rusher_18.expectedYards
rusher_17.expectedYards
rusher_df = pd.merge(rusher_17[['DisplayName','yards_over_expected_17']],rusher_18[['DisplayName','yards_over_expected_18']])
sns.regplot(rusher_df.yards_over_expected_17,rusher_df.yards_over_expected_18)

plt.title('Yards over expected for runningback with 20+ carries, 17-18')

plt.xlabel('Yards over expected 2017')

plt.ylabel('Yards over expected 2018')
rusher_speed = pd.merge(rusher_17[['DisplayName','Dis']],rusher_18[['DisplayName','Dis']],on='DisplayName',how='inner')

rusher_speed.columns = ['DisplayName','Dis_17','Dis_18']
sns.regplot(rusher_speed.Dis_17*10,rusher_speed.Dis_18*10)

plt.title('Speed at handoff for runningback with 20+ carries, 17-18')

plt.xlabel('Speed in 2017 season (yards/s)')

plt.ylabel('Speed in 2018 season (yards/s)')
rusher_expected = pd.merge(rusher_17[['DisplayName','expectedYards']],rusher_18[['DisplayName','expectedYards']],on='DisplayName',how='inner')

rusher_expected.columns = ['DisplayName','expected_17','expected_18']
sns.regplot(rusher_expected.expected_17,rusher_expected.expected_18)

plt.title('Expected yards for runningback with 20+ carries, 17-18')

plt.xlabel('Expected yards in 2017 season')

plt.ylabel('Expected yards in 2018 season ')