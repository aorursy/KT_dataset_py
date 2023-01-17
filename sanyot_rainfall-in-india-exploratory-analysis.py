%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
file_indirain = '../input/rainfall in india 1901-2015.csv'

file_distrain = '../input/district wise rainfall normal.csv'



indirain = pd.read_csv(file_indirain).rename(columns=str.lower)

distrain = pd.read_csv(file_distrain).rename(columns=str.lower)
indirain.head()
indirain.info()
subs = indirain['subdivision'].unique()

print (subs)

print ('Total subdivisions: ', len(subs))
unique_state_counts_yearwise = indirain.groupby(by='year')[['subdivision']].count()['subdivision'].value_counts()



print (unique_state_counts_yearwise)
state_count = indirain.groupby(by='subdivision')[['annual']].count().sort_values(by='annual')

print (state_count.head(10))
(indirain.groupby(by='year')[['annual']]

 .sum()

 .plot(figsize=(12, 6), title='Rainfall in India', fontsize=12, legend=False))
(indirain.groupby(by='year')[['annual']]

 .sum()

 .rolling(10)

 .mean()

 .plot(figsize=(12, 6), title='Rolling average (10 years) of rainfall in India', fontsize=12, legend=False)

)
overall = indirain.groupby(by='subdivision').sum()[['annual']].sort_values(by='annual', ascending=False)

overall.head()
highest_sub = overall.index.values[0]

lowest_sub = overall.index.values[-1]

print ('Highest rain in: ', highest_sub)

print ('Lowest rain in: ', lowest_sub)
drop_col = ['annual','jan-feb','mar-may','jun-sep','oct-dec']



fig, ax = plt.subplots()



(indirain.groupby(by='year')

 .sum()

 .drop(drop_col, axis=1)

 .T

 .plot(alpha=0.1, figsize=(12, 6), legend=False, fontsize=12, ax=ax)

)



ax.set_xlabel('Months', fontsize=12)

ax.set_ylabel('Rainfall (in mm)', fontsize=12)
fig = plt.figure(figsize=(18, 9))

plt.xticks(rotation='vertical')

sns.boxplot(x='subdivision', y='annual', data=indirain)