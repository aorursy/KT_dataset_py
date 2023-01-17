# Imports

import os

import numpy as np

import pandas as pd



# Locating .csv files

path = '../input/silkboard-bangalore-ambient-air-covid19lockdown/'

csv = os.listdir(path)

csv
# Creating list of all files name

files = {'SO2':0, 'PM10':0, 'CO':0, 'PM25':0, 'O3':0, 'NO2':0}



# Reading all .csv files

count = 0

for key in files.keys():

    files[key] = pd.read_csv(path + csv[count], index_col=0)

    count += 1   
# Lets look at any one dataset 

files['PM25'].head(3)
# Info

files['PM25'].info()   # Note that every other dataset has same format 
# Adding a new column 'Date' that represents date for each file

for key in files.keys():

    files[key]['Date'] = pd.to_datetime(files[key]['local']).apply(lambda x: x.date().strftime('%d-%m-%Y'))
# Lets look at the unique terms and attributes for each dataset

for key in files.keys():

    print('Gas : {}'.format(key))

    print(files[key].nunique())

    print()
files['PM25'].head(3)
# Imports 

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cff

init_notebook_mode(connected=True)

cff.go_offline()

%matplotlib inline
# Creating an empty dataframe that will hold all mean values for one particular date

df = pd.DataFrame(columns=['SO2', 'PM10', 'CO', 'PM25', 'O3', 'NO2'])

df
# Replacing old files with new one that holds date as index & value as columns

for col in df.columns:

    x = pd.DataFrame(data=np.array(files[col]['value']), index=files[col]['Date'],columns=[col])

    x = pd.DataFrame(x.groupby(by=x.index)[col].mean())

    files[str(col)] = x

    

# Concating all files into single dataframe

df = pd.concat([files[key] for key in files.keys()], axis=1)

df = df.dropna()

df.index.name = 'Date'

df.head()
# Dataframe info

df.info()
# Adding a day & month column

df['Day'] = [date.split('-')[0] for date in df.index]

df['Month'] = [date.split('-')[1] for date in df.index]



# Sorting dataframe by month and day

df = df.sort_values(by=['Month', 'Day'])

df.head()
# Visualizing Gases Concentration per Day using Line Plot

df.drop(['Day', 'Month'], axis=1).iplot(kind='line', title='Gases Concentration per Day')
fig = df[['SO2', 'NO2', 'O3', 'PM10', 'CO']].iplot(kind='line', title='Gases Concentration per Day (excluding PM25)')
# Mean Concentration of Gases per Month

df_2 = df.groupby(by='Month').mean()



# Replacing index with month names

df_2['Month'] = ['February', 'March', 'April', 'May', 'June']

df_2.set_index('Month', inplace=True)

df_2
# Relation between Gases wrt df_2

sns.heatmap(data=df_2.corr(), cmap='pink_r',linewidth=1, linecolor='white', annot=True)
# Visualizing Gases Concentration per Month using Line plot

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,14))



sns.lineplot(x=df_2.index, y=df_2['SO2'], ax=axes[0, 0], sort=False, marker='o')

sns.lineplot(x=df_2.index, y=df_2['PM10'], ax=axes[0, 1], sort=False, marker='o')

sns.lineplot(x=df_2.index, y=df_2['CO'], ax=axes[1, 0], sort=False, marker='o')

sns.lineplot(x=df_2.index, y=df_2['PM25'], ax=axes[1, 1], sort=False, marker='o')

sns.lineplot(x=df_2.index, y=df_2['O3'], ax=axes[2, 0], sort=False, marker='o')

sns.lineplot(x=df_2.index, y=df_2['NO2'], ax=axes[2, 1], sort=False, marker='o')
# Pie Plot representing Gases Concentration per Month in percentage



fig, axes = plt.subplots(3, 2, figsize=(22,15))



fig1 = axes[0, 0].pie(df_2['SO2'], labels=df_2.index, autopct='%1.1f%%', startangle=90)

axes[0, 0].set_title('SO2 Concentration per Month')



fig2 = axes[0, 1].pie(df_2['PM10'], labels=df_2.index, autopct='%1.1f%%', startangle=90)

axes[0, 1].set_title('PM10 Concentration per Month')



fig3 = axes[1, 0].pie(df_2['CO'], labels=df_2.index, autopct='%1.1f%%', startangle=90)

axes[1, 0].set_title('CO Concentration per Month')



fig4 = axes[1, 1].pie(df_2['PM25'], labels=df_2.index, autopct='%1.1f%%', startangle=90)

axes[1, 1].set_title('PM25 Concentration per Month')



fig5 = axes[2, 0].pie(df_2['O3'], labels=df_2.index, autopct='%1.1f%%', startangle=90)

axes[2, 0].set_title('O3 Concentration per Month')



fig6 = axes[2, 1].pie(df_2['NO2'], labels=df_2.index, autopct='%1.1f%%', startangle=90)

axes[2, 1].set_title('NO2 Concentration per Month')
