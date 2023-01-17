# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas_profiling

import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
#read the data from a csv file

base_data_df = pd.read_csv("../input/data-scientist-job-market-in-the-us/alldata.csv")
#Finding out the null rows if any 

base_data_df.info()
#finding out the data distribution 

base_data_df.describe(include='all')
#top 20 rows

base_data_df.head(20)
#pandas profiling

pandas_profiling.ProfileReport(base_data_df)
#finding the duplicated rows

base_data_df[base_data_df.duplicated()]
#Make a copy of the base dataset

ds_df = base_data_df.copy()
#drop all na rows

ds_df.dropna(inplace=True)
#replace all na with 0 for reviews

ds_df['reviews'].fillna(0, inplace=True)
#finding the top 40 companies by open positions

from matplotlib.pyplot import figure, show

import matplotlib.pyplot as plt

fig,a=plt.subplots(figsize=(10,6))   

sns.barplot(data=ds_df.groupby('company').count().nlargest(40, 'position').reset_index(), x ='company', y='position')

a.set_xticklabels(ds_df.groupby('company').count().nlargest(40, 'position').reset_index()['company'],rotation=90)   

a.set_ylabel('No Of Positions',fontsize=16,color='red')

a.set_xlabel('Company Name',fontsize=16,color='red') 
#finding the top 40 designations which are in high demand 

from matplotlib.pyplot import figure, show

import matplotlib.pyplot as plt

fig,a=plt.subplots(figsize=(10,6))   

sns.barplot(data=ds_df.groupby('position').count().nlargest(40, 'company').reset_index(), x ='position', y='company')

a.set_xticklabels(ds_df.groupby('position').count().nlargest(40, 'company').reset_index()['position'],rotation=90)   

a.set_ylabel('No Of Positions',fontsize=16,color='red')

a.set_xlabel('Designation',fontsize=16,color='red') 
## Grouping the designations into the following roles

## Data Scientist, Lead Data Scientist, Data Engineer, Data Analyst, Director



## Filtered all the directors from the table

ds_df.loc[ds_df['position'].str.contains('Director', case=False) & ds_df['position'].str.contains('data', case=False), 'designation'] = 'Director'



## Filtered all the Data Scientists from the table

ds_df.loc[ds_df['position'] == 'Data Scientist', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Senior Data Scientist', 'designation'] = 'Lead Data Scientist'

ds_df.loc[ds_df['position'] == 'Sr. Data Scientist', 'designation'] = 'Lead Data Scientist'

ds_df.loc[ds_df['position'] == 'Principal Data Scientist', 'designation'] = 'Lead Data Scientist'

ds_df.loc[ds_df['position'] == 'Lead Data Scientist', 'designation'] = 'Lead Data Scientist'

ds_df.loc[ds_df['position'] == 'Staff Data Scientist', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Junior Data Scientist', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Machine Learning Scientist', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Principal Research Scientist', 'designation'] = 'Lead Data Scientist'

ds_df.loc[ds_df['position'] == 'Data Engineer', 'designation'] = 'Data Engineer'

ds_df.loc[ds_df['position'] == 'Research Analyst', 'designation'] = 'Data Engineer'

ds_df.loc[ds_df['position'] == 'Machine Learning Engineer', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Quantitative Analyst', 'designation'] = 'Data Analyst'

ds_df.loc[ds_df['position'] == 'Research Scientist', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Data Analyst', 'designation'] = 'Data Analyst'

ds_df.loc[ds_df['position'] == 'Market Research Analyst', 'designation'] = 'Data Analyst'

ds_df.loc[ds_df['position'] == 'Senior Research Analyst', 'designation'] = 'Data Analyst'

## Check scientist, Senior Scientist, Senior Applied Scientist Staff Scientist,Applied Scientist, Associate Scientist role description

ds_df.loc[ds_df['position'] == 'Software Engineer, Machine Learning', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Senior Data Engineer', 'designation'] = 'Data Engineer'

ds_df.loc[ds_df['position'] == 'Sr. Data Engineer', 'designation'] = 'Data Engineer'

ds_df.loc[ds_df['position'] == 'Lead Data Engineer', 'designation'] = 'Data Engineer'

ds_df.loc[ds_df['position'] == 'Research Data Analyst', 'designation'] = 'Data Analyst'

ds_df.loc[ds_df['position'] == 'Senior Associate, Data Scientist', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Senior Machine Learning Scientist', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Business Analyst', 'designation'] = 'Data Analyst'

ds_df.loc[ds_df['position'] == 'Data and Machine Learning Technical Trainer, Google Cloud Platform', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'International Research Analyst', 'designation'] = 'Data Analyst'

ds_df.loc[ds_df['position'] == 'Manager, Data Science', 'designation'] = 'Lead Data Scientist'

ds_df.loc[ds_df['position'] == 'Sales Engineer, Machine Learning, Google Cloud', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Data Analysis Manager', 'designation'] = 'Data Engineer'

ds_df.loc[ds_df['position'] == 'Data Science Manager', 'designation'] = 'Lead Data Scientist'

ds_df.loc[ds_df['position'] == 'Data Scientist, Analytics', 'designation'] = 'Data Scientist'

ds_df.loc[ds_df['position'] == 'Machine Learning Data Scientist', 'designation'] = 'Data Scientist'
#drop all the designations that are not related to Data Science

ds_df['designation'].dropna(inplace=True)

#drop all null rows

ds_df.dropna(inplace=True)
# Finding the number of positions for the assigned designations

from matplotlib.pyplot import figure, show

import matplotlib.pyplot as plt

fig,a=plt.subplots(figsize=(10,6))   

sns.countplot(data=ds_df, x ='designation')

a.set_ylabel('No Of Positions',fontsize=16,color='red')

a.set_xlabel('Designation',fontsize=16,color='red') 

## the same plot can be done using 

## ds_df['designation'].value_counts().plot.bar()
ds_df['location'].value_counts().head(30)

# We can observe that different location with similar names are having different vaulues.

#for example New York, NY and New York, NY 10003 both are pointing to the same location hence we need to rename

#all such locations in a standard format
# Renaming all the locations to the standard format

ds_df.loc[ds_df['location'].str.contains('New York', case=False), 'location'] = 'New York'

ds_df.loc[ds_df['location'].str.contains('San Francisco', case=False), 'location'] = 'San Francisco'

ds_df.loc[ds_df['location'].str.contains('San Diego', case=False), 'location'] = 'San Diego'

ds_df.loc[ds_df['location'].str.contains('Atlanta', case=False), 'location'] = 'Atlanta'

ds_df.loc[ds_df['location'].str.contains('Chicago', case=False), 'location'] = 'Chicago'

ds_df.loc[ds_df['location'].str.contains('Seattle', case=False), 'location'] = 'Seattle'

ds_df.loc[ds_df['location'].str.contains('Boston', case=False), 'location'] = 'Boston'

ds_df.loc[ds_df['location'].str.contains('Washington', case=False), 'location'] = 'Washington'

ds_df.loc[ds_df['location'].str.contains('Los Angeles', case=False), 'location'] = 'Los Angeles'

ds_df.loc[ds_df['location'].str.contains('Mountain View', case=False), 'location'] = 'Mountain View'

ds_df.loc[ds_df['location'].str.contains('Cambridge', case=False), 'location'] = 'Cambridge'

ds_df.loc[ds_df['location'].str.contains('Sunnyvale', case=False), 'location'] = 'California'

ds_df.loc[ds_df['location'].str.contains('CA', case=True), 'location'] = 'California'

ds_df.loc[ds_df['location'].str.contains('NJ', case=True), 'location'] = 'New Jersey'

ds_df.loc[ds_df['location'].str.contains('NY', case=True), 'location'] = 'New York'

ds_df.loc[ds_df['location'].str.contains('TX', case=True), 'location'] = 'Texas'

ds_df.loc[ds_df['location'].str.contains('WA', case=True), 'location'] = 'Washington'



# How the jobs are distributed geographically?

from matplotlib.pyplot import figure, show

width=18

height=6

figure(figsize=(width,height))

sns.countplot(data=ds_df, x='location')

# same can be done using: sns.barplot(data=ds_df.groupby(['location'])['position'].count().reset_index(), x='location', y='position')

show()
# Based on the previous graph we have only figured out total open positions at every location

# It would be more intuitive if we know different designated open positions at each location, let us plot that.

# It can be achieved by adding hue to the above countplot

from matplotlib.pyplot import figure, show

width=18

height=6

figure(figsize=(width,height))

sns.countplot(data=ds_df, x='location',hue='designation')

show()
top_20_company_list = ds_df.groupby(['company'])['location'].count().reset_index().sort_values('location',ascending=False).head(20)['company'].tolist()

top_20_company_list
# Based on the previous graph we have only figured out total open positions at every location

# It would be more intuitive if we know different designated open positions at each location, let us plot that.

# It can be achieved by adding hue to the above countplot

from matplotlib.pyplot import figure, show

import matplotlib.pyplot as plt

fig,a=plt.subplots(figsize=(18,10))   

ax = sns.countplot(data=ds_df[ds_df['company'].isin(top_20_company_list)], x='company',hue='designation')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
df_final =  pd.DataFrame(ds_df.groupby(['company','location','designation'])['position'].count().reset_index())





from matplotlib.pyplot import figure, show

import matplotlib.pyplot as plt

fig,a=plt.subplots(figsize=(3,3))   

ax = sns.countplot(data=df_final[df_final['company'] == 'KPMG'], x='location',hue='designation')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
fig,a=plt.subplots(figsize=(3,3))  

ax = sns.countplot(data=df_final[df_final['company'] == 'Amazon.com'], x='location',hue='designation')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
fig,a=plt.subplots(figsize=(3,3))  

ax = sns.countplot(data=df_final[df_final['company'] == 'Google'], x='location',hue='designation')

ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")

plt.tight_layout()

plt.show()