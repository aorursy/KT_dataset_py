#Importing libraries and the dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

#Selecting the data with respect to india
dataset_india = dataset[dataset.loc[:,'country_txt'] == "India"]
import numpy as np
#printing count of null values
null_count_array = pd.isna(dataset_india).sum().values
median = np.median(null_count_array)
filter = lambda x : x<10000
filter_vectorise = np.vectorize(filter)
boolean_array = filter_vectorise(null_count_array)

dataset_filtered = dataset_india.iloc[:,boolean_array]
#display(dataset_filtered)
#Replacing remaining null with appropriate Datatype

datatype_int = dataset_filtered.select_dtypes(include='int64')
datatype_float = dataset_filtered.select_dtypes(include='float64')
datatype_object  = dataset_filtered.select_dtypes(include='object')

datatype_int = datatype_int.replace(np.nan,0,regex=True)
datatype_float = datatype_float.replace(np.nan,0.0,regex=True)
#datatype_object = datatype_object.replace(np.nan,"",regex=True)
dataset_india_merged = pd.concat([datatype_int,datatype_float,datatype_object],axis = 1)
dataset_india_merged['provstate'].loc[dataset_india_merged['provstate'] == 'Odisha'] = 'Orissa'
#display(dataset_india_merged)
#Time series analysis
import seaborn as sns
from matplotlib import pyplot as plt

sns.set()

date_wise_count = pd.pivot_table(dataset_india_merged,values = ['eventid'],index = ['iyear'],aggfunc='count')
date_wise_count.reset_index(inplace=True)
date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
date_wise_count.columns  = ['year','attack_count']
plt.figure(figsize=(32,9))
plt.title = 'Time series analysis of terrorist attacks'
sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count)
plt.fill_between(date_wise_count.year.values, date_wise_count.attack_count.values)
plt.show()
#State wise attack count
plt.figure(figsize=(60,15))
sns.set()
plt.title = "Time series analysis of terrorist attacks"
sns.countplot(data=dataset_india_merged,x='provstate')
plt.show()
#State wise attack type
plt.figure(figsize=(5,5))
sns.set(style='darkgrid')
plt.title = "Time series analysis of terrorist attacks"
sns.catplot(data=dataset_india_merged,y='attacktype1_txt',kind='count',col='provstate',col_wrap=6)
plt.show()
#State wise target type
plt.figure(figsize=(5,5))
sns.set(style='darkgrid')
plt.title = "Time series analysis of terrorist attacks"
sns.catplot(data=dataset_india_merged,y='targtype1_txt',kind='count',col='provstate',col_wrap=6)
plt.show()
#Time series combined evolution of state attack with time

states = dataset_india_merged['provstate'].unique()
plt.figure(figsize=(60,12))
plt.title = 'Time series analysis of terrorist attacks'
for x in states:
    temp = dataset_india_merged[dataset_india_merged['provstate'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count,lw=3)
plt.legend(states)
plt.show()
#Attack type analysis
plt.figure(figsize=(60,10))
sns.set(style='darkgrid')
sns.countplot(data = dataset_india_merged,y= 'attacktype1_txt')
plt.show()
# Time series evolution of attack types

#states = dataset_india_merged['provstate'].unique()
plt.figure(figsize=(60,12))

states = dataset_india_merged['attacktype1_txt'].unique()

for x in states:
    temp = dataset_india_merged[dataset_india_merged['attacktype1_txt'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count,lw=3)
plt.legend(states)
plt.show()

#Target type analysis
plt.figure(figsize=(60,10))
sns.set(style='darkgrid')
sns.countplot(data = dataset_india_merged,y= 'targtype1_txt')
plt.show()
# Time series evolution of targets
plt.figure(figsize=(60,12))

states = dataset_india_merged['targtype1_txt'].unique()

for x in states:
    temp = dataset_india_merged[dataset_india_merged['targtype1_txt'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count,lw=3)
plt.legend(states)
plt.show()
#Group responsible for attacks
plt.figure(figsize=(60,65),dpi=150)
sns.countplot(data= dataset_india_merged,y='gname',log=True)
plt.show()
# Top terror groups

terror_groups = pd.pivot_table(dataset_india_merged,index='gname',values='eventid',aggfunc='count')
terror_groups.reset_index(inplace=True)
terror_groups = terror_groups[terror_groups['eventid'] >= 10]
terror_groups.sort_values('eventid',ascending=False,inplace=True)
terror_groups = terror_groups['gname'].head(20).values
# Time series analysis
states = terror_groups
for x in states:
    plt.figure(figsize=(60,10))
    plt.figtext = x
    temp = dataset_india_merged[dataset_india_merged['gname'] == x]
    date_wise_count = pd.pivot_table(temp,values = ['eventid'],index = ['iyear'],aggfunc='count')
    date_wise_count.reset_index(inplace=True)
    date_wise_count.rename_axis({"iyear":"year","eventid":'count'},inplace=True)
    date_wise_count.columns  = ['year','attack_count']
    sns.lineplot(data=date_wise_count,x=date_wise_count.year,y=date_wise_count.attack_count)
    plt.fill_between(date_wise_count.year.values, date_wise_count.attack_count.values)
    display(x)
    plt.show()
    
data_subset = dataset_india_merged.loc[dataset_india_merged['gname'].isin(terror_groups)]
sns.catplot(data=data_subset,y='weaptype1_txt',kind='count',col='gname',col_wrap=6,log = True)
plt.show()
data_subset = dataset_india_merged.loc[dataset_india_merged['gname'].isin(terror_groups)]
sns.catplot(data=data_subset,y='weapsubtype1_txt',kind='count',col='gname',col_wrap=6,log = True)
plt.show()