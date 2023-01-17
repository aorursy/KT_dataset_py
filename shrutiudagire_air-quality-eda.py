import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
#As the date is of object type we need to cnvert it to datetime format hence will reading the data:
abc=pd.read_csv("../input/airquality/data.csv", parse_dates=['date'])
abc.info()
abc['type'].value_counts()
#as there are alot of duplicate types
#cleaning type column and should have only four columns Industrial,Residential,Sensitive and RIRUO
#Updating the changes to abc data frame
abc.loc[(abc['type']=="Residential, Rural and other Areas"),'type']='Residential'
abc.loc[(abc['type']=="Residential and others"),'type']='Residential'
abc.loc[(abc['type']=="Industrial Area"),'type']='Industrial'
abc.loc[(abc['type']=="Industrial Areas"),'type']='Industrial'
abc.loc[(abc['type']=="Sensitive Area"),'type']='Sensitive'
abc.loc[(abc['type']=="Sensitive Areas"),'type']='Sensitive'
abc['type'].value_counts()
#Filling missing values for rspm and spm hence grouping by location and type
grp_location=abc.groupby(['location','type'])
dict_grp_location=dict(list(grp_location))
# dict_grp_location
print(abc['rspm'].isnull().sum())
print(abc['spm'].isnull().sum())
#Forward filling
grouped_location=pd.DataFrame()
for key in dict_grp_location:
    df1=dict_grp_location[key].sort_values(by='date')
    df1['rspm'].fillna(method='ffill',inplace=True)
    df1['spm'].fillna(method='ffill',inplace=True)
    grouped_location=pd.concat([grouped_location,df1])
print(grouped_location['rspm'].isnull().sum())
print(grouped_location['spm'].isnull().sum())

#Initially we have grouped by 'location' and 'type' and then did foward fill but some values were not filled hence backward fill
backwardfill=grouped_location.groupby(['location','type'])
backwardfill=dict(list(backwardfill))
backwardfill
grouped_location=pd.DataFrame()
for key in backwardfill:
    df2=backwardfill[key].sort_values(by='date')
    df2['rspm'].fillna(method='bfill',inplace=True)
    df2['spm'].fillna(method='bfill',inplace=True)
    grouped_location=pd.concat([grouped_location,df2])
print(grouped_location['rspm'].isnull().sum())
print(grouped_location['spm'].isnull().sum())
#now we are grouping it on larger scale that is 'state' and thn by 'type' so as to fill null values
dict_grouped_state=dict(list(grouped_location.groupby(['state','type'])))

grouped_state=pd.DataFrame()
for key in dict_grouped_state:
    df3=dict_grouped_state[key]
    df3['rspm'].fillna(df3['rspm'].median(),inplace=True)
    df3['spm'].fillna(df3['spm'].median(),inplace=True)
    grouped_state=pd.concat([grouped_state,df3])
print(grouped_state['spm'].isnull().sum())
print(grouped_state['rspm'].isnull().sum())
#Now we are grouping by 'type' and replacimg all remaining nan values
grouped_type=grouped_state.groupby('type').median()
grouped_type
dataframe=grouped_state
dataframe.loc[(dataframe['type']=='Industrial') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['Industrial']
dataframe.loc[(dataframe['type']=='RIRUO') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['RIRUO']
dataframe.loc[(dataframe['type']=='Residential') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['Residential']
dataframe.loc[(dataframe['type']=='Sensitive') & (dataframe['rspm'].isnull()),'rspm']=grouped_type['rspm']['Sensitive']

dataframe.loc[(dataframe['type']=='Industrial') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['Industrial']
dataframe.loc[(dataframe['type']=='RIRUO') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['RIRUO']
dataframe.loc[(dataframe['type']=='Residential') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['Residential']
dataframe.loc[(dataframe['type']=='Sensitive') & (dataframe['spm'].isnull()),'spm']=grouped_type['spm']['Sensitive']
print(dataframe['rspm'].isnull().sum())
print(dataframe['spm'].isnull().sum())
#adding a new 'year' column from 'date' column
dataframe['year']=dataframe['date'].dt.year
print(dataframe['year'].isnull().sum())
#filling null values in year by either doing forward fill or backwadr fill
dataframe['year']=dataframe['year'].fillna(method='ffill')
print(dataframe['year'].isnull().sum())
dataframe['year']=dataframe['year'].astype(int)
#ploting states in descending order as per the level of spm
dataframe
state=dataframe.groupby('state').median()
state=state[['rspm','spm']]
state=state.sort_values(by='spm',ascending=False)
state.plot(kind='bar',figsize=(15,10))
# potting a graph in  descending order as per the level of spm
state.sort_values(by='rspm',ascending=False).plot(kind='bar',figsize=(15,10))
states=state.reset_index().head(5)
top_five_states=states['state']
for i in top_five_states:
    print(i)
group_by_state=dict(list(dataframe.groupby('state')))
plot_five_states=pd.DataFrame()
for i in top_five_states:
    df=group_by_state[i][['state','location','spm','rspm','type']]
    plot_five_states=pd.concat([plot_five_states,df])
plot_five_states
plot_five_states=plot_five_states.groupby(['state','location','type']).median()
plot_five_states
plt.figure(figsize = (20,20))
plt.subplot(3,2,1)
plt.title('Delhi')
a = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Delhi'].reset_index())
a.set(ylim = (0,600))
plt.subplot(3,2,2)
plt.title('Haryana')
b = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Haryana'].reset_index())
b.set(ylim = (0,600))
plt.subplot(3,2,3)
plt.title('Rajasthan')
c = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Rajasthan'].reset_index())
c.set(ylim = (0,600))
plt.subplot(3,2,4)
plt.title('Uttarakhand')
d = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Uttarakhand'].reset_index())
d.set(ylim = (0,600))
plt.subplot(3,2,5)
plt.title('Uttar Pradesh')
plt.xticks(rotation = 90)
g = sns.barplot(x = 'location',y = 'spm',hue = 'type',data = plot_five_states.loc['Uttar Pradesh'].reset_index())
g.set(ylim = (0,600))
states_year=dataframe.groupby(['state','year']).median()['spm']
states_year
states_year=states_year.reset_index()
states_year['spm'].isnull().sum()
pivot=pd.pivot_table(states_year,values='spm',index='state',columns='year')
pivot.fillna(0,inplace= True)
pivot
plt.figure(figsize=(20,20))
sns.heatmap(data=pivot,annot=True)
#Finding the null values for so2 and no2 for karnataka state and the replacing them with median values.
#1)SO2
karnataka=abc.groupby(['state','type'])
a=dict(list(karnataka))
kar_ind=a[('Karnataka','Industrial')]
print(kar_ind['so2'].isnull().sum())
kar_res=a[('Karnataka','Residential')]
print(kar_res['so2'].isnull().sum())
kar_sensitive=a[('Karnataka','Sensitive')]
print(kar_sensitive['so2'].isnull().sum())
# Karnataka has no RIRUO
# kar_riruo=a[('Karnataka','RIRUO')]
# kar_riruo['so2'].isnull().sum()
#now replacing all these null values of So2  with median values  in copy_abc data frame
copy_abc=abc.copy()

copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Industrial') & (copy_abc['so2'].isnull()),'so2']=kar_ind.median()['so2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Residential') & (copy_abc['so2'].isnull()),'so2']=kar_res.median()['so2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Sensitive') & (copy_abc['so2'].isnull()),'so2']=kar_sensitive.median()['so2']

# Checking if null values are imputed S02 
karnataka=copy_abc.groupby(['state','type'])
a=dict(list(karnataka))
print(a[('Karnataka','Industrial')]['so2'].isnull().sum())
print(a[('Karnataka','Residential')]['so2'].isnull().sum())
print(a[('Karnataka','Sensitive')]['so2'].isnull().sum())
#2)Replacing below null values for no2 state of karnataka with median in copy_abc
#No2
kar_ind=a[('Karnataka','Industrial')]
kar_ind['no2'].isnull().sum()
kar_ind=a[('Karnataka','Residential')]
kar_ind['no2'].isnull().sum()
kar_sensitive=a[('Karnataka','Sensitive')]
kar_sensitive['no2'].isnull().sum()
# Karnataka has no RIRUO
# kar_riruo=a[('Karnataka','RIRUO')]
# kar_riruo['no2'].isnull().sum()
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Industrial') & (copy_abc['no2'].isnull()),'no2']=kar_ind.median()['no2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Residential') & (copy_abc['no2'].isnull()),'no2']=kar_res.median()['no2']
copy_abc.loc[(copy_abc['state']=='Karnataka') & (copy_abc['type']=='Sensitive') & (copy_abc['no2'].isnull()),'no2']=kar_sensitive.median()['no2']

# Checking if null values are imputed No2
karnataka=copy_abc.groupby(['state','type'])
a=dict(list(karnataka))
print(a[('Karnataka','Industrial')]['no2'].isnull().sum())
print(a[('Karnataka','Residential')]['no2'].isnull().sum())
print(a[('Karnataka','Sensitive')]['no2'].isnull().sum())
#Now we are grouping the data by just Bangalore state as we have to draw graph of no2 and so2
bangalore=copy_abc.groupby('location')
bangalore=dict(list(bangalore))
bangalore=bangalore['Bangalore'][['so2','no2','date']]
bangalore['year']=bangalore['date'].dt.year
bangalore=bangalore[['so2','no2','year']]
bangalore['year'].isnull().sum()
#as we have one null value for year field we have to do forward fill or backward fill.Performing 1st backward fill
bangalore['year']=bangalore['year'].fillna(method='bfill')
bangalore['year']=bangalore['year'].astype('int')
bangalore['year'].isnull().sum()
bangalore=bangalore.groupby('year').median()
bangalore=bangalore.reset_index()
bangalore
plt.figure(figsize=(15,5))
plt.xticks(np.arange(1980,2016))
plt.yticks(np.arange(5,55,5))
sns.pointplot(bangalore['year'],bangalore['so2'],color='r')
sns.pointplot(bangalore['year'],bangalore['no2'],color='g')
plt.legend(['so2','no2'])

