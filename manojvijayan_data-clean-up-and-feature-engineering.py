# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from haversine import haversine # finds out the distance between two places
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pd.set_option('display.max_colwidth', -1)

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/world-cities-database/worldcitiespop.csv")
df= df[df.Country == 'in']
df_core = pd.read_csv("../input/the-interview-attendance-problem/Interview.csv")
df_core.head()
renamed   =  {'Are you clear with the venue details and the landmark.':'Knows venue' ,                                                
              'Have you taken a printout of your updated resume. Have you read the JD and understood the same':'Resume printed',        
              'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much':'Alternative number',      
              'Can I Call you three hours before the interview and follow up on your attendance for the interview': 'Call three hours before',    
              'Hope there will be no unscheduled meetings':'Unscheduled meetings',                                                           
              'Has the call letter been shared':'Call letter shared',                                                                       
              'Have you obtained the necessary permission to start at the required time':'Permissions'}
df_core.rename(renamed, axis='columns', inplace=True)
plt.figure(figsize=(10,10))
(df_core.isnull().sum()/df_core.isnull().count()).sort_values(ascending=True).plot(kind='barh')
series = df_core.isnull().sum()/df_core.isnull().count()
df_core.drop(series[series == 1].index, axis=1, inplace=True)
(df_core.isnull().sum()/df_core.isnull().count()).sort_values(ascending=False)
df_core.dropna(subset=['Observed Attendance'], inplace = True)
df_core['Knows venue'].unique()
df_core['Knows venue'].fillna('unknown', inplace=True)
df_core['Knows venue'] = df_core['Knows venue'].apply(lambda x: x.lower())
df_core.loc[(df_core['Knows venue'].isin(['no- i need to check','na']), 'Knows venue')] = 'no'
df_core['Knows venue'].unique()
df_core['Resume printed'].unique()
df_core['Resume printed'].fillna('unknown', inplace=True)
df_core['Resume printed'] = df_core['Resume printed'].apply(lambda x: x.lower())
df_core.loc[df_core['Resume printed'].isin(['no- will take it soon', 'not yet', 'na']), 'Resume printed'] = 'no'
df_core['Resume printed'].unique()
df_core['Alternative number'].unique()
df_core['Alternative number'].fillna('unknown', inplace=True)
df_core['Alternative number'] = df_core['Alternative number'].apply(lambda x: x.lower())
df_core.loc[df_core['Alternative number'].isin(['no i have only thi number', 'not yet', 'na']), 'Alternative number'] = 'no'
df_core['Alternative number'].unique()
df_core['Call three hours before'].unique()
df_core['Call three hours before'].fillna('unknown', inplace=True)
df_core['Call three hours before'] = df_core['Call three hours before'].apply(lambda x: x.lower())
df_core.loc[df_core['Call three hours before'].isin(['no dont', 'na']), 'Call three hours before'] = 'no'
df_core['Call three hours before'].unique()
df_core['Unscheduled meetings'].unique()
df_core['Unscheduled meetings'].fillna('unknown', inplace=True)
df_core['Unscheduled meetings'] = df_core['Unscheduled meetings'].apply(lambda x: x.lower())
df_core.loc[df_core['Unscheduled meetings'].isin(['not sure', 'cant say']), 'Unscheduled meetings'] = 'uncertain'
df_core.loc[df_core['Unscheduled meetings'].isin(['na']), 'Unscheduled meetings'] = 'no'
df_core['Unscheduled meetings'].unique()
df_core['Call letter shared'].unique()
df_core['Call letter shared'].fillna('unknown', inplace=True)
df_core['Call letter shared'] = df_core['Call letter shared'].apply(lambda x: x.lower())
df_core.loc[df_core['Call letter shared'].isin(['havent checked', 'need to check', 'yet to check', 'not sure']), 'Call letter shared'] = 'uncertain'
df_core.loc[df_core['Call letter shared'].isin(['not yet', 'na']), 'Call letter shared'] = 'no'
df_core['Call letter shared'].unique()
df_core['Permissions'].unique()
df_core['Permissions'].fillna('unknown', inplace=True)
df_core['Permissions'] = df_core['Permissions'].apply(lambda x: x.lower())
df_core.loc[df_core['Permissions'].isin(['yet to confirm']), 'Permissions'] = 'uncertain'
df_core.loc[df_core['Permissions'].isin(['not yet', 'na']), 'Permissions'] = 'no'
df_core['Permissions'].unique()
df_core['Expected Attendance'].unique()
df_core['Expected Attendance'].fillna('unknown', inplace=True)
df_core['Expected Attendance'] = df_core['Expected Attendance'].apply(lambda x: x.lower())
df_core.loc[df_core['Expected Attendance'].isin(['11:00 am', '10.30 am']), 'Expected Attendance'] = 'yes'
df_core['Expected Attendance'].unique()
df_core['Marital Status'].unique()
df_core['Interview Type'].unique()
df_core['Interview Type'].fillna('unknown', inplace=True)
df_core['Interview Type'] = df_core['Interview Type'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Interview Type'].isin(['sceduled walkin', 'scheduled walk in']), 'Interview Type'] = 'scheduled walkin'
df_core['Interview Type'].unique()
df_core['Client name'].unique()
df_core['Client name'] = df_core['Client name'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Client name'].isin(['standard chartered bank chennai']), 'Client name'] = 'standard chartered bank'
df_core.loc[df_core['Client name'].isin(['aon hewitt', 'aon hewitt gurgaon']), 'Client name'] = 'hewitt'
df_core['Client name'].unique()
df_core['Industry'].unique()
pd.Series(df_core['Industry'] + ' ' + df_core['Client name']).value_counts()
df_core['Industry'] = df_core['Industry'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Client name'].isin(['hewitt', 'ust', 'williams lea']), 'Industry'] = 'it products and services'
pd.Series(df_core['Industry'] + ' ' + df_core['Client name']).value_counts()
df_core['Location'].unique()
df_core['Location'] = df_core['Location'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Location'].isin(['gurgaonr']), 'Location'] = 'gurgaon'
df_core.loc[df_core['Location'].isin(['- cochin-']), 'Location'] = 'cochin'
df_core['Location'].unique()
set(df_core.Location).difference(set(df.City))
df_core['Position to be closed'].unique()
df_core['Nature of Skillset'].unique()
df_core['Nature of Skillset'].nunique()
df_core['Nature of Skillset'] = df_core['Nature of Skillset'].apply(lambda x: x.strip().lower())
import fuzzywuzzy.process
choices = df_core['Nature of Skillset'].unique()
for each in choices:
    print(fuzzywuzzy.process.extract(each, choices, limit=4))
df_core['Nature of Skillset'] = df_core['Nature of Skillset'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Nature of Skillset'].isin(['cdd kyc']), 'Nature of Skillset'] = 'aml/kyc/cdd'
df_core.loc[df_core['Nature of Skillset'].isin(['biosimiliars', 'biosimillar']), 'Nature of Skillset'] = 'biosimilars'
df_core.loc[df_core['Nature of Skillset'].isin(['lending & liability', 'lending and liabilities', 'lending&liablities']), 'Nature of Skillset'] = 'l & l'
df_core.loc[df_core['Nature of Skillset'].isin(['tech lead- mednet']), 'Nature of Skillset'] = 'tech lead-mednet'
df_core.loc[df_core['Nature of Skillset'].isin(['java j2ee', 'java,j2ee', 'java ,j2ee']), 'Nature of Skillset'] = 'java/j2ee'
df_core.loc[df_core['Nature of Skillset'].isin(['java, spring, hibernate', 'java,spring,hibernate']), 'Nature of Skillset'] = 'java/spring/hibernate/jsf'
df_core.loc[df_core['Nature of Skillset'].isin(['java, sql']), 'Nature of Skillset'] = 'java,sql'
df_core.loc[df_core['Nature of Skillset'].isin(['analytical r & d']), 'Nature of Skillset'] = 'analytical r&d'
df_core.loc[df_core['Nature of Skillset'].isin(['11.30 am', '10.00 am', '9.00 am', '12.30 pm', '9.30 am']), 'Nature of Skillset'] = 'unknown'     
df_core['Nature of Skillset'].nunique()
df_core['Interview Venue'].unique()
df_core['Interview Venue'] = df_core['Interview Venue'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Interview Venue'].isin(['- cochin-']), 'Interview Venue'] = 'cochin'
df_core['Interview Venue'].unique()
set(df_core['Interview Venue']).difference(set(df.City))
df_core['Name(Cand ID)'].nunique()
df_core.drop('Name(Cand ID)', inplace=True, axis = 'columns')
df_core['Gender'].unique()
df_core['Candidate Current Location'].unique()
df_core['Candidate Current Location'] = df_core['Candidate Current Location'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Candidate Current Location'].isin(['- cochin-']), 'Candidate Current Location'] = 'cochin'
df_core['Candidate Current Location'].unique()
set(df_core['Candidate Current Location']).difference(set(df.City))
df_core['Candidate Job Location'].unique()
df_core['Candidate Job Location'] = df_core['Candidate Job Location'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Candidate Job Location'].isin(['- cochin-']), 'Candidate Job Location'] = 'cochin'
df_core.loc[df_core['Candidate Job Location'].isin(['visakapatinam']), 'Candidate Job Location'] = 'visakhapatnam'
df_core['Candidate Job Location'].unique()
set(df_core['Candidate Job Location']).difference(set(df.City))
df_core['Observed Attendance'].unique()
df_core['Observed Attendance'] = df_core['Observed Attendance'].apply(lambda x: x.strip().lower())
df_core['Observed Attendance'].unique()
df_core['Candidate Native location'].unique()
df_core['Candidate Native location'] = df_core['Candidate Native location'].apply(lambda x: x.strip().lower())
df_core.loc[df_core['Candidate Native location'].isin(['- cochin-']), 'Candidate Native location'] = 'cochin'
df_core.loc[df_core['Candidate Native location'].isin(['delhi /ncr']), 'Candidate Native location'] = 'delhi'
df_core.loc[df_core['Candidate Native location'].isin(['visakapatinam']), 'Candidate Native location'] = 'visakhapatnam'
df_core.loc[df_core['Candidate Native location'].isin(['chitoor']), 'Candidate Native location'] = 'chittoor'
df_core['Candidate Native location'].unique()
set(df_core['Candidate Native location']).difference(set(df.City))
df_core['Date of Interview'].unique()
df_core.loc[df_core['Date of Interview'].isin(['13.02.2015']), 'Date of Interview'] = '2015-02-13'
df_core.loc[df_core['Date of Interview'].isin(['19.06.2015']), 'Date of Interview'] = '2015-06-19'
df_core.loc[df_core['Date of Interview'].isin(['23.06.2015']), 'Date of Interview'] = '2015-06-23'
df_core.loc[df_core['Date of Interview'].isin(['29.06.2015']), 'Date of Interview'] = '2015-06-29'
df_core.loc[df_core['Date of Interview'].isin(['25.06.2015']), 'Date of Interview'] = '2015-06-25'
df_core.loc[df_core['Date of Interview'].isin(['25.06.2015','25.05.16', '25.5.2016', '25-05-2016']), 'Date of Interview'] = '2015-06-25'
df_core.loc[df_core['Date of Interview'].isin(['25.05.2016','25-5-2016']), 'Date of Interview'] = '2015-06-25'
df_core.loc[df_core['Date of Interview'].isin(['04/12/16']), 'Date of Interview'] = '2016-04-12'
df_core.loc[df_core['Date of Interview'].isin(['13.04.2016']), 'Date of Interview'] = '2016-04-13'
df_core.loc[df_core['Date of Interview'].isin(['27.02.2016']), 'Date of Interview'] = '2016-02-27'
df_core.loc[df_core['Date of Interview'].isin(['07.05.2016']), 'Date of Interview'] = '2016-05-07'
df_core.loc[df_core['Date of Interview'].isin(['5.5.16']), 'Date of Interview'] = '2016-05-05'
df_core.loc[df_core['Date of Interview'].isin(['4.5.16']), 'Date of Interview'] = '2016-05-04'
df_core.loc[df_core['Date of Interview'].isin(['21.4.16']), 'Date of Interview'] = '2016-04-21'
df_core.loc[df_core['Date of Interview'].isin(['22.4.16']), 'Date of Interview'] = '2016-04-22'
df_core.loc[df_core['Date of Interview'].isin(['23.4.16']), 'Date of Interview'] = '2016-04-23'
df_core.loc[df_core['Date of Interview'].isin(['15 Apr 16']), 'Date of Interview'] = '2016-04-15'
df_core.loc[df_core['Date of Interview'].isin(['19 Apr 16']), 'Date of Interview'] = '2016-04-19'
df_core.loc[df_core['Date of Interview'].isin(['20 Apr 16']), 'Date of Interview'] = '2016-04-20'
df_core.loc[df_core['Date of Interview'].isin(['21-Apr -16']), 'Date of Interview'] = '2016-04-21'
df_core.loc[df_core['Date of Interview'].isin(['22 -Apr -16']), 'Date of Interview'] = '2016-04-22'
df_core.loc[df_core['Date of Interview'].isin(['25 – Apr-16', '25 Apr 16']), 'Date of Interview'] = '2016-04-25'
df_core.loc[df_core['Date of Interview'].isin(['18 Apr 16']), 'Date of Interview'] = '2016-04-18'
df_core.loc[df_core['Date of Interview'].isin(['11.5.16']), 'Date of Interview'] = '2016-05-11'
df_core.loc[df_core['Date of Interview'].isin(['10.5.16']), 'Date of Interview'] = '2016-05-10'
df_core.loc[df_core['Date of Interview'].isin(['11.05.16']), 'Date of Interview'] = '2016-05-11'
df_core.loc[df_core['Date of Interview'].isin(['12.04.2016','12.04.2017','12.04.2018','12.04.2019']), 'Date of Interview'] = '2016-04-12'
df_core.loc[df_core['Date of Interview'].isin(['12.04.2020','12.04.2021','12.04.2022','12.04.2023']), 'Date of Interview'] = '2016-04-12'
df_core.loc[df_core['Date of Interview'].isin(['8.5.16']), 'Date of Interview'] = '2016-05-08'
df_core.loc[df_core['Date of Interview'].isin(['7.5.16']), 'Date of Interview'] = '2016-05-07'
df_core.loc[df_core['Date of Interview'].isin(['19.03.16']), 'Date of Interview'] = '2016-03-19'
df_core.loc[df_core['Date of Interview'].isin(['24.05.2016']), 'Date of Interview'] = '2016-05-24'
df_core.loc[df_core['Date of Interview'].isin(['05/11/2016']), 'Date of Interview'] = '2016-05-11'
df_core.loc[df_core['Date of Interview'].isin(['26/05/2016']), 'Date of Interview'] = '2016-05-26'
df_core.loc[df_core['Date of Interview'].isin(['10.05.2016']), 'Date of Interview'] = '2016-05-10'
df_core.loc[df_core['Date of Interview'].isin(['28.08.2016 & 09.00 AM', '28.08.2016 & 9.30 AM']), 'Date of Interview'] = '2016-08-28'
df_core.loc[df_core['Date of Interview'].isin(['28.8.2016 & 12.00 PM', '28.08.2016 & 09.30 AM']), 'Date of Interview'] = '2016-08-28'
df_core.loc[df_core['Date of Interview'].isin(['28.8.2016 & 10.30 AM', '28.8.2016 & 09.30 AM']), 'Date of Interview'] = '2016-08-28'
df_core.loc[df_core['Date of Interview'].isin(['28.8.2016 & 04.00 PM', '28.08.2016 & 11.30 AM']), 'Date of Interview'] = '2016-08-28'
df_core.loc[df_core['Date of Interview'].isin(['28.08.2016 & 11.00 AM', '28.08.2016 & 10.30 AM']), 'Date of Interview'] = '2016-08-28'
df_core.loc[df_core['Date of Interview'].isin(['28.8.2016 & 03.00 PM', '28.08.2016 & 10.00 AM']), 'Date of Interview'] = '2016-08-28'
df_core.loc[df_core['Date of Interview'].isin(['28.8.2016 & 02.00 PM', '28.8.2016 & 11.00 AM']), 'Date of Interview'] = '2016-08-28'
df_core.loc[df_core['Date of Interview'].isin(['13.06.2016']), 'Date of Interview'] = '2016-06-13'
df_core.loc[df_core['Date of Interview'].isin(['02.09.2016']), 'Date of Interview'] = '2016-09-02'
df_core.loc[df_core['Date of Interview'].isin(['02.12.2015']), 'Date of Interview'] = '2015-12-02'
df_core.loc[df_core['Date of Interview'].isin(['23.02.2016']), 'Date of Interview'] = '2016-02-23'
df_core.loc[df_core['Date of Interview'].isin(['22.03.2016']), 'Date of Interview'] = '2016-02-22'
df_core.loc[df_core['Date of Interview'].isin(['26.02.2016']), 'Date of Interview'] = '2016-02-26'
df_core.loc[df_core['Date of Interview'].isin(['06.02.2016']), 'Date of Interview'] = '2016-02-06'
df_core.loc[df_core['Date of Interview'].isin(['21.4.2016']), 'Date of Interview'] = '2016-04-21'
df_core.loc[df_core['Date of Interview'].isin(['21/04/16']), 'Date of Interview'] = '2016-04-21'
df_core.loc[df_core['Date of Interview'].isin(['21.4.15']), 'Date of Interview'] = '2015-04-21'
df_core.loc[df_core['Date of Interview'].isin(['22.01.2016']), 'Date of Interview'] = '2016-01-22'
df_core.loc[df_core['Date of Interview'].isin(['3.6.16']), 'Date of Interview'] = '2016-06-03'
df_core.loc[df_core['Date of Interview'].isin(['03/06/16']), 'Date of Interview'] = '2016-06-03'
df_core.loc[df_core['Date of Interview'].isin(['09.01.2016']), 'Date of Interview'] = '2016-01-09'
df_core.loc[df_core['Date of Interview'].isin(['09-01-2016']), 'Date of Interview'] = '2016-01-09'
df_core.loc[df_core['Date of Interview'].isin(['03.04.2015']), 'Date of Interview'] = '2015-04-03'
df_core.loc[df_core['Date of Interview'].isin(['13/03/2015']), 'Date of Interview'] = '2015-03-13'
df_core.loc[df_core['Date of Interview'].isin(['17/03/2015','17.03.2015']), 'Date of Interview'] = '2015-03-17'
df_core.loc[df_core['Date of Interview'].isin(['18.03.2014']), 'Date of Interview'] = '2014-03-18'
df_core.loc[df_core['Date of Interview'].isin(['4.04.15']), 'Date of Interview'] = '2015-04-04'
df_core.loc[df_core['Date of Interview'].isin(['16.04.2015']), 'Date of Interview'] = '2015-04-16'
df_core.loc[df_core['Date of Interview'].isin(['17.04.2015']), 'Date of Interview'] = '2015-04-17'
df_core.loc[df_core['Date of Interview'].isin(['9.04.2015']), 'Date of Interview'] = '2015-04-09'
df_core.loc[df_core['Date of Interview'].isin(['05/02/15']), 'Date of Interview'] = '2015-02-05'
df_core.loc[df_core['Date of Interview'].isin(['30.05.2016']), 'Date of Interview'] = '2016-05-30'
df_core.loc[df_core['Date of Interview'].isin(['07.06.2016']), 'Date of Interview'] = '2015-06-07'
df_core.loc[df_core['Date of Interview'].isin(['20.08.2016']), 'Date of Interview'] = '2016-08-20'
df_core.loc[df_core['Date of Interview'].isin(['14.01.2016']), 'Date of Interview'] = '2016-01-14'
df_core.loc[df_core['Date of Interview'].isin(['30.1.16 ','30.01.2016','30/01/16','30.1.16']), 'Date of Interview'] = '2016-01-30'
df_core.loc[df_core['Date of Interview'].isin(['30.1.2016','30.01.16','30-1-2016']), 'Date of Interview'] = '2016-01-30'
df_core.loc[df_core['Date of Interview'].isin(['06.05.2016']), 'Date of Interview'] = '2016-05-06'
df_core['Date of Interview'].unique()
df_core['Date of Interview_DT'] = pd.to_datetime(df_core['Date of Interview'])
df.info()
df = df.append({'Country':'in', 'City':'baddi', 'AccentCity':'baddi', 'Region':' ', 'Population':0, 'Latitude':30.9578, 'Longitude':76.7914},ignore_index=True)
df = df.append({'Country':'in', 'City':'noida', 'AccentCity':'noida', 'Region':' ', 'Population':0, 'Latitude':28.5355, 'Longitude':77.3910},ignore_index=True)
df_core[df_core["Location"] !=df_core['Candidate Current Location']]
df_core.drop("Location", axis=1, inplace=True)
def distance(row1, row2):
      return haversine((df[df.City == row1]['Latitude'].values[0], df[df.City == row1]['Longitude'].values[0]), (df[df.City == row2]['Latitude'].values[0], df[df.City == row2]['Longitude'].values[0]))   
df_core['ven-curr_loc']= df_core.apply(lambda row: distance(row['Interview Venue'], row['Candidate Current Location']), axis=1)
df_core['curr_loc-job_loc']= df_core.apply(lambda row: distance(row['Candidate Current Location'], row['Candidate Job Location']), axis=1)
df_core['curr_loc-nat_loc']= df_core.apply(lambda row: distance(row['Candidate Current Location'], row['Candidate Native location']), axis=1)
df_core['job_loc-nat_loc']= df_core.apply(lambda row: distance(row['Candidate Job Location'], row['Candidate Native location']), axis=1)
df_core['Attendance'] = df_core['Observed Attendance'].apply(lambda x: 1 if x == 'yes' else 0)
temp = pd.DataFrame.transpose(pd.DataFrame([df_core.pivot_table( values=['Attendance'],index='Date of Interview_DT', aggfunc='count')['Attendance'].rename_axis(None), df_core.pivot_table( values=['Attendance'],index='Date of Interview_DT', aggfunc='sum')['Attendance'].rename_axis(None)]))
temp.columns =['Invited', 'Attended']
temp.plot(kind='bar', figsize=(20,5))
(temp['Attended']/temp['Invited']).plot(kind='bar', figsize=(20,5))
data = df_core.pivot_table( values=['Attendance'],index='Date of Interview_DT', aggfunc='count')['Attendance'].reindex(pd.date_range(start="2014", end="2016-12-31", freq='D'), fill_value=0.0)
data1 = df_core.pivot_table( values=['Attendance'],index='Date of Interview_DT', aggfunc='sum')['Attendance'].reindex(pd.date_range(start="2014", end="2016-12-31", freq='D'), fill_value=0.0)
groups = data.resample('M').sum().groupby(pd.Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
    try:
        years[name.year] = group.values
    except:
        continue

groups = data1.resample('M').sum().groupby(pd.Grouper(freq='A'))
years1 = pd.DataFrame()
for name, group in groups:
    try:
        years1[name.year] = group.values
    except:
        continue
        
        
years = years.add_suffix(" Invited")
years1 = years1.add_suffix(" Attended")
ax = years.plot(figsize=(10,10))
years1.plot(figsize=(10,10),ax=ax)
years.boxplot(figsize=(10,5))
years = years.T
plt.matshow(years, interpolation=None, aspect='auto', cmap='coolwarm')
years1 = years1.T
plt.matshow(years1, interpolation=None, aspect='auto', cmap='coolwarm')
df_core['First Half'] = df_core['Date of Interview_DT'].apply(lambda x: 0 if (x.month > 6) else 1)
df_core = df_core.merge(temp, left_on='Date of Interview_DT', right_index=True, how='left')
df_core[(df_core['curr_loc-job_loc'] >300) & df_core['Observed Attendance'] == 1]['Marital Status'].value_counts()
df_core[(df_core['ven-curr_loc'] < 20)]['Observed Attendance'].value_counts()
df_core[(df_core['curr_loc-nat_loc'] > 100) & (df_core['job_loc-nat_loc'] <= 20)]['Observed Attendance'].value_counts()
df_core[(df_core['curr_loc-job_loc'] > 100)]['Observed Attendance'].value_counts()
df_core.info()
cols= ['Client name', 'Industry', 'Interview Type', 'Gender', 'Permissions','Unscheduled meetings',
'Call three hours before', 'Alternative number', 'Resume printed', 'Knows venue', 'Call letter shared', 'Marital Status',
'ven-curr_loc', 'curr_loc-job_loc', 'curr_loc-nat_loc', 'job_loc-nat_loc', 'Attendance', 'Position to be closed', 
'Invited']
df_exp = df_core.filter(cols, axis=1)
df_exp
dummy= ['Client name', 'Industry', 'Interview Type', 'Gender', 'Permissions','Unscheduled meetings',
'Call three hours before', 'Alternative number', 'Resume printed', 'Knows venue', 'Call letter shared', 'Marital Status',
'Position to be closed']
for cols in dummy:
        df_exp = pd.concat([df_exp, pd.get_dummies(df_exp[cols],drop_first=True,prefix=cols, prefix_sep='_')], axis=1)
        df_exp.drop(cols, inplace=True, axis=1)
df_exp.info()
y = df_exp['Attendance']
df_exp.drop('Attendance', axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_exp.as_matrix(), y.values, test_size=0.30, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
mm_X = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(mm_X.fit_transform(df_exp), y.values, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',  solver='liblinear', C=0.4)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=5, weights = 'uniform' )
kn.fit(X_train, y_train)
pred = kn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
from sklearn.svm import SVC
svc = SVC(C=5.0, kernel='linear')
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))