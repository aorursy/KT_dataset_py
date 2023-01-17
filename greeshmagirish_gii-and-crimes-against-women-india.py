import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os
crime_df = pd.read_csv('/kaggle/input/crime-against-women-20012014-india/crimes_against_women_2001-2014.csv')

crime_df.info()
crime_df = crime_df.drop(['Unnamed: 0', 'DISTRICT'], axis=1)

crime_df.head()
def get_case_consistency(row):

    row = row['STATE/UT'].strip()

    row = row.upper()

    row = row.title()

    return row



crime_df['STATE/UT'] = crime_df.apply(get_case_consistency, axis=1)

crime_df['STATE/UT'].unique()
north_india = ['Jammu & Kashmir', 'Punjab', 'Himachal Pradesh', 'Haryana', 'Uttarakhand', 'Uttar Pradesh', 'Chandigarh']

east_india = ['Bihar', 'Odisha', 'Jharkhand', 'West Bengal']

south_india = ['Andhra Pradesh', 'Karnataka', 'Kerala' ,'Tamil Nadu', 'Telangana']

west_india = ['Rajasthan' , 'Gujarat', 'Goa','Maharashtra','Goa']

central_india = ['Madhya Pradesh', 'Chhattisgarh']

north_east_india = ['Assam', 'Sikkim', 'Nagaland', 'Meghalaya', 'Manipur', 'Mizoram', 'Tripura', 'Arunachal Pradesh']

ut_india = ['A & N ISLANDS', 'Delhi', 'LAKSHADWEEP', 'PUDUCHERRY', 'A&N Islands', 'Daman & Diu', 'Delhi Ut', 'Lakshadweep',

       'Puducherry', 'D & N Haveli', 'DAMAN & DIU', 'D&N Haveli', 'A & N Islands']



def get_zonal_names(row):

    if row['STATE/UT'].title().strip() in north_india:

        val = 'North Zone'

    elif row['STATE/UT'].title().strip()  in south_india:

        val = 'South Zone'

    elif row['STATE/UT'].title().strip()  in east_india:

        val = 'East Zone'

    elif row['STATE/UT'].title().strip()  in west_india:

        val = 'West Zone'

    elif row['STATE/UT'].title().strip()  in central_india:

        val = 'Central Zone'

    elif row['STATE/UT'].title().strip()  in north_east_india:

        val = 'NE Zone'

    elif row['STATE/UT'].title().strip()  in ut_india:

        val = 'Union Terr'

    else:

        val = 'No Value'

    return val



crime_df['Zones'] = crime_df.apply(get_zonal_names, axis=1)

crime_df['Zones'].unique()
crime_df[(crime_df['Zones'] == 'No Value')]['STATE/UT'].unique()
crimes = ['Rape','Kidnapping and Abduction','Dowry Deaths','Assault on women with intent to outrage her modesty','Insult to modesty of Women',

'Cruelty by Husband or his Relatives','Importation of Girls']
rape_df = crime_df.groupby(by=['Year', 'STATE/UT', 'Zones'])['Rape'].sum().reset_index().sort_values(crimes[0], ascending=False)
plt.figure(figsize=(20,15))

count = 1



for zone in rape_df.Zones.unique():

    plt.subplot(len(rape_df.Zones.unique()),1,count)



    sns.lineplot(rape_df[(rape_df['Zones'] == zone)]['Year'],rape_df[(rape_df['Zones'] == zone)]['Rape'],ci=None)

    plt.subplots_adjust(hspace=0.8)

    plt.xlabel('Years')

    plt.ylabel('# Rape Cases')

    plt.title(zone)

    count+=1
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(rape_df.Zones, rape_df.Rape, errwidth=0)

plt.ylabel('# Rape Cases')

plt.title('Zone-Wise Rape Cases Registered', fontdict = {'fontsize' : 15})
rape_st_df = rape_df[(rape_df['Zones'] == 'Central Zone') | (rape_df['Zones'] == 'East Zone')]

rape_st_df = rape_st_df.groupby(by=['STATE/UT'])['Rape'].sum().reset_index().sort_values('Rape', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(rape_st_df['STATE/UT'], rape_st_df.Rape, errwidth=0)

plt.ylabel('# Rape Cases')

plt.title('States with Rape Cases Registered', fontdict = {'fontsize' : 15})

rape_st_df.head(5)
kidnap_df = crime_df.groupby(by=['Year', 'STATE/UT', 'Zones'])['Kidnapping and Abduction'].sum().reset_index().sort_values(crimes[1], ascending=False)
plt.figure(figsize=(20,15))

count = 1



for zone in kidnap_df.Zones.unique():

    plt.subplot(len(kidnap_df.Zones.unique()),1,count)



    sns.lineplot(kidnap_df[(kidnap_df['Zones'] == zone)]['Year'],kidnap_df[(kidnap_df['Zones'] == zone)]['Kidnapping and Abduction'],ci=None)

    plt.subplots_adjust(hspace=0.9)

    plt.xlabel('Years')

    plt.ylabel('# Cases')

    plt.title(zone)

    count+=1
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(kidnap_df.Zones, kidnap_df['Kidnapping and Abduction'], errwidth=0)

plt.ylabel('# Kidnapping/Abduction Cases')

plt.title('Zone-Wise Kidnapping/Abduction Cases Registered', fontdict = {'fontsize' : 15})
kidnap_st_df = kidnap_df[(kidnap_df['Zones'] == 'East Zone') | (kidnap_df['Zones'] == 'West Zone')]

kidnap_st_df = kidnap_st_df.groupby(by=['STATE/UT'])['Kidnapping and Abduction'].sum().reset_index().sort_values('Kidnapping and Abduction', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(kidnap_st_df['STATE/UT'], kidnap_st_df['Kidnapping and Abduction'], errwidth=0)

plt.ylabel('# Kidnapping and Abduction Cases')

plt.title('States with Kidnapping and Abduction Cases Registered', fontdict = {'fontsize' : 15})

kidnap_st_df.head(5)
dowry_df = crime_df.groupby(by=['Year', 'STATE/UT', 'Zones'])['Dowry Deaths'].sum().reset_index().sort_values('Dowry Deaths', ascending=False)
plt.figure(figsize=(20,15))

count = 1



for zone in dowry_df.Zones.unique():

    plt.subplot(len(dowry_df.Zones.unique()),1,count)



    sns.lineplot(dowry_df[(dowry_df['Zones'] == zone)]['Year'],dowry_df[(dowry_df['Zones'] == zone)]['Dowry Deaths'],ci=None)

    plt.subplots_adjust(hspace=0.9)

    plt.xlabel('Years')

    plt.ylabel('# Cases')

    plt.title(zone)

    count+=1
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(dowry_df.Zones, dowry_df['Dowry Deaths'], errwidth=0)

plt.ylabel('# Dowry Deaths Cases')

plt.title('Zone-Wise Dowry Deaths Cases Registered', fontdict = {'fontsize' : 15})
dowry_st_df = dowry_df[(dowry_df['Zones'] == 'East Zone') | (dowry_df['Zones'] == 'Central Zone')]

dowry_st_df = dowry_st_df.groupby(by=['STATE/UT'])['Dowry Deaths'].sum().reset_index().sort_values('Dowry Deaths', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(dowry_st_df['STATE/UT'], dowry_st_df['Dowry Deaths'], errwidth=0)

plt.ylabel('# Dowry Deaths Cases')

plt.title('States with Dowry Deaths Cases Registered', fontdict = {'fontsize' : 15})

dowry_st_df.head(5)
assault_df = crime_df.groupby(by=['Year', 'STATE/UT', 'Zones'])['Assault on women with intent to outrage her modesty'].sum().reset_index().sort_values('Assault on women with intent to outrage her modesty', ascending=False)
plt.figure(figsize=(20,15))

count = 1



for zone in assault_df.Zones.unique():

    plt.subplot(len(assault_df.Zones.unique()),1,count)



    sns.lineplot(assault_df[(assault_df['Zones'] == zone)]['Year'],assault_df[(assault_df['Zones'] == zone)]['Assault on women with intent to outrage her modesty'],ci=None)

    plt.subplots_adjust(hspace=0.9)

    plt.xlabel('Years')

    plt.ylabel('# Cases')

    plt.title(zone)

    count+=1
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(assault_df.Zones, assault_df['Assault on women with intent to outrage her modesty'], errwidth=0)

plt.ylabel('# Dowry Deaths Cases')

plt.title('Zone-Wise Assault on Women Cases Registered', fontdict = {'fontsize' : 15})
assault_st_df = assault_df[(assault_df['Zones'] == 'Central Zone') | (assault_df['Zones'] == 'South Zone')]

assault_st_df = assault_st_df.groupby(by=['STATE/UT'])['Assault on women with intent to outrage her modesty'].sum().reset_index().sort_values('Assault on women with intent to outrage her modesty', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(assault_st_df['STATE/UT'], assault_st_df['Assault on women with intent to outrage her modesty'], errwidth=0)

plt.ylabel('# Assault on women with intent to outrage her modesty Cases')

plt.title('States with Assault on Women Cases Registered', fontdict = {'fontsize' : 15})

assault_st_df.head(5)
insult_df = crime_df.groupby(by=['Year', 'STATE/UT', 'Zones'])['Insult to modesty of Women'].sum().reset_index().sort_values('Insult to modesty of Women', ascending=False)
plt.figure(figsize=(20,15))

count = 1



for zone in insult_df.Zones.unique():

    plt.subplot(len(insult_df.Zones.unique()),1,count)



    sns.lineplot(insult_df[(insult_df['Zones'] == zone)]['Year'],insult_df[(insult_df['Zones'] == zone)]['Insult to modesty of Women'],ci=None)

    plt.subplots_adjust(hspace=0.9)

    plt.xlabel('Years')

    plt.ylabel('# Cases')

    plt.title(zone)

    count+=1
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(insult_df.Zones, insult_df['Insult to modesty of Women'], errwidth=0)

plt.ylabel('# Insult to modesty of Women Cases')

plt.title('Zone-Wise Insult to modesty of Women Cases Registered', fontdict = {'fontsize' : 15})
insult_st_df = insult_df[(insult_df['Zones'] == 'South Zone')]

insult_st_df = insult_st_df.groupby(by=['STATE/UT'])['Insult to modesty of Women'].sum().reset_index().sort_values('Insult to modesty of Women', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(insult_st_df['STATE/UT'], insult_st_df['Insult to modesty of Women'], errwidth=0)

plt.ylabel('# Insult to modesty of Women Cases')

plt.title('States with Insult to modesty of Women Cases Registered', fontdict = {'fontsize' : 15})

insult_st_df.head(5)
cruel_df = crime_df.groupby(by=['Year', 'STATE/UT', 'Zones'])['Cruelty by Husband or his Relatives'].sum().reset_index().sort_values('Cruelty by Husband or his Relatives', ascending=False)
plt.figure(figsize=(20,15))

count = 1



for zone in cruel_df.Zones.unique():

    plt.subplot(len(cruel_df.Zones.unique()),1,count)



    sns.lineplot(cruel_df[(cruel_df['Zones'] == zone)]['Year'],cruel_df[(cruel_df['Zones'] == zone)]['Cruelty by Husband or his Relatives'],ci=None)

    plt.subplots_adjust(hspace=0.9)

    plt.xlabel('Years')

    plt.ylabel('# Cases')

    plt.title(zone)

    count+=1
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(cruel_df.Zones, cruel_df['Cruelty by Husband or his Relatives'], errwidth=0)

plt.ylabel('# Cruelty by Husband or his Relatives Cases')

plt.title('Zone-Wise Cruelty by Husband or his Relatives Cases Registered', fontdict = {'fontsize' : 15})
cruel_st_df = cruel_df[(cruel_df['Zones'] == 'West Zone') | (cruel_df['Zones'] == 'South Zone')]

cruel_st_df = cruel_st_df.groupby(by=['STATE/UT'])['Cruelty by Husband or his Relatives'].sum().reset_index().sort_values('Cruelty by Husband or his Relatives', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(cruel_st_df['STATE/UT'], cruel_st_df['Cruelty by Husband or his Relatives'], errwidth=0)

plt.ylabel('# Cruelty by Husband or his Relatives Cases')

plt.title('States with Cruelty by Husband or his Relatives Registered', fontdict = {'fontsize' : 15})

cruel_st_df.head(5)
import_df = crime_df.groupby(by=['Year', 'STATE/UT', 'Zones'])['Importation of Girls'].sum().reset_index().sort_values('Importation of Girls', ascending=False)
plt.figure(figsize=(20,15))

count = 1



for zone in import_df.Zones.unique():

    plt.subplot(len(import_df.Zones.unique()),1,count)



    sns.lineplot(import_df[(import_df['Zones'] == zone)]['Year'],import_df[(import_df['Zones'] == zone)]['Importation of Girls'],ci=None)

    plt.subplots_adjust(hspace=0.9)

    plt.xlabel('Years')

    plt.ylabel('# Cases')

    plt.title(zone)

    count+=1
fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(import_df.Zones, import_df['Importation of Girls'], errwidth=0)

plt.ylabel('# Importation of Girls Cases')

plt.title('Zone-Wise Importation of Girls Cases Registered', fontdict = {'fontsize' : 15})
import_st_df = import_df[(import_df['Zones'] == 'East Zone')]

import_st_df = import_st_df.groupby(by=['STATE/UT'])['Importation of Girls'].sum().reset_index().sort_values('Importation of Girls', ascending=False)

fig, ax = plt.subplots(figsize=(15,10))

sns.barplot(import_st_df['STATE/UT'], import_st_df['Importation of Girls'], errwidth=0)

plt.ylabel('# Importation of Girls Cases')

plt.title('States with Importation of Girls Cases Registered', fontdict = {'fontsize' : 15})

import_st_df.head(5)