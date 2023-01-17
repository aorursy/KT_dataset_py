import os

os.getcwd()
# Ignore harmless warnings

import warnings

warnings.filterwarnings("ignore")



#Importing libraries for data analysis and cleaning

import numpy as np

import pandas as pd



#importing visualisation libraries for data visualisation

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

init_notebook_mode(connected=True)



#load datasets

tiers = pd.read_csv('../input/health-facilities-gh/health-facility-tiers.csv')

facilities = pd.read_csv('../input/health-facilities-gh/health-facilities-gh.csv')
#checking dataset 1

tiers.head()
#checking the general summary of the dataset

tiers.info()
#checking for duplicated data

tiers.duplicated().sum()
#examining the 20 rows that are duplicated

tiers.loc[tiers.duplicated(keep=False),:]
#removing duplicated data

tiers = tiers.drop_duplicates()



#confirming 

tiers.duplicated().sum()
#checking dataset 2

facilities.head()
#general info of dataset 2

facilities.info()
facilities.isnull().sum()
#Investigating missing rows in the town column

facilities[facilities['Town'].isnull()]
#checking for duplicated data

facilities.duplicated().sum()
#examing those 30 duplicated rows

facilities.loc[facilities.duplicated(),:]
#removing duplicated rows

facilities = facilities.drop_duplicates()
#Investigating categorical data. This is to identify any duplicates resulting from many possible factors

no_regions = facilities['Region'].unique()



for x in no_regions:

    print(x)
#Identifying the available facility types accross the nation and cross-checking for errors

no_types = facilities['Type'].unique()



for x in no_types:

    print(x)
#investigating category clinic under 'Type'

facilities[facilities['Type'] == 'clinic']
#fixing error. Adding 'clinc' to 'Clinc'.

facilities['Type'].loc[[2010,2056]] = 'Clinic'
#Investigating the misspelled 'CPHS'

facilities[facilities['Type'] == 'CPHS']
#correcting that error

facilities['Type'].loc[646] = 'CHPS'
facilities[facilities['Type'] == 'DHD']

facilities['Type'].loc[1250] = 'Municipal Health Directorate'
#investigating all misclassifed CHPS types and ownerships

pd.set_option('display.max_rows', None)

facilities[facilities['FacilityName'].str.contains('CHPS')]
#duplicated data. One is classified as Clinic and the other CHPS with the same Lat and Long.

facilities.loc[[2183,2187]]
#dropping the category type 'clinic'

facilities = facilities.drop([2183])



#misclassified 'CHPS' under type

facilities.loc[[953,1008,1163,1166,2516,3103,3265]]
#correcting the wrongly classified 'Clinic' to the right category 'CHPS'

facilities['Type'].loc[[953,1008,1163,1166,2516,3103,3265]] = 'CHPS'
#duplicated. The spelling of it was spaced out

facilities[facilities['Type'] == 'Municipal  Health Directorate']
#compiling it into one

facilities['Type'].loc[1134] = 'Municipal Health Directorate'
#Investigating category 'Centre'

facilities[facilities['Type'] == 'Centre']
#reassigning to its correct category 'Health Centre'

facilities['Type'].loc[[99,667]] = 'Health Centre'
#Identifying the types of ownerships for these facilities accross the nation

no_ownship = facilities['Ownership'].unique()



for x in no_ownship:

    print(x)
#Investigating category 'Muslim'

facilities[facilities['Ownership'] == 'Muslim']



#Adding it to category 'Islamic'

facilities['Ownership'].loc[930] = 'Islamic'
#Investigating 'Clinic'

facilities[facilities['Ownership'] == 'Clinic']



#Since the clinic is a rural clinic ('Adadiem Rural Clinic'), it is reasonable to assign the ownership as 'Government'



#reassigning to 'Government'

facilities['Ownership'].loc[971] = 'Government'
#Investigating 'Maternity Home'

facilities[facilities['Ownership'] == 'Maternity Home']
#checking the most owned facilities

facilities['Ownership'].value_counts().head()
#reassigning to 'Government'

facilities['Ownership'].loc[[969,970]] = 'Government'
#investigating government

facilities[facilities['Ownership'] == 'government']
#fixing the error

facilities['Ownership'].loc[[2127, 3209, 3226, 3228, 3229, 3230]] = 'Government'
#investigating private

facilities[facilities['Ownership'] == 'private']
#fixing the error

facilities['Ownership'].loc[[1413,1608]] = 'Private'
#Investigating 'missions'

facilities[facilities['Ownership'] == 'Mission']



#reassigning to NGO

facilities['Ownership'].loc[3398] = 'NGO'
#wrongly classified Ownerships

facilities.loc[[704,2463,3265,3312]]
#correcting these to their right category 'Government'

facilities['Ownership'].loc[[704,2463,3265,3312]] = 'Government'
print('Two types of health facility tiers. Tier 2 and Tier 3:',tiers['Tier'].unique())

print('\n')

print('Tier 3 coverage percentage in Ghana:',round(100* len(tiers[tiers['Tier'] == 3])/len(tiers['Tier'])),'%')

print('With a total number of',tiers[tiers['Tier'] == 3]['Tier'].count())



print('\n')

print('Tier 2 coverage percentage in Ghana:',round(100* len(tiers[tiers['Tier'] == 2])/len(tiers['Tier'])),'%')

print('With a total number of',tiers[tiers['Tier'] == 2]['Tier'].count())



print('\n')

ylabel='Count'

xlabel='Types of Health Facility Tiers'

ax = tiers['Tier'].value_counts().plot(kind='bar',figsize=(12,5),title='Number of health facilty Tiers in Ghana',color='red');

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel);
tiers_per_region = tiers.groupby(['Region','Tier']).count()

tiers_per_region
plt.figure(figsize=(15,5));

plt.title('Count of Health facility Tiers per Region');

sns.countplot(data=tiers,x = 'Region',hue='Tier');
max_long = facilities['Longitude'].max()

min_long = facilities['Longitude'].min()

max_lat = facilities['Latitude'].max()

min_lat = facilities['Latitude'].min()
facilities['FacilityName'] = facilities['FacilityName'].str.lower()

tiers['Facility'] = tiers['Facility'].str.lower()

merged = pd.merge(facilities, tiers, left_on=['FacilityName'], right_on=['Facility'])
data = []

for index, tier in enumerate(merged['Tier'].unique()):

    facils = merged[merged['Tier'] == tier]

    data.append(

        go.Scattergeo(

        lon = facils['Longitude'],

        lat = facils['Latitude'],

        text = facils['FacilityName'],

        mode = 'markers',

        marker_color = index,

        name = "Tier " + str(tier)

        )

    )



layout = dict(

        title = 'Health facilities in Ghana based on Tier',

        geo = dict(

        scope = 'africa',

        landcolor = "rgb(212, 212, 212)",

        subunitcolor = "rgb(255, 255, 255)",

        lonaxis = dict(

            showgrid = True,

            gridwidth = 0.5,

            range= [ min_long - 5, max_long + 5 ],

            dtick = 5

        ),

        lataxis = dict (

            showgrid = True,

            gridwidth = 0.5,

            range= [ min_lat - 1, max_lat + 1 ],

            dtick = 5

        )

    )

)

fig = dict(data = data, layout = layout)

go.Figure(fig)
#Examining the overall count of the Health Facilities in Ghana

ylabel='Count'

xlabel='Types of Health Facilities'

ax1 = facilities['Type'].value_counts().plot(kind='bar',figsize=(11,5),title='The most common health facilities in Ghana');

ax1.autoscale(axis='x',tight=True)

ax1.set(xlabel=xlabel, ylabel=ylabel);
#Investigating the 5 most common health facilities and their total counts

facilities['Type'].value_counts().head()
df2 = facilities[facilities['Type'].str.contains('Clinic')]

df2 =df2['Region'].value_counts()



df3 = facilities[facilities['Type'].str.contains('Health Centre')]

df3=df3['Region'].value_counts()



df4 = facilities[facilities['Type'].str.contains('CHPS')]

df4=df4['Region'].value_counts()



df5 = facilities[facilities['Type'].str.contains('Maternity Home')]

df5 = df5['Region'].value_counts()



df6 = facilities[facilities['Type'].str.contains('Hospital')]

df6 = df6['Region'].value_counts()



per_reg = pd.concat([df2, df3,df4,df5,df6], axis=1).reset_index()

per_reg.columns = ['Region','Clinic','Health Centre','CHPS','Maternity Home','Hospital']

per_reg = per_reg.set_index('Region')

per_reg
#Analysing the highest count per each health facility

per_reg.describe().loc['max']
per_reg.iplot(kind='bar',barmode='stack',title='Distribution of the five most common health facilities per regional area',xTitle='Regions',yTitle='Count')
data = []

for index, region in enumerate(facilities['Region'].unique()):

    selected_facilities = facilities[facilities['Region'] == region]

    data.append(

        go.Scattergeo(

        lon = selected_facilities['Longitude'],

        lat = selected_facilities['Latitude'],

        text = selected_facilities['FacilityName'],

        mode = 'markers',

        marker_color = index,

        name = region

        )

    )



layout = dict(

        title = 'Health facilities in Ghana based on Region',

        geo = dict(

        scope = 'africa',

        landcolor = "rgb(212, 212, 212)",

        subunitcolor = "rgb(255, 255, 255)",

        lonaxis = dict(

            showgrid = True,

            gridwidth = 0.5,

            range= [ min_long - 5, max_long + 5 ],

            dtick = 5

        ),

        lataxis = dict (

            showgrid = True,

            gridwidth = 0.5,

            range= [ min_lat - 1, max_lat + 1 ],

            dtick = 5

        )

    )

)

fig = dict(data = data, layout = layout)

go.Figure(fig)





#Structuring into a dataframe

grp_ownships = pd.DataFrame(facilities['Ownership'].value_counts())

grp_ownships['Percentage Ownerships'] = round(100 * (grp_ownships['Ownership']/grp_ownships['Ownership'].sum()),1)

grp_ownships = pd.DataFrame(grp_ownships).reset_index()

grp_ownships.columns = ['Type','Ownership','Percentage Ownerships']



#Pie chart 

fig = px.pie(grp_ownships, values='Ownership', names='Type',

             title='Ownership Percentages', labels=dict(grp_ownships['Ownership']))

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
#Complete list of ownerships

grp_ownships