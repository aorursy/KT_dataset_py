import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math
# reading in the data

data = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

data = data.drop(['Unnamed: 0','Unnamed: 0.1'], axis = 1)

data.head()
# collecting and counting the number of missions by whether the mission was a success or failure

success = data.groupby('Status Mission').count()['Detail'].rename('Number of Missions')

success = success.drop('Prelaunch Failure')



fig, ax = plt.subplots(figsize=(12,8))

plt.rcParams.update({'font.size': 19})

ax.pie(x=success, labels = ['Failure','Partial Failure', 'Success'], rotatelabels=True, startangle = -18)

ax.set_title('Proportion of Mission Successes')
# formatting the data by year and collecting the number of launches by year

data['Year'] = data['Datum'].apply(lambda x: int(x.split()[3]))

yearLaunches = data.groupby('Year').count().reset_index()[['Year','Detail']]



fig, ax = plt.subplots(figsize=(20,10))

plt.rcParams.update({'font.size': 15})

sns.barplot(x='Year', y='Number of Launches', data=yearLaunches.rename(columns = {'Detail':'Number of Launches'}))

plt.xticks(rotation=270)
# translating the launch location into a single country and collecting the data by country

data['Country'] = data['Location'].apply(lambda x: x.split()[-1])

countries = data.groupby('Country').count()['Detail'].reset_index().rename(columns = {'Detail':'Number of Launches'}).sort_values(by='Number of Launches', ascending=False)

countries['Country'] = countries['Country'].apply(lambda x: 'New Zealand' if x == 'Zealand' else x)

countries['Country'] = countries['Country'].apply(lambda x: 'Other' if (x == 'Sea' or x == 'Facility' or x == 'Site')  else x)

countries = countries.rename(columns = {'Country':'Launch Country'})



fig, ax = plt.subplots(figsize=(12,8))

plt.rcParams.update({'font.size': 22})

sns.barplot(x='Launch Country', y='Number of Launches', data = countries)

plt.xticks(rotation=90)
# collecting the data by whether the mission is still active.

status = data.groupby('Status Rocket').count()['Detail']



fig, ax = plt.subplots(figsize=(12,8))

plt.rcParams.update({'font.size': 22})

ax.pie(x=status, labels = data['Status Rocket'].unique(), colors=['#28c930','red'], rotatelabels=True, startangle = -33)

ax.set_title('Proportion of Missions Still Active')
# Convert the company names into the nation that owns the organisation (if a state-owned agency). This one was done manually so please tell me if I've missed

# one of the national agencies (they're all acronymised).

data['Company Country'] = data['Company Name'].map({

                                'CASC':'China',

                                'IAI':'Israel',

                                'VKS RF':'Russia',

                                'ISA':'Israel',

                                'KARI':'South Korea',

                                'AEB':'Brazil',

                                'ISRO':'India',

                                'IRGC':'Iran',

                                'CASIC':'China',

                                'KCST':'North Korea',

                                'ESA':'Europe',

                                'NASA':'USA',

                                'ISAS':'Japan',

                                'RVSN USSR':'USSR',

                                'ASI':'Italy',

                                'US Air Force':'USA',

                                'CNES':'France',

                                "Arm??e de l'Air":'France',

                                'US Navy':'USA'

                            }).fillna('Private')
# grouping the relavent data by the agency country and the outcome of the mission. The "prelaunch failures" of which there are 4 are omitted.

newData = data[['Company Country','Status Mission','Detail']].groupby(['Company Country','Status Mission']).count().rename(columns={'Detail':"Number of Launches"}).reset_index()

newData.drop(newData[newData['Status Mission']=='Prelaunch Failure'].index, inplace=True)



# loop through the country and slice the dataframe by that country, appending the new dataframe to a list with the country label

countryLaunches = []

for country in list(newData['Company Country'].unique()):

    dataToDraw = newData[newData['Company Country']==country][['Number of Launches','Status Mission']]

    countryLaunches.append([dataToDraw,country])

    

# it seems that matplotlib auto formats the data because my attempt was bad so this column/row calculation is useless

ncols = 3

nrows = math.ceil(len(newData['Company Country'].unique())/3)



# maps the outcome to the correct colour

colorMap = {'Success':'#28c930','Failure':'red','Partial Failure':'#e8aa00','Prelaunch Failure':'grey'}



# loop through each country and plot them a pie chart

plt.subplots(ncols = ncols, nrows = nrows, figsize=(26,13))

plt.suptitle('National Space Mission Outcome')



for i in range(0, len(countryLaunches)):

    plt.subplot(ncols, nrows, i+1)

    plt.rcParams.update({'font.size': 16})

    dataToDraw = countryLaunches[i][0]['Number of Launches']

    labels = countryLaunches[i][0]['Status Mission'].unique()

    country = countryLaunches[i][1]

    plt.pie(dataToDraw, labels=labels, colors=[colorMap[k] for k in labels])

    plt.title(country)

plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
# change the agency nation to simply national agency or private agency. Change the year to just the decade.

newData = pd.DataFrame()

newData['Agency Type'] = data['Company Country'].apply(lambda x: 'Private' if x == 'Private' else 'National')

newData[['Year','Launches']] = data[['Year','Detail']]

newData['Decade'] = newData['Year'].apply(lambda x: math.floor(x/10)*10)

newData = newData.groupby(['Decade','Agency Type']).count().reset_index()



fig, ax = plt.subplots(figsize = (15,10))

sns.barplot(x ='Decade', y='Launches',hue='Agency Type',data = newData)

plt.title('National and Private Space Launches')