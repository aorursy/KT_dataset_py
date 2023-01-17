import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('../input/rec-crime-pfa.csv')
dataset.head()
dataset.tail()
print('There are '+ str(dataset.shape[0])+ ' rows and '+ str(dataset.shape[1]) +' columns')
dataset.info()
data = {'unique_values' : dataset.nunique(),

        'na_values' : dataset.isna().sum(),

        'data_types' : dataset.dtypes}

pd.DataFrame(data)
dataset['12 months ending'] = pd.to_datetime(dataset['12 months ending'])



#Get year, month, day

dataset['Year'] = dataset['12 months ending'].dt.year

dataset['Month'] = dataset['12 months ending'].dt.month

dataset['Day'] = dataset['12 months ending'].dt.day
dataset.head()
dataset.drop(['12 months ending'], inplace=True, axis=1)

dataset.head()
dataset.rename(inplace=True, columns={'PFA':'pfa', 'Region':'region', 'Offence':'offence', 'Rolling year total number of offences':'total', 'Year':'year', 'Month':'month', 'Day':'day'})
dataset.head()
# Making data more simple

dataset.loc[dataset['offence'] == 'Domestic burglary', 'offence'] = 'Burglary'

dataset.loc[dataset['offence'] == 'Non-domestic burglary', 'offence'] = 'Burglary'

dataset.loc[dataset['offence'] == 'Non-residential burglary', 'offence'] = 'Burglary'

dataset.loc[dataset['offence'] == 'Residential burglary', 'offence'] = 'Burglary'



dataset.loc[dataset['offence'] == 'Bicycle theft', 'offence'] = 'Theft'

dataset.loc[dataset['offence'] == 'Shoplifting', 'offence'] = 'Theft'

dataset.loc[dataset['offence'] == 'Theft from the person', 'offence'] = 'Theft'

dataset.loc[dataset['offence'] == 'All other theft offences', 'offence'] = 'Theft'



dataset.loc[dataset['offence'] == 'Violence with injury', 'offence'] = 'Violence'

dataset.loc[dataset['offence'] == 'Violence without injury', 'offence'] = 'Violence'

{

    'unique_pfa': dataset['pfa'].unique(),

    'unique_region': dataset['region'].unique(), 

    'unique_offence': dataset['offence'].unique()

}
## Crime based on year

plt.figure(figsize=(15,6))

ax = sns.barplot(x='year', y='total', data=dataset)

plt.xticks(rotation=45,fontsize=10)

plt.show()
## Crime based on month 

plt.figure(figsize=(15,6))

ax = sns.barplot(x='month', y='total', data=dataset)

plt.show()
## Crime based on region

plt.figure(figsize=(16,5))

ax = sns.barplot(x='region',y='total', data=dataset)

plt.xticks(rotation=70)

plt.show()
dataset1 = dataset[dataset['year']>2014]

dataset2=dataset1.sort_values('total', ascending=False).head(10)

dataset2.sort_values('year')
east = dataset[dataset['region'] == 'East']

east.head(10)
# East Distribution based on year

sns.barplot(x='year', y='total', data=east)

plt.xticks(rotation=45)

plt.show()
#Most Popular offence in East

sns.set()

sns.jointplot(x='year',y='total',data=east)

plt.show()
#Popular crime in east

popular = east.groupby('offence')['total'].count()

popular.head()
#increase of offence

sns.lineplot(x='offence', y='total', data=dataset)

plt.xticks(rotation=90)

plt.show()
# rise in CIFAS and action fraud offence

offen= dataset[dataset['offence']=='CIFAS']

offen1= dataset[dataset['offence']=='Action Fraud']

sns.lineplot(x='year',y='total', data=offen)

sns.lineplot(x='year',y='total', data=offen1)

plt.xticks(rotation=90)

plt.legend('upper center')

plt.show()

label_weapon = 'Possession of weapons offences'

df_weapon = dataset.loc[dataset['offence'] == label_weapon]

labels_weapon_high = ['Metropolitan Police', 'Greater Manchester', 'West Midlands', 'West Yorkshire']

df_weapon_high = df_weapon.loc[df_weapon['pfa'].isin(labels_weapon_high)]



sns.lineplot(data=df_weapon_high, x='year', y='total', hue='pfa')

plt.show()