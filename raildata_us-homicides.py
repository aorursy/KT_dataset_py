import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
# read the dataset

dataset = pd.read_csv('../input/database.csv',index_col=0,low_memory=False,header=0)
# Lets have a look at what weapons are used

WeaponOfChoice = dataset.groupby('Weapon').State.count()

print(WeaponOfChoice)
# how is gender distributed

pv_gender = dataset.pivot_table(columns='Perpetrator Sex', index='Victim Sex',values='Incident',aggfunc='count')

pv_gender
pv_gender.Female['Male'] / (pv_gender.Female['Female'] + pv_gender.Female['Male'])
pv_gender.Male['Male'] / (pv_gender.Male['Female'] + pv_gender.Male['Male'])
pv_weapons = dataset.pivot_table(columns='Crime Solved', index='Weapon',values='Incident',aggfunc='count')

pv_weapons.sort_values(by='No', ascending=0)
pv_CasesClosed = dataset.pivot_table(columns='Crime Solved', index='Victim Sex',values='Incident',aggfunc='count')

pv_CasesClosed

# Cases with Male victims are less likely to be solved ~31.7%.

pv_CasesClosed.No['Male'] / pv_CasesClosed.T.Male.sum()
TimeSeries = dataset.groupby('Year').Incident.count()

pt = TimeSeries.plot()

# from 1993 to 1999 there is a steady decline, from 1999 to 2014 the level stays relatively low.
dataset['Victim Age'][dataset['Victim Age'] > 100 ] = 100
# The most common victim age is 20-30

plt_VA = dataset['Victim Age'].hist()

dataset.groupby('Relationship').Incident.count()
MO = dataset.pivot_table(index='Relationship', columns='Weapon',values='Incident',aggfunc='count')
# normalise the number of incidents involving each weapon for each relationship type. 

MO_NormalisedPercentage = 100*MO.div(MO.sum(axis=1),axis=0)

MO_NormalisedPercentage
# Who is most likely to use which weapon

MO_NormalisedPercentage.T.idxmax()