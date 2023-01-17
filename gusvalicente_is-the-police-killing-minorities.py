# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
%matplotlib inline
df1 = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')

df1.head()
df1.shape
df1.info()
df1.isnull().sum()
df1.describe()
df1.id = df1.id.astype('category')
df1.armed = df1.armed.astype('category')
df1.gender = df1.gender.astype('category')
df1.city = df1.city.astype('category')
df1.state = df1.state.astype('category')
df1.race = df1.race.astype('category')
df1.threat_level = df1.threat_level.astype('category')
df1.flee = df1.flee.astype('category')
df1.manner_of_death = df1.manner_of_death.astype('category')

#Properly assinging categorical records as a category
df1.info()
df1.corr()

sns.heatmap(df1.corr())
df1.replace(to_replace = ['A'], value = ['Asian'], inplace = True)
df1.replace(to_replace = ['B'], value = ['Black'], inplace = True)
df1.replace(to_replace = ['H'], value = ['Hispanic'], inplace = True)
df1.replace(to_replace = ['N'], value = ['Native American'], inplace = True)
df1.replace(to_replace = ['O'], value = ['Other'], inplace = True)
df1.replace(to_replace = ['W'], value = ['White'], inplace = True)

#Properly naming each one of the races, to facilitate analysis and comprehension in visualizations
df1['month'] = pd.to_datetime(df1['date']).dt.month
df1['year'] = pd.to_datetime(df1['date']).dt.year
df1.head()
MissingPercentage = (((df1.isna().sum())/df1.shape[0])*100)
MissingPercentage
# Exploratory look at the data. Focus on Manner of Death, Armed, Gender, Race, Threat Level and Flee
df1.manner_of_death.value_counts()
#Majority of individuals were 'just' shot and not tasered and shot. 
df1.armed.unique()

# Large variety of armed categories. Will have to be categorized in order to improve comprehension 
df1.armed.value_counts(normalize=True)

#we can see the majority of the armed categories were gun, knife, toy weapon and undetermined
df1.race.value_counts(normalize=True)
# White, Black and Hispanic accounted for 95.5% of all deaths. Might be worth focusing on them, and contrasting these three races with other races
df1.threat_level.value_counts(normalize=True)

# Majority of individuals killed attacked the Police. One observation is that the 'Other' and 'Undertermined' categories are very subjective.
df1.flee.value_counts(normalize=True)
# We can see that a large part of the individuals don't run from the police. 
# In order to facilitate our analysis, and understand if there is racial baisis in shootings, we will create categories for the following
# Armed = Will be categorized into Armed and Unarmed
# Fleeing = Will be categorized into Fleeing and Not Fleeing

list(df1.armed.unique())
UnavailableUndetermined = ['NaN','undetermined',]
Unarmed = ['unarmed']
Armed = ['gun',
 'toy weapon',
 'nail gun',
 'knife',
 'shovel',
 'hammer',
 'hatchet',
 'sword',
 'machete',
 'box cutter',
 'metal object',
 'screwdriver',
 'lawn mower blade',
 'flagpole',
 'guns and explosives',
 'cordless drill',
 'crossbow',
 'metal pole',
 'Taser',
 'metal pipe',
 'metal hand tool',
 'blunt object',
 'metal stick',
 'sharp object',
 'meat cleaver',
 'carjack',
 'chain',
 "contractor's level",
 'unknown weapon',
 'stapler',
 'beer bottle',
 'bean-bag gun',
 'baseball bat and fireplace poker',
 'straight edge razor',
 'gun and knife',
 'ax',
 'brick',
 'baseball bat',
 'hand torch',
 'chain saw',
 'garden tool',
 'scissors',
 'pole',
 'pick-axe',
 'flashlight',
 'vehicle',
 'baton',
 'spear',
 'chair',
 'pitchfork',
 'hatchet and gun',
 'rock',
 'piece of wood',
 'bayonet',
 'pipe',
 'glass shard',
 'motorcycle',
 'pepper spray',
 'metal rake',
 'crowbar',
 'oar',
 'machete and gun',
 'tire iron',
 'air conditioner',
 'pole and knife',
 'baseball bat and bottle',
 'fireworks',
 'pen',
 'chainsaw',
 'gun and sword',
 'gun and car',
 'pellet gun',
 'claimed to be armed',
 'BB gun',
 'incendiary device',
 'samurai sword',
 'bow and arrow',
 'gun and vehicle',
 'vehicle and gun',
 'wrench',
 'walking stick',
 'barstool',
 'grenade',
 'BB gun and vehicle',
 'wasp spray',
 'air pistol',
 'Airsoft pistol',
 'baseball bat and knife',
 'vehicle and machete',
 'ice pick',
 'car, knife and mace']
df_UnavailableUndetermined = pd.DataFrame({'armed': UnavailableUndetermined})
df_UnavailableUndetermined ['category'] = 'Unavailable_Undetermined'
df_UnavailableUndetermined
df_Unarmed = pd.DataFrame({'armed': Unarmed})
df_Unarmed ['category'] = 'Unarmed'
df_Unarmed
df_Armed = pd.DataFrame({'armed': Armed})
df_Armed ['category'] = 'Armed'
df_Armed
df_lookup2 = df_Armed
df_lookup2
df_lookup1 = df_lookup2.append(df_Unarmed)
df_lookup1.shape
df_lookup = df_lookup1.append(df_UnavailableUndetermined)
df_lookup
df2 = pd.merge(df1, df_lookup, on = 'armed', how = 'outer' )

df2 = df2.rename({'category':'armed_category'}, axis = 1)
df2.head()
df2.armed_category.value_counts(normalize = True)

df2.flee.unique()
Fleeing = ['Car', 'Foot', 'Other']
NotFleeing = ['Not fleeing']

FleeLookUp2 = pd.DataFrame({'flee': Fleeing})
FleeLookUp2['flee_category'] = "Fleeing"
FleeLookUp1 = pd.DataFrame({'flee': NotFleeing})
FleeLookUp1['flee_category'] = "Not_Fleeing"

FleeLookUp = FleeLookUp1.append(FleeLookUp2)
FleeLookUp.head()
df3 = pd.merge(df2,FleeLookUp,how='outer', on = 'flee')
df3.head()
df3.flee_category.value_counts(normalize=True)

df3.race.value_counts(normalize=True)
#As we've seen previously, the majority of crimes are committed by 3 racial groups. White, Black and Hispanic
df3.race.value_counts(normalize=True).plot(kind='pie', figsize = (8,8))
plt.title('Deaths by Race\nNormalized Data')
df3.state.value_counts(normalize=True)[:10]
df3.state.value_counts(normalize=True)[:10].sum()
#we can see that the top 10 states in the US account for 53.32% of all deaths in the US. Migh be worth focusing on these states to look for trends
df3.city.value_counts(normalize=True)[:10]
#Interesting topic: For the top 10 states, some capitals were not present in the top 10 cities, or the opposite where the city is in the top 10, but not the state. This is the case for:
# Denver/CO, Kansas City/Kansas,Oklahoma / Oklahoma City, Georgia/ Atlanta, North Carolina / Raleigh, Washington / Seattle

# I will make a few filtered data sets to evaluate only specific sections of the dataset related to race, state and city
RaceList = ['White', 'Black', 'Hispanic']
df3_race = df3[df3.race.isin(RaceList)]
df3_race.race.unique()
#StateList = ['CA','TX','FL','AZ','CO','GA','OK','NC','OH','WA']
#df3_race_state = df3_race[df3_race.state.isin(StateList)]
#df3_race_state.state.unique()
CityList = ['Los Angeles','Phoenix','Houston','Las Vegas','San Antonio','Columbus','Chicago','Albuquerque','Kansas City','Jacksonville']
df3_race_city = df3_race[df3_race.city.isin(CityList)]
df3_race_city.city.unique()
df3_race_city.groupby('race').city.value_counts(normalize=True).unstack().plot(kind='bar', figsize=(18,8))
plt.title('Deaths Per Race and City')
plt.ylabel('% of Total Deaths per Race')
df3_race_city.groupby('race').city.value_counts(normalize = True).unstack()
df3_race_state.groupby('race').state.value_counts(normalize=True).unstack().plot(kind='bar', figsize=(18,8))
plt.title('Deaths Per Race and State')
plt.ylabel('% of Total Deaths per Race')
df3_race_state.groupby('race').state.value_counts(normalize=True).unstack()

df3.groupby('race').armed_category.value_counts().unstack().plot(kind = 'bar', stacked=True,figsize = (15,6))
plt.title('Total Number of Armed Individuals by Race')


df3.groupby('race').armed_category.value_counts(normalize=True).unstack().plot(kind = 'bar', stacked=True,figsize = (15,6))
plt.title('Percentage of Armed Individuals by Race')


vis1b_df = df3.groupby('race').flee_category.value_counts(normalize=True).unstack()
vis1b_df
vis1b_df.plot(kind = 'bar', stacked = True, figsize=(15,6))
plt.title('Percentage of Individuals by Flee Category')
VIS1D = df3[df3.armed_category == 'Armed'].groupby('race').threat_level.value_counts(normalize=True).unstack().plot(kind='bar', stacked= True, figsize=(18,6))
plt.title('Likelyhood of Individual to Attack When Armed')


VIS1E = df3[df3.armed_category == 'Unarmed'].groupby('race').threat_level.value_counts(normalize=True).unstack().plot(kind='bar', stacked= True, figsize=(18,6))
plt.title('Likelyhood of Individual to Attack When Unarmed')


# We can see all races are less likely to attack police when unarmed. 
# Asians are least likely to attack police overall. 
# Black, Other and White are the most likely to attack police both Armed and Unarmed
df3.groupby('race').armed_category.value_counts(normalize = True).unstack()
df3[df3.flee_category == 'Fleeing'].groupby('race').armed_category.value_counts(normalize=True).unstack()
#As a surprise, Asians are the most likely to try to flee in case they are unarmed, followed by Black
df3[df3.flee_category == 'Fleeing'].groupby('race').armed_category.value_counts(normalize=True).unstack().plot(kind = 'bar', stacked=False,figsize = (12,6))
# Likelyhood of individual trying to flee in case they are armed or unarmed
df3.state.value_counts(normalize=False)[:10].plot(kind='pie', figsize=(10,10))
plt.title('Percentage of Deaths in Top 10 States')
VIS2A = df3_race_state[df3_race_state.armed_category == 'Armed'].groupby(['state','armed_category']).race.value_counts().unstack().plot(kind = 'bar', stacked=False, figsize = (18,6))
plt.title('Total Number of Individuals Killed When Armed, by State and Race')

VIS2B = df3_race_state[df3_race_state.armed_category == 'Unarmed'].groupby(['state','armed_category']).race.value_counts().unstack().plot(kind = 'bar', stacked=False, figsize=(18,6))
plt.title('Total Number of Individuals Killed When Unrmed, by State and Race')
df3.groupby(['armed_category','race']).threat_level.value_counts(normalize=True).unstack()

df3[df3.armed_category == 'Unarmed'].groupby('race').threat_level.value_counts(normalize=False).unstack().plot(kind='bar', figsize=(15,6))
plt.title('Number of Deaths of Unarmed Individuals categorized by Threat Level and Race')

#The owner of the dataset probably needs to be more specific on what 'Other' in Threat Level means, given that it was the largest category for all races
VIS2B = df3_race_state.groupby('race').state.value_counts(normalize=True).unstack().plot(kind = 'bar', figsize = (18,6))
#Where do most races die based on % of total deaths in top 10 states

df3.city.value_counts(normalize=False)[:10].plot(kind='pie', figsize=(10,10))
plt.title('Deadliest Cities in the US')
VIS3A = df3_race_city.groupby('race').city.value_counts(normalize=False).unstack().plot(kind='bar', figsize=(18,6))
plt.title('Deadliest Cities in the US by Race')
VIS3A
df3_race_city[df3_race_city.armed_category == 'Unarmed'].groupby(['city','armed_category','threat_level']).race.value_counts(normalize=False).unstack()
df3_race_city.groupby(['armed_category','race']).city.value_counts(normalize=False).unstack().plot(kind='bar', stacked=True, figsize=(18,8))
plt.title('Armed Category and Race of Individuals Killed in Deadliest Cities')
#trend remains the same in deadliest cities, with the majority individuals killed being armed

(df3.groupby('armed_category').flee_category.value_counts().unstack())
((df3.groupby('armed_category').flee_category.value_counts().unstack())/(df3.shape[0]))*100

# "Only" 3.5% of all deaths were related to unarmed civilians that were not fleeing. 


((df3.groupby(['armed_category','threat_level']).flee_category.value_counts().unstack())/(df3.shape[0]))*100

# "Only" 1.9% of all deaths were related to unarmed civilians that were not fleeing and were not attacking the police. 

ThreatLevelList = ['other', 'undetermined']

df_unarmed_nothreat_notfleeing = df3[(df3.threat_level.isin(ThreatLevelList)) & (df3.armed_category == 'Unarmed') & (df3.flee_category == 'Not_Fleeing')]
df_unarmed_nothreat_notfleeing.shape
df_unarmed_nothreat_notfleeing.race.value_counts(normalize=True)
ThreatLevelList = ['attack']

df_armed_threat_notfleeing = df3[(df3.threat_level.isin(ThreatLevelList)) & (df3.armed_category == 'Armed') & (df3.flee_category == 'Not_Fleeing')]
df_armed_threat_notfleeing.shape
df_armed_threat_notfleeing.race.value_counts(normalize=True)
ThreatLevelList = ['attack']

df_armed_threat_fleeing = df3[(df3.threat_level.isin(ThreatLevelList)) & (df3.armed_category == 'Armed') & (df3.flee_category == 'Fleeing')]
df_armed_threat_fleeing.shape
df_armed_threat_fleeing.race.value_counts(normalize=True)
# Percentage of killings per state, of citiezed that were unarmed, no threat and not fleeing

((df_unarmed_nothreat_notfleeing.state.value_counts(normalize=False)/df3.state.value_counts(normalize=False))*100).sort_values(ascending=False)
((df_unarmed_nothreat_notfleeing.state.value_counts()/df3.state.value_counts())*100).sort_values(ascending=False)[:10].plot(kind='pie', figsize=(10,10))

# Despite of low mortality rates in these states, the chance of being shot while unarmed, not posing threat and not fleeing is higher than in the states with higher total killings
df3.year.value_counts(normalize=True)
df3.groupby('month').race.value_counts(normalize=True).unstack().plot(kind='bar', figsize=(18,6))
df3.groupby('year').race.value_counts(normalize=True).unstack().plot(kind='bar', figsize=(18,6))
df3.groupby('race').body_camera.value_counts(normalize=False).unstack().plot(kind='bar', figsize=(18,8))
plt.title('Total Number of Fatalities Captured on Body Camera by Race')
df3.groupby('race').body_camera.value_counts(normalize=True).unstack().plot(kind='bar', figsize=(18,8))
plt.title('Percentage of Fatalities Captured on Body Camera by Race')


df3.groupby('race').body_camera.value_counts().unstack()
df3.groupby('race').body_camera.value_counts(normalize=True).unstack()
