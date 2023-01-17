import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt 

from collections import Counter

import plotly.express as px

import seaborn as sns

import datetime



df=pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')
df.sample(10)
df.info()
df['date']=pd.to_datetime(df['date'], format = "%Y-%m-%d")
df['months']=df['date'].dt.month

df['years']=df['date'].dt.year
df.isnull().sum()
df.loc[df['armed'].isnull()]
df['armed'].unique()
df['armed']=df['armed'].str.lower()

gun=['gun', 'guns and explosives','crossbows','gun and knife','hatchet and gun','machete and gun','gun and sword', 'gun and car','incendiary device','gun and vehicle','vehicle and gun','grenade','crossbow']

perforating_weapon=['nail gun','knife','hatchet','baseball bat and knife','sword', 'machete','box cutter','screwdriver','lawn mower blade','sharp object','meat cleaver','beer bottle','straight edge razor','ax','chain saw', 'garden tool', 'scissors','pick-axe','spear','pitchfork','bayonet','glass shard','metal rake','crowbar','pole and knife','pen','chainsaw','samurai sword', 'bow and arrow','ice pick','pellet gun']

no_perforating_weapon=['shovel','hammer','metal object','flagpole','cordless drill','metal pole', 'metal pipe', 'metal hand tool','blunt object','metal stick','chain', "contractor's level",'stapler','bean-bag gun','baseball bat and fireplace poker', 'brick', 'baseball bat', 'hand torch','pole','flashlight','baton','chair','rock', 'piece of wood','pipe','oar', 'tire iron','air conditioner','baseball bat and bottle','fireworks','wrench','walking stick','barstool']

vehicle=['vehicle','carjack','motorcycle','vehicle and machete','car, knife and mace']

non_lethal=['taser','wasp spray','pepper spray']

fake_gun=['claimed to be armed','toy weapon','bb gun and vehicle','air pistol','airsoft pistol','bb gun']

undertermined=['undertemined','unknown weapon',np.nan]

for i,x in enumerate(df['armed']):

    if x in gun:

         df['armed'][i]='Gun'

    elif x in perforating_weapon:

         df['armed'][i]='Perforating_weapon'   

    elif x in no_perforating_weapon:

         df['armed'][i]='no_perforating_weapon' 

    elif x in vehicle:

         df['armed'][i]='Vehicle'     

    elif x in non_lethal:

         df['armed'][i]='Non_lethal' 

    elif x in fake_gun:

         df['armed'][i]='Fake_gun' 

    elif x in undertermined:

         df['armed'][i]='undetermined'
(df['age'].isnull().sum()/df['age'].shape[0])*100
for x in df.loc[df['age'].isnull(),['city','state','id']].values:

    city,state,i=x

    if df.loc[df['city']==city,'age'].median()>10:

        df.loc[df['id']==i,'age']=df.loc[df['city']==city,'age'].median()

    else:

         df.loc[df['id']==i,'age']=df.loc[df['state']==state,'age'].median()
df['race'].fillna('U',inplace=True)
df.dropna(inplace=True)
fig, ax1 = plt.subplots(figsize=(10,8))

race_types=df['race'].value_counts().index

amount_race=df['race'].value_counts().values

ax1.bar(race_types,amount_race)
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(15,5))



days=df['date'].dt.day



sns.distplot(df['years'].values,kde=False,bins=10,ax=ax2)

ax2.set_xlabel('Year')

sns.distplot(df['months'].values,kde=False,bins=12,ax=ax1)

ax1.set_xlabel('Months')

sns.distplot(days,kde=False,bins=10,ax=ax3)

ax3.set_xlabel('Days')
df.info()
fig, ax1 = plt.subplots(figsize=(8,8))

x=df['armed'].value_counts().index.values

y=df['armed'].value_counts().values

ax1.bar(x,y)

plt.xticks(rotation=20)
fig, ax1 = plt.subplots(figsize=(8,8))

x=df['gender'].value_counts().index

y=df['gender'].value_counts().values

ax1.bar(x,y)
fig, ax1 = plt.subplots(figsize=(10,8))

ax1.set_xlabel('Ages')

ax1.set_ylabel('Amount')

age_values=Counter(min(x//10*10,90) for x in df['age'].values )

ax1=plt.bar(age_values.keys(),age_values.values(),width=8)

plt.xticks([10 * i for i in range(11)])
y=df.state.value_counts().values[0:5]

x=df.state.value_counts().index[0:5]

plt.bar(x,y)
y=df.city.value_counts().values[0:5]

x=df.city.value_counts().index[0:5]

plt.bar(x,y)
ages=Counter(min(x//10*10,90) for x in df['age'].values )

df['count']=1

fig = px.bar(df, x="age", y='count', color="signs_of_mental_illness")

fig.show()
threat_level = df[['race','threat_level']]

threat_level ['kills'] =1

threat_level  = threat_level .groupby(['race','threat_level']).sum()

threat_level = threat_level.reset_index()

fig = px.bar(threat_level , y='kills', x='threat_level',color='race', barmode='group')

fig.show()