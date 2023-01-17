# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shootings=pd.read_csv('/kaggle/input/us-police-shootings/shootings.csv')

shootings.head()
shootings.shape
shootings.dtypes
shootings['date']=pd.to_datetime(shootings['date'])

shootings.dtypes
shootings.isnull().sum()
yrs_count=shootings.groupby('date').apply(lambda x:x['name'].count()).reset_index(name='Count')

plt.figure(figsize=(15,15))

plt.scatter(yrs_count['date'],yrs_count['Count'],color='r')

plt.xlabel('Date',size=15)

plt.ylabel('Death Counts',size=15)

plt.title('Death over the Years',size=20)
People_death=shootings.name.nunique()

Death_avg_age=np.average(shootings.age)

print('{} People with an average age of {} have been killed in Shooting'.format(People_death,round(Death_avg_age,2)))
age_death=shootings.groupby('age').apply(lambda x:x['name'].count()).reset_index(name='Counts')

sns.regplot(age_death['age'],age_death['Counts'],fit_reg=True)

plt.title('Death Counts by Age',size=20)
gender_dist=shootings.groupby('gender').apply(lambda x:x['name'].count()).reset_index(name='Counts')

plt.bar(gender_dist['gender'],gender_dist['Counts'],color='br')

plt.xlabel('Gender',size=15)

plt.ylabel('# of Shooting Deaths',size=15)

plt.title('Genderwise Distribution of Shooting Deaths',size=20)
shootings['year'] = pd.DatetimeIndex(shootings['date']).year

gender_yr_dist=shootings.groupby(['year','gender']).apply(lambda x:x['name'].count()).reset_index(name='Counts')

female_yr_dist=gender_yr_dist[gender_yr_dist['gender']=='F']

male_yr_dist=gender_yr_dist[gender_yr_dist['gender']=='M']

plt.plot(male_yr_dist['year'],male_yr_dist['Counts'],color='r')

plt.plot(female_yr_dist['year'],female_yr_dist['Counts'],color='b')

plt.legend(['Male','Female'])

plt.xlabel('Year',size=15)

plt.ylabel('Gender wise Death Counts',size=15)

plt.title('Genderwise Death Counts over the years',size=20)

armed_threat=shootings.groupby(['armed','threat_level']).apply(lambda x:x['name'].count()).reset_index(name='Deaths')

pivot_attack=pd.pivot(armed_threat,index='armed',columns='threat_level',values='Deaths')

plt.figure(figsize=(5,25))

sns.heatmap(pivot_attack,annot=True,fmt='.0f',cmap='GnBu')

plt.xlabel('Threat Level',size=10)

plt.ylabel('Attacking Weapon',size=15)

plt.title('Attacking Weapon vs Treat from Victims',size=25)
armed_threat=shootings.groupby(['arms_category','flee']).apply(lambda x:x['name'].count()).reset_index(name='Deaths')

pivot_attack=pd.pivot(armed_threat,index='arms_category',columns='flee',values='Deaths')

plt.figure(figsize=(5,25))

sns.heatmap(pivot_attack,annot=True,fmt='.0f',cmap='RdBu')

plt.xlabel('Threat Level',size=10)

plt.ylabel('Attacking Weapon',size=15)

plt.title('Attacking Weapon vs Treat from Victims',size=25)
mental_attack=shootings.groupby(['signs_of_mental_illness','threat_level']).apply(lambda x:x['name'].count()).reset_index(name='Counts')

sick_threat=pd.pivot(mental_attack,index='threat_level',columns='signs_of_mental_illness',values='Counts')

sns.heatmap(sick_threat,annot=True,fmt='.0f',cmap='pink')
yr_race=shootings.groupby(['year','race']).apply(lambda x:x['name'].count()).reset_index(name='Counts')

pivoted_yr_race=pd.pivot(yr_race,columns='year',index='race',values='Counts')

plt.figure(figsize=(6,6))

plot=sns.heatmap(pivoted_yr_race,annot=pivoted_yr_race.values,fmt='d',cmap='YlOrBr')

plot.set_xlabel('Year',size=15)

plot.set_ylabel('Race',size=15)

plot.set_title('Count of Shooting Deaths by Race- Over the years',size=15)
ct_total_shoot=shootings.groupby(['city']).apply(lambda x:x['name'].count()).reset_index(name='Shoots')

ct_max=ct_total_shoot.sort_values(by='Shoots', ascending=False)

max_shoot=ct_max[0:10] # Only considering 10 cities- taht have the maximum Shooting Death Counts

print('10 Cities with the most shooting deaths:\n',max_shoot)

city_target=shootings[shootings['city'].isin(max_shoot['city'])]

ct_shoot=city_target.groupby(['year','city']).apply(lambda x:x['name'].count()).reset_index(name='Shoot')

shoot_ct=pd.pivot(ct_shoot,values='Shoot',index='year',columns='city')

shoot_ct.replace(np.NaN,0,inplace=True)

shoot_ct.reset_index(inplace=True)

plt.figure(figsize=(10,10))

shoot_ct.reset_index(inplace=True)

for i in ct_shoot['city']:

    plt.plot(shoot_ct['year'],shoot_ct[i])

    plt.scatter(shoot_ct['year'],shoot_ct[i])

plt.legend(max_shoot['city'])

plt.xlabel('Year',size=15)

plt.ylabel('Shoot Count',size=15)

plt.title('Yearwise counts in the Cities with most Shooting Deaths',size=20)
state_summ=shootings.groupby(['state']).apply(lambda x:x['manner_of_death'].count()).reset_index(name='Counts')

state_kill_max=state_summ.sort_values(by='Counts',ascending=False)

Top_state_kill=state_kill_max[:15]

print('States in US with the most number of Police Shootings\n',Top_state_kill)

state_yr_summ=shootings.groupby(['year','state']).apply(lambda x:x['manner_of_death'].count()).reset_index(name='Counts')

pivot_st_yr=pd.pivot(state_yr_summ,columns='year',index='state',values='Counts')

pivot_st_yr.replace(np.NaN,0)

plt.figure(figsize=(10,20))

sns.heatmap(pivot_st_yr,annot=True,fmt='.0f',cmap='RdPu')

plt.xlabel('Year',size=15)

plt.ylabel('State',size=15)

plt.title('Yearwise Shooting Deaths in US- States Data',size=20)
import geopandas as gpd

fp = "/kaggle/input/us-shape-files/USA_States.shp"

map_df = gpd.read_file(fp)

merged = map_df.set_index('STATE_ABBR').join(state_summ.set_index('state'))

variable = 'Counts'

vmin, vmax = np.min(merged.loc[:,['Counts']]), np.max(merged.loc[:,['Counts']])

fig, ax = plt.subplots(1,figsize=(20,20))

pt=merged.plot(column=variable, cmap='Reds',ax=ax,linewidth=0.8, edgecolor='0.8')

ax.axis('on')

ax.set_title('US Statewise Shooting Counts', fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate('Source:US Police Shootings',xy=(0.01, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=20, color='#555555')

sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm,ax=ax)