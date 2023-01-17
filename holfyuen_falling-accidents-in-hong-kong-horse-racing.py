# Loading packages and data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

plt.style.use('seaborn-colorblind')

horse = pd.read_csv('../input/race-result-horse.csv')

race = pd.read_csv('../input/race-result-race.csv')
horse.head()
race.head()
print (horse.shape, race.shape)
print ('The dataset covers races from ' + race.race_date.min() + ' to ' + race.race_date.max())
plt.figure(figsize=(12,9))



plt.subplot(221)

temp = race.race_course.value_counts(ascending=True)

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.5, i, str(v), fontsize=12)

plt.title('Race Course')



plt.subplot(222)

temp = race.race_distance.value_counts(ascending=True)

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v+0.3, i-0.05, str(v), fontsize=12)

plt.title('Race Distance (m)')



plt.subplot(223)

temp = race.track.value_counts(ascending=True)

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v+0.5, i, str(v), fontsize=12)

plt.title('Track')



plt.subplot(224)

temp = race.track_condition.value_counts(ascending=True)

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.3, i-0.05, str(v), fontsize=12)

plt.title('Track Condition')



plt.show()
pd.crosstab(race.race_course, race.track)
pd.crosstab(race.race_course, race.track_condition)
pd.crosstab(race.track, race.track_condition)
race.loc[race.race_course=='Happy Valley','course_type'] = 'Happy Valley turf'

race.loc[race.track=='ALL WEATHER TRACK','course_type'] = 'Sha Tin all weather'

race.loc[(race.race_course=='Sha Tin') & (race.track!='ALL WEATHER TRACK'),'course_type'] = 'Sha Tin turf'

race.course_type.value_counts()
pd.crosstab(race.race_distance, race.course_type)
plt.figure(figsize=(8,6))

temp = race.race_class.value_counts(ascending=True)

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v+0.3, i, str(v), fontsize=12)

plt.title('Race Class')

plt.show()
def classify(x):

    if x[0:6]=='Class ':

        return x[0:7]

    elif x.find('Group') != -1 or x=='Restricted Race':

        return 'Group and Restricted'

    else:

        return x

race['class_adj'] = race['race_class'].apply(lambda x: classify(x))



plt.figure(figsize=(7,5))

temp = race.class_adj.value_counts(ascending=True)

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v+0.3, i, str(v), fontsize=12)

plt.title('Race Class (adjusted)')

plt.show()
horse.finishing_position.unique()
fall = horse.loc[horse.finishing_position.isin(['FE','UR']),:]

print ('No of races with falling incidents: ' + str(fall.race_id.nunique()))

print ('No of horses involved with falling incidents: ' + str(fall.shape[0]))

fall_p = fall.race_id.nunique()*1.0/race.race_id.nunique()

print ('Proportion of races with falling indicents: '+ '{:.2%}'.format(fall_p))
fall.race_id.value_counts().head()
fall.horse_name.nunique()
race_w_fall = fall.race_id.unique().tolist()

race['has_fall'] = race['race_id'].isin(race_w_fall)*1

races_fallen = race.loc[race.has_fall==1,:]

races_fallen.head()
plt.figure(figsize=(14,5))

plt.subplot(121)

temp = races_fallen.course_type.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-2, i, str(v), fontsize=12)

plt.title('Accident Count by Race Course')



plt.subplot(122)

temp = (race.groupby('course_type').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Race Course')

plt.axvline(x=fall_p*100)



plt.show()
plt.figure(figsize=(14,5))

plt.subplot(121)

temp = races_fallen.race_distance.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-1, i, str(v), fontsize=12)

plt.title('Accident Count by Race Distance')



plt.subplot(122)

temp = (race.groupby('race_distance').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Race Distance')

plt.axvline(x=fall_p*100)



plt.show()
plt.figure(figsize=(14,5))

plt.subplot(121)

temp = races_fallen.class_adj.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-1, i, str(v), fontsize=12)

plt.title('Accident Count by Race Class')



plt.subplot(122)

temp = (race.groupby('class_adj').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Race Class')

plt.axvline(x=fall_p*100)



plt.show()
plt.figure(figsize=(16,10))



plt.subplot(221)

temp = races_fallen[races_fallen.course_type!='Sha Tin all weather'].track_condition.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-1, i, str(v), fontsize=12)

plt.title('Accident Count by Track Condition (Turf)')



plt.subplot(222)

temp = (race[race.course_type!='Sha Tin all weather'].groupby('track_condition').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.02, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Track Condition (Turf)')

plt.axvline(x=fall_p*100)



plt.subplot(223)

temp = races_fallen[races_fallen.course_type=='Sha Tin all weather'].track_condition.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-0.5, i, str(v), fontsize=12)

plt.title('Accident Count by Track Condition (All Weather)')



plt.subplot(224)

temp = (race[race.course_type=='Sha Tin all weather'].groupby('track_condition').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.02, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Track Condition (All Weather)')

plt.axvline(x=fall_p*100)



plt.show()
pd.crosstab(races_fallen.track, races_fallen.course_type)
plt.figure(figsize=(16,10))



plt.subplot(221)

temp = races_fallen[races_fallen.course_type=='Sha Tin turf'].track.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-0.5, i, str(v), fontsize=12)

plt.title('Accident Count by Track (Sha Tin Turf)')



plt.subplot(222)

temp = (race[race.course_type=='Sha Tin turf'].groupby('track').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.02, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Track (Sha Tin Turf)')

plt.axvline(x=fall_p*100)



plt.subplot(223)

temp = races_fallen[races_fallen.course_type=='Happy Valley turf'].track.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-0.3, i, str(v), fontsize=12)

plt.title('Accident Count by Track (Happy Valley Turf)')



plt.subplot(224)

temp = (race[race.course_type=='Happy Valley turf'].groupby('track').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.02, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Track (Happy Valley Turf)')

plt.axvline(x=fall_p*100)



plt.show()
horse['starter'] = horse.finishing_position.apply(lambda x: 1- (str(x)[0]=="W" or str(x)=='nan'))

in_race = horse.groupby('race_id').sum()['starter']

in_race = in_race.reset_index()

race = race.merge(in_race, how='left', on='race_id')

race.head()
pd.crosstab(race.starter, race.course_type, margins=True)
races_fallen = race.loc[race.has_fall==1,:]



plt.figure(figsize=(14,5))

plt.subplot(121)

temp = races_fallen.starter.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v-1, i, str(v), fontsize=12)

plt.title('Accident Count by No. of Starters')



plt.subplot(122)

temp = (race.groupby('starter').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by No. of Starters')

plt.axvline(x=fall_p*100)



plt.show()
pd.crosstab(races_fallen.starter, races_fallen.course_type, margins=True)
plt.figure(figsize=(16,5))

plt.subplot(131)

temp = (race[race.course_type=='Sha Tin turf'].groupby('starter').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.xlabel('% with Fall')

plt.title('Sha Tin Turf')

plt.axvline(x=fall_p*100)



plt.subplot(132)

temp = (race[race.course_type=='Happy Valley turf'].groupby('starter').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.xlabel('% with Fall')

plt.title('Happy Valley Turf')

plt.axvline(x=fall_p*100)



plt.subplot(133)

temp = (race[race.course_type=='Sha Tin all weather'].groupby('starter').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.xlabel('% with Fall')

plt.title('Sha Tin All Weather')

plt.axvline(x=fall_p*100)



plt.show()
plt.figure(figsize=(14,5))

plt.subplot(121)

temp = races_fallen.race_number.value_counts().sort_index()

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v+0.1, i, str(v), fontsize=12)

plt.title('Accident Count by Race Number')



plt.subplot(122)

temp = (race.groupby('race_number').mean()['has_fall']*100).sort_index()

g = temp.plot(kind='barh', color='forestgreen')

for i, v in enumerate(temp):

    g.text(v+0.05, i, str(round(v,2)), fontsize=12)

plt.ylabel('')

plt.title('% with Fall by Race Number')

plt.axvline(x=fall_p*100)



plt.show()
fall = horse.loc[horse.finishing_position.isin(['FE','UR']),:]

fall = fall.merge(race.loc[:,['race_id','race_distance','course_type']], how='left', on='race_id')
dist_sec_map = {1000:3, 1200:3, 1400:4, 1600:4, 1650:4, 1800:5, 2000:5, 2200:6, 2400:6}



def get_fall_pos(temp):

    sections = dist_sec_map[temp['race_distance']]

    last = 'running_position_' + str(sections)

    bflast = 'running_position_' + str(sections-1)

    bf2last = 'running_position_' + str(sections-2)

    if np.isnan(temp['running_position_1']):

        return "At start"

    elif np.isnan(temp[last]) and ~np.isnan(temp[bflast]):

        return "Final 400"

    elif np.isnan(temp[bflast]) and ~np.isnan(temp[bf2last]):

        return "Final 800"

    else:

        return "Others"



for index, row in fall.iterrows():

    fall.loc[index, 'fall_pos'] = get_fall_pos(row)
plt.figure(figsize=(7,5))

temp = fall.fall_pos.value_counts(ascending=True)

g = temp.plot(kind='barh', color='darkseagreen')

for i, v in enumerate(temp):

    g.text(v+0.3, i, str(v), fontsize=12)

plt.title('Section Where the Falling Happened')

plt.show()