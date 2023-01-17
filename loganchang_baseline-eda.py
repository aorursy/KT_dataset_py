# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import math

from collections import OrderedDict 

# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import itertools

N = 10

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_beat = pd.read_csv('/kaggle/input/clean-cpp-data/df_beat_clean.csv')

df_subject_id = pd.read_csv('/kaggle/input/clean-cpp-data/df_subject_id.csv')

df_trr_by_beat = pd.read_csv('/kaggle/input/clean-cpp-data/df_trr_by_beat_clean.csv')

df_trr_id = pd.read_csv('/kaggle/input/clean-cpp-data/df_trr_id_final-2.csv')
df_beat.columns
df_beat.head()
df_subject_id.columns
df_subject_id.head()
df_trr_by_beat.columns
df_trr_by_beat.head()
df_trr_id.columns
df_trr_id.head(10)
#table of 20 beats with highest reported number of crimes

df_beat.sort_values(by = 'number_of_reported_crimes', ascending = False, inplace = True)

df_beat.head(20)
#graph of above table

top20_beat = df_beat.beat[:20]

#print(top20_beat)

top20_crime = df_beat.number_of_reported_crimes[:20]

sns.set(font_scale = 1)

ax = sns.barplot(x = top20_beat, y = top20_crime, order = top20_beat, label = 'small')

ax.set_title('20 Beats with HIGHEST Number of Reported Crimes')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

plt.show()
#table showing 20 beats with lowest number of reported crimes

df_beat.tail(20)
#graph of above table

last20_beat = df_beat.beat[-20:]

#print(last20_beat)

last20_crime = df_beat.number_of_reported_crimes[-20:]

sns.set(font_scale = 1)

ax = sns.barplot(x = last20_beat, y = last20_crime, order = last20_beat, label = 'small')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

ax.set_title('20 Beats with Fewest Number of Reported Crimes')

plt.show()
#create counter for the 20 beats with highest number of crimes

top20_crime = {}

for ind in df_beat.index[:20]:

    beat = df_beat['beat'][ind]

    numCrimes = df_beat['number_of_reported_crimes'][ind]

    top20_crime[beat] = numCrimes

print(top20_crime)
#sort counter by district

top20_crime_district = {}

for beat in top20_crime:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in top20_crime_district.keys():

        top20_crime_district[district] += 1

    else:

        top20_crime_district[district] = 1

#print(top20_crime_district)

top20_crime_district_sorted_keys = sorted(top20_crime_district, key=top20_crime_district.get, reverse=True)

top20_crime_district_sort ={}

for district in top20_crime_district_sorted_keys:

    top20_crime_district_sort[district] = top20_crime_district[district]

print(top20_crime_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(top20_crime_district_sort.keys()), y = list(top20_crime_district_sort.values()), order = top20_crime_district_sort, label = 'small')

ax.set_title('Districts with HIGHEST Number of Crimes')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
#create counter for the 20 beats with lowest number of crimes

low20_crime = {}

for ind in df_beat.index[-20:]:

    beat = df_beat['beat'][ind]

    numCrimes = df_beat['number_of_reported_crimes'][ind]

    low20_crime[beat] = numCrimes

print(low20_crime)
#sort this counter by district

low20_crime_district = {}

for beat in low20_crime:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in low20_crime_district.keys():

        low20_crime_district[district] += 1

    else:

        low20_crime_district[district] = 1

#print(low20_crime_district)

low20_crime_district_sorted_keys = sorted(low20_crime_district, key=low20_crime_district.get, reverse=True)

low20_crime_district_sort ={}

for district in low20_crime_district_sorted_keys:

    low20_crime_district_sort[district] = low20_crime_district[district]

print(low20_crime_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(low20_crime_district_sort.keys()), y = list(low20_crime_district_sort.values()), order = low20_crime_district_sort, label = 'small')

ax.set_title('Districts with LOWEST Number of Crimes')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
#table showing 20 beats with highest number of arrests

df_beat.sort_values(by = 'arrests_by_beat', ascending = False, inplace = True)

df_beat.head(20)
#graph of above table

top20_beat = df_beat.beat[:20]

top20_arrest = df_beat.arrests_by_beat[:20]

#print(top20_beat)

sns.set(font_scale = 1)

ax = sns.barplot(x = top20_beat, y = top20_arrest, order = top20_beat, label = 'small')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

ax.set_title('20 Beats with HIGHEST Number of Arrests')

plt.show()
#table showing 20 beats with lowest number of arrests

df_beat.tail(20)
#graph of above table

last20_beat = df_beat.beat[-20:]

last20_arrest = df_beat.arrests_by_beat[-20:]

#print(last20_beat)

sns.set(font_scale = 1)

ax = sns.barplot(x = last20_beat, y = last20_arrest, order = last20_beat, label = 'small')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

ax.set_title('20 Beats with LOWEST Number of Arrests')

plt.show()
#create counter for the 20 beats with highest number of arrests

top20_arrest = {}

for ind in df_beat.index[:20]:

    beat = df_beat['beat'][ind]

    numarrests = df_beat['arrests_by_beat'][ind]

    top20_arrest[beat] = numarrests

print(top20_arrest)
#sort counter by district

top20_arrest_district = {}

for beat in top20_arrest:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in top20_arrest_district.keys():

        top20_arrest_district[district] += 1

    else:

        top20_arrest_district[district] = 1

#print(top20_arrest_district)

top20_arrest_district_sorted_keys = sorted(top20_arrest_district, key=top20_arrest_district.get, reverse=True)

top20_arrest_district_sort ={}

for district in top20_arrest_district_sorted_keys:

    top20_arrest_district_sort[district] = top20_arrest_district[district]

print(top20_arrest_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(top20_arrest_district_sort.keys()), y = list(top20_arrest_district_sort.values()), order = top20_arrest_district_sort, label = 'small')

ax.set_title('Districts with HIGHEST Number of arrests')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
#create counter for the 20 beats with highest number of arrests

low20_arrest = {}

for ind in df_beat.index[-20:]:

    beat = df_beat['beat'][ind]

    numarrests = df_beat['arrests_by_beat'][ind]

    low20_arrest[beat] = numarrests

print(low20_arrest)
#sort coutner by district

low20_arrest_district = {}

for beat in low20_arrest:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in low20_arrest_district.keys():

        low20_arrest_district[district] += 1

    else:

        low20_arrest_district[district] = 1

#print(low20_arrest_district)

low20_arrest_district_sorted_keys = sorted(low20_arrest_district, key=low20_arrest_district.get, reverse=True)

low20_arrest_district_sort ={}

for district in low20_arrest_district_sorted_keys:

    low20_arrest_district_sort[district] = low20_arrest_district[district]

print(low20_arrest_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(low20_arrest_district_sort.keys()), y = list(low20_arrest_district_sort.values()), order = low20_arrest_district_sort, label = 'small')

ax.set_title('Districts with LOWEST Number of Arrests')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
df_beat.columns
#creating the ratio column

df_beat['arrest_to_crime_ratio'] = df_beat.apply(lambda x: x.arrests_by_beat/x.number_of_reported_crimes, axis = 1)
#table showing 20 beats with highest arrest:crime ratio

df_beat.sort_values(by = 'arrest_to_crime_ratio', ascending = False, inplace = True)

df_beat.head(20)
#graphing above data

top20_beat = df_beat.beat[:20]

top20_ratio = df_beat.arrest_to_crime_ratio[:20]

#print(top20_beat)

sns.set(font_scale = 1)

ax = sns.barplot(x = top20_beat, y = top20_ratio, order = top20_beat, label = 'small')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

ax.set_title('20 Beats with HIGHEST Arrest to Number of Reported Crimes Ratio')

plt.show()
#table showing 20 beats with lowest arrest:crime ratio

df_beat.tail(20)
#graphing above table

low20_beat = df_beat.beat[-20:]

low20_ratio = df_beat.arrest_to_crime_ratio[-20:]

#print(low20_beat)

sns.set(font_scale = 1)

ax = sns.barplot(x = low20_beat, y = low20_ratio, order = low20_beat, label = 'small')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

ax.set_title('20 Beats with LOWEST Arrest to Reported Crimes Ratio')

plt.show()
#create counter for the 20 beats with highest number of ratios

top20_ratio = {}

for ind in df_beat.index[:20]:

    beat = df_beat['beat'][ind]

    numratios = df_beat['arrest_to_crime_ratio'][ind]

    top20_ratio[beat] = numratios

print(top20_ratio)
top20_ratio_district = {}

for beat in top20_ratio:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in top20_ratio_district.keys():

        top20_ratio_district[district] += 1

    else:

        top20_ratio_district[district] = 1

#print(top20_ratio_district)

top20_ratio_district_sorted_keys = sorted(top20_ratio_district, key=top20_ratio_district.get, reverse=True)

top20_ratio_district_sort ={}

for district in top20_ratio_district_sorted_keys:

    top20_ratio_district_sort[district] = top20_ratio_district[district]

print(top20_ratio_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(top20_ratio_district_sort.keys()), y = list(top20_ratio_district_sort.values()), order = top20_ratio_district_sort, label = 'small')

ax.set_title('Districts with HIGHEST Arrest to Crime Ratio')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
#create counter for the 20 beats with lowest arrest to crime ratio

low20_ratio = {}

for ind in df_beat.index[-20:]:

    beat = df_beat['beat'][ind]

    numratio = df_beat['arrest_to_crime_ratio'][ind]

    low20_ratio[beat] = numratio

print(low20_ratio)
#sort coutner by district

low20_ratio_district = {}

for beat in low20_ratio:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in low20_ratio_district.keys():

        low20_ratio_district[district] += 1

    else:

        low20_ratio_district[district] = 1

#print(low20_ratio_district)

low20_ratio_district_sorted_keys = sorted(low20_ratio_district, key=low20_ratio_district.get, reverse=True)

low20_ratio_district_sort ={}

for district in low20_ratio_district_sorted_keys:

    low20_ratio_district_sort[district] = low20_ratio_district[district]

print(low20_ratio_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(low20_ratio_district_sort.keys()), y = list(low20_ratio_district_sort.values()), order = low20_ratio_district_sort, label = 'small')

ax.set_title('Districts with LOWEST Number of ratios')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
print('TOP 20 BEATS IN CRIME, ARREST, AND RATIO\n')

print('CRIME: ')

print(top20_crime_district_sort)

print('-'*60)

print('ARREST: ')

print(top20_arrest_district_sort)

print('-'*60)

print('RATIO: ')

print(top20_ratio_district_sort)

print('-'*60)
print('BOTTOM 20 BEATS IN CRIME, ARREST, AND RATIO\n')

print('CRIME: ')

print(low20_crime_district_sort)

print('-'*60)

print('ARREST: ')

print(low20_arrest_district_sort)

print('-'*60)

print('RATIO: ')

print(low20_ratio_district_sort)

print('-'*60)
df_subject_id.head()
df_subject_id.race.value_counts()
#graphing this subject breakdown by race

ax = sns.barplot(x = df_subject_id.race.value_counts().index, y = df_subject_id.race.value_counts(), label = 'small')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.set_xlabel('Race')

ax.set_ylabel('Number of Subjects')

ax.tick_params(labelsize=8.5)

ax.set_title('Number of Subjects by Race')

plt.show()
#making the aforementioned column

df_subject_id['number_of_trr'] = df_subject_id.list_of_trr_id.apply(lambda x: len(x.split(','))-1)

df_subject_id.head()
#we have a null value for race for some reason

#assuming this means the subjects race was either undisclosed or not one of the 5 main races, we just call it 'OTHER'

df_subject_id.race.replace(np.nan, 'OTHER', inplace = True)

df_subject_id.race.unique()
#counting and storing the number of trr incidents by race

trr_by_race_dict ={}

for ind in df_subject_id.index:

    race = df_subject_id['race'][ind]

    num_trr = df_subject_id['number_of_trr'][ind]

    if race in trr_by_race_dict.keys():

        trr_by_race_dict[race] += num_trr

    else:

        trr_by_race_dict[race] = num_trr

out = dict(itertools.islice(trr_by_race_dict.items(), N)) 

print(out)
#graphing our data

plt.bar(x = trr_by_race_dict.keys(), height =trr_by_race_dict.values())

plt.title('Number of TRR Incidents by Race')

plt.xlabel('Race')

plt.ylabel("Number of TRR Incidents")

plt.xticks(rotation=90)

plt.show()
#estbalish our age bands and count the number of suspects in each age band

age_band_dict = {}

age_band_dict['<20'] = 0

age_band_dict['20-29'] = 0

age_band_dict['30-39'] = 0

age_band_dict['40-49'] = 0

age_band_dict['50-59'] = 0

age_band_dict['60-69'] = 0

age_band_dict['>=70'] = 0

for ind in df_subject_id.index:

    age = df_subject_id['age'][ind]

    if (age<20):

        age_band_dict['<20'] += 1

    elif(age>=20 and age<30):

        age_band_dict['20-29'] += 1

    elif(age>=30 and age<40):

        age_band_dict['30-39'] += 1

    elif(age>=40 and age<50):

        age_band_dict['40-49'] += 1

    elif(age>=50 and age<60):

        age_band_dict['50-59'] += 1

    elif(age>=60 and age<70):

        age_band_dict['60-69'] += 1

    else:

        age_band_dict['>=70'] += 1

out = dict(itertools.islice(age_band_dict.items(), N)) 

print(out)
#helper function to apply the age banding

def bander(age):

    if (age<20):

        return '<20'

    elif(age>=20 and age<30):

        return '20-29'

    elif(age>=30 and age<40):

        return '30-39'

    elif(age>=40 and age<50):

        return '40-49'

    elif(age>=50 and age<60):

        return '50-59'

    elif(age>=60 and age<70):

        return '60-69'

    else:

        return '>=70'
#creating a new column in our data table for the age banding and applying it to each subject

df_subject_id['age_bands'] = df_subject_id.age.apply(lambda x: bander(x))
#graphing the frequencies of the age bands

plt.bar(x = age_band_dict.keys(), height =age_band_dict.values())

plt.title('Number of Subjects by Age')

plt.xlabel('Age Bands')

plt.ylabel("Number of Subjects")

plt.show()
#graphing the subjects by frequency of gender

ax = sns.countplot(x="gender", data=df_subject_id)

ax.set_title('Number of Subject by Gender')

plt.show()
#first, males

df_subject_id_males = df_subject_id.loc[df_subject_id.gender == 'MALE']
#sorting the males by race

trr_by_race_dict_male ={}

for ind in df_subject_id_males.index:

    race = df_subject_id_males['race'][ind]

    num_trr = df_subject_id_males['number_of_trr'][ind]

    if race in trr_by_race_dict_male.keys():

        trr_by_race_dict_male[race] += num_trr

    else:

        trr_by_race_dict_male[race] = num_trr

dict1 = OrderedDict(sorted(trr_by_race_dict_male.items())) 

out = dict(itertools.islice(dict1.items(), N)) 

print(out)
#now the ladies

df_subject_id_females = df_subject_id.loc[df_subject_id.gender == 'FEMALE']
#sorting females by race

trr_by_race_dict_female ={}

for ind in df_subject_id_females.index:

    race = df_subject_id_females['race'][ind]

    num_trr = df_subject_id_females['number_of_trr'][ind]

    if race in trr_by_race_dict_female.keys():

        trr_by_race_dict_female[race] += num_trr

    else:

        trr_by_race_dict_female[race] = num_trr

dict2 = OrderedDict(sorted(trr_by_race_dict_female.items())) 

out = dict(itertools.islice(dict2.items(), N)) 

print(out)
#combining the two on a bar chart

n = 6

index = np.arange(n)

width = .3

fig, ax = plt.subplots()

y1 = list(dict1.values())

y2 = list(dict2.values())

rects1 = ax.bar(index, y1, width, color='r')

rects2 = ax.bar(index + width, y2, width, color='y')

ax.set_ylabel('Number of TRR Incidents')

ax.set_title('TRR Incidents by Gender and Race')

ax.set_xticks(index + width / 2)

#ax.xticks(rotation=90)

ax.set_xticklabels(('ASIAN/PACIFIC ISLANDER', 'BLACK', 'HISPANIC', 'NATIVE AMERICAN/ALASKAN NATIVE', 'OTHER', 'WHITE'), rotation = 90)

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))

plt.show()
df_subject_id.columns = ['subject_ID', 'list_of_trr_id', 'gender', 'race', 'age','number_of_trr', 'age_bands']
#a really useful table to sum up our findings in this dataset

pd.set_option('display.max_rows', None)

table = pd.pivot_table(df_subject_id, values='number_of_trr', index=['race', 'gender', 'age_bands'], aggfunc=np.sum)

table
df_trr_by_beat.head()
#show table sorted by 20 beats with highest number of trr reports

df_trr_by_beat.sort_values(by = 'number_of_trr_reports', ascending = False, inplace = True)

df_trr_by_beat.head(20)
#graph above table

top20_beat = df_trr_by_beat.beat[:20]

#print(top20_beat)

top20_trr = df_trr_by_beat.number_of_trr_reports[:20]

sns.set(font_scale = 1)

ax = sns.barplot(x = top20_beat, y = top20_trr, order = top20_beat, label = 'small')

ax.set_title('20 Beats with HIGHEST Number of TRR Reports')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

plt.show()
#show table with 20 beats with lowest number of trr reports

df_trr_by_beat.tail(20)
#graph above table

low20_beat = df_trr_by_beat.beat[-20:]

#print(top20_beat)

low20_trr = df_trr_by_beat.number_of_trr_reports[-20:]

sns.set(font_scale = 1)

ax = sns.barplot(x = low20_beat, y = low20_trr, order = low20_beat, label = 'small')

ax.set_title('20 Beats with LOWEST Number of TRR Reports')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

plt.show()
#create counter of districts from top 20 beats and sort

top20_district = {}

for beat in top20_beat:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in top20_district.keys():

        top20_district[district] += 1

    else:

        top20_district[district] = 1

#print(top20_district)

top20_district_sorted_keys = sorted(top20_district, key=top20_district.get, reverse=True)

top20_district_sort ={}

for district in top20_district_sorted_keys:

    top20_district_sort[district] = top20_district[district]
#graph above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(top20_district_sort.keys()), y = list(top20_district_sort.values()), order = top20_district_sort, label = 'small')

ax.set_title('20 Districts with HIGHEST Number of TRR Reports')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

plt.show()
#create counter of districts from low 20 beats and sort

low20_district = {}

for beat in low20_beat:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in low20_district.keys():

        low20_district[district] += 1

    else:

        low20_district[district] = 1

#print(low20_district)

low20_district_sorted_keys = sorted(low20_district, key=low20_district.get, reverse=True)

low20_district_sort ={}

for district in low20_district_sorted_keys:

    low20_district_sort[district] = low20_district[district]
#graph above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(low20_district_sort.keys()), y = list(low20_district_sort.values()), order = low20_district_sort, label = 'small')

ax.set_title('20 Districts with LOWEST Number of TRR Reports')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

plt.show()
df_trr_id.head()
df_trr_id.columns
len(df_trr_id.event_id.unique())
#create a coutner for each resistance level

HRL = df_trr_id.highest_resistance_level.value_counts()

HRL.sort_index(inplace = True)

print(HRL)
#map for each resistance level to its label

rl_dict = {}

rl_dict[0.0] = 'Passive Resister'

rl_dict[1.0] = 'Active Resister'

rl_dict[2.0] = 'Assailant Assault'

rl_dict[3.0] = 'Assailant Assault/Battery'

rl_dict[4.0] = 'Assailant Battery'

rl_dict[5.0] = 'Assailant Deadly Force'
#re-label the counter we create two blocks ago

HRL.rename(index = {0.0: 'Passive Resister', 1.0: 'Active Resister', 2.0 : 'Assailant Assault', 3.0: 'Assailant Assault/Battery', 4.0: 'Assailant Battery', 5.0: 'Assailant Deadly Force'}, inplace = True)
#graph the counter of resistance levels

sns.set(font_scale = 1)

ax = sns.barplot(x = HRL.index, y = HRL, order =HRL.index, label = 'small')

ax.set_title('Distribution of Resistance Levels Across TRR Incidents')

ax.set(xlabel='Resistance Level', ylabel='Number of Incidents')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.tick_params(labelsize=8.5)

plt.show()
#isolate the trr's where the officer logged his action

df_trr_id_cat_iso = df_trr_id.loc[df_trr_id.list_of_cats.notnull()]
#create a counter for the frequencies of each action category

cat_counter = {}

for i in range (2,7):

    cat_counter[i] = 0

for ind in df_trr_id_cat_iso.index:

    cat_list_str = df_trr_id_cat_iso['list_of_cats'][ind]

    cat_list = cat_list_str.split(",")

    cat_list = cat_list[:-1].copy()

    #print(cat_list)

    #cat_list.pop()

    for cat in cat_list:

        intCat = int(float(cat))

        cat_counter[intCat] += 1
#label each action category on the counter we just made

cat_labels = {}

cat_labels[2] = 'Other Force'

cat_labels[3] = 'Physical Force - Holding, Taser Display'

cat_labels[4] = 'Physical Force - Stunning, Chemical'

cat_labels[5] = 'Impact Weapon, Taser, Physical Force - Direct Mechanical'

cat_labels[6] = 'Firearm'

cat_counter_labeled = {}

for i in range(2,7):

    amt = cat_counter[i]

    cat_counter_labeled[cat_labels[i]] = amt
#graph the counter!

plt.bar(x = cat_counter_labeled.keys(), height =cat_counter_labeled.values())

plt.title('Frequency of Member Action across All Incidents')

plt.xlabel('Action Categories')

plt.xticks(rotation=90)

plt.ylabel("Number of Member Action Instances")

plt.show()
#counter for each subject action (sorry it's so sloppy!)

subject_acts_dict = {}

subject_acts_dict['subject_pulled_away'] = 0

subject_acts_dict['subject_disobey_verbal'] = 0

subject_acts_dict['subject_stiffened'] = 0

subject_acts_dict['subject_fled'] = 0

subject_acts_dict['subject_attack_no_weapon'] = 0

subject_acts_dict['subject_battery_threat'] = 0

subject_acts_dict['subject_had_weapon'] = 0

subject_acts_dict['subject_attack_with_weapon'] = 0

subject_acts_dict['subject_deadly_force'] = 0

subject_acts_dict['subject_other'] = 0

subject_acts_dict['subject_armed'] = 0

for ind in df_trr_id.index:

    if df_trr_id['subject_pulled_away'][ind] == 1:

        subject_acts_dict['subject_pulled_away'] += 1

    if df_trr_id['subject_disobey_verbal'][ind] == 1:

        subject_acts_dict['subject_disobey_verbal'] += 1

    if df_trr_id['subject_stiffened'][ind] == 1:

        subject_acts_dict['subject_stiffened'] += 1

    if df_trr_id['subject_fled'][ind] == 1:

        subject_acts_dict['subject_fled'] += 1

    if df_trr_id['subject_battery_threat'][ind] == 1:

        subject_acts_dict['subject_battery_threat'] += 1

    if df_trr_id['subject_had_weapon'][ind] == 1:

        subject_acts_dict['subject_had_weapon'] += 1

    if df_trr_id['subject_attack_with_weapon'][ind] == 1:

        subject_acts_dict['subject_attack_with_weapon'] += 1

    if df_trr_id['subject_deadly_force'][ind] == 1:

        subject_acts_dict['subject_deadly_force'] += 1

    if df_trr_id['subject_other'][ind] == 1:

        subject_acts_dict['subject_other'] += 1

    if df_trr_id['subject_armed'][ind] == 1:

        subject_acts_dict['subject_armed'] += 1

out = dict(itertools.islice(subject_acts_dict.items(), N)) 

print(out)
#graph this counter

plt.bar(x = subject_acts_dict.keys(), height =subject_acts_dict.values())

plt.title('Frequency of Subject Action across All Incidents')

plt.xlabel('Type of Subject Action')

plt.xticks(rotation= 90)

plt.ylabel("Number of Subject Action Instances")

plt.show()
df_trr_id.head()
df_subject_id.head()
#put the TRR incident ID's for each event ID in an accesible format

trr_by_event_dict = {}

for ind in df_trr_id.index:

    eventID = df_trr_id['event_id'][ind]

    trrID = str(df_trr_id['trr_id'][ind])

    if(eventID in trr_by_event_dict.keys()):

        trr_by_event_dict[eventID] += (trrID+',')

    else:

        trr_by_event_dict[eventID] = (trrID+',')

for key in trr_by_event_dict.keys():

    lizt = trr_by_event_dict[key].split(',')

    trr_by_event_dict[key] = lizt[:-1].copy()

out = dict(itertools.islice(trr_by_event_dict.items(), N)) 

print(out)
#put the races of each subject in a more accessible format

race_by_id = {}

for ind in df_subject_id.index:

    subjectID = df_subject_id['subject_ID'][ind]

    race = df_subject_id['race'][ind]

    race_by_id[subjectID] = race

out = dict(itertools.islice(race_by_id.items(), N)) 

print(out)
#get subject ID of each subject in each TRR incident

subject_by_trr = {}

for ind in df_subject_id.index:

    subjectID = df_subject_id['subject_ID'][ind]

    trr_list = df_subject_id['list_of_trr_id'][ind].split(',')[:-1]

    for trr in trr_list:

        subject_by_trr[trr] = subjectID

out = dict(itertools.islice(subject_by_trr.items(), N)) 

print(out)
#get subject ID of each suject in each event ID

subjectID_in_event = {}

for eventID in trr_by_event_dict.keys():

    subjectID_in_event[eventID] = ''

    for trr in trr_by_event_dict[eventID]:

        subID = str(subject_by_trr[trr])

        if subID not in subjectID_in_event[eventID]:

            subjectID_in_event[eventID] += (subID+',')

for key in subjectID_in_event.keys():

    lizt = subjectID_in_event[key].split(',')

    subjectID_in_event[key] = lizt[:-1].copy()

out = dict(itertools.islice(subjectID_in_event.items(), N)) 

print(out)
#use subject id's to get race of each subject in each event

race_in_event = {}

for eventID in subjectID_in_event.keys():

    race_in_event[eventID] = ''

    for subjectID in subjectID_in_event[eventID]:

        race_in_event[eventID] += (race_by_id[int(float(subjectID))]+',')

out = dict(itertools.islice(race_in_event.items(), N)) 

print(out)
#begin making our dataframe that's indexed by event ID by adding the race of each subject in each event

df_event_id = pd.DataFrame.from_dict(race_in_event, orient = 'index')

df_event_id.columns = ['race_of_subjects']

df_event_id.index.name = 'event_ID'

df_event_id.head()
#adding number of subjcets to that dataframe

df_event_id['num_of_subjects'] = df_event_id.race_of_subjects.apply(lambda x: len(x.split(','))-1)

df_event_id.head()
#map each event id to a list of trr incidents that occured during that event ID

trr_by_event = {}

for eventID in trr_by_event_dict.keys():

    trr_by_event[eventID] = ''

    for trrID in trr_by_event_dict[eventID]:

        trr_by_event[eventID] += str(trrID) + ','

out = dict(itertools.islice(trr_by_event.items(), N)) 

print(out)
#adding the list of trr events to that dataframe

temp = pd.DataFrame.from_dict(trr_by_event, orient = 'index')

temp.columns = ['list_of_trr']

temp.index.name = 'event_ID'

temp.head()
#table concatenation

df_event_id = pd.concat([df_event_id, temp], axis = 1)

df_event_id.head()
#isolate the trr's where an injury was officialy reported or allegedly occured

df_trr_id_injured = df_trr_id.loc[(df_trr_id.injured == 1) | (df_trr_id.alleged_injury == 1)]
df_trr_id_injured.info()
#see how many TRR's had an alleged injury, but the officer didn't report such an injury

df_trr_id_injured.loc[(df_trr_id_injured.injured == 0) & (df_trr_id_injured.alleged_injury == 1)].info()
#sorting the resistance levels in the incidents where an injury occured

injured_HRL = df_trr_id_injured.highest_resistance_level.value_counts()

print(injured_HRL)
#re-labeling

injured_HRL.rename(index = {0.0: 'Passive Resister', 1.0: 'Active Resister', 2.0 : 'Assailant Assault', 3.0: 'Assailant Assault/Battery', 4.0: 'Assailant Battery', 5.0: 'Assailant Deadly Force'}, inplace = True)
#graphing the sorting of resitance levels where an injury occured we just did

ax = sns.barplot(x = injured_HRL.index, y = injured_HRL, order = injured_HRL.index.sort_values(), label = 'small')

ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

ax.set_xlabel('Resitance Level')

ax.set_ylabel('Number of TRR Incidents Reporting Injury (Alleged or Officer-Reported)')

ax.tick_params(labelsize=8.5)

ax.set_title('Number of Reported Injuries by Resistance Level')

plt.show()
#sorting our index for neatness

HRL.sort_index(inplace = True)
#printing the level of resistance and percentage of incidnets resulting in injury at this resistance level

for ind in HRL.index:

    print('Level of Resistance: ' + ind)

    print('Number of Injuries: '+ str(injured_HRL[ind]))

    print('Number of Incidents: '+ str(HRL[ind]))

    print('Percentage of Injuries per Incident: '+str(injured_HRL[ind]/float(HRL[ind])*100)+ '%')

    print('-'*50)
df_trr_id_injured.head()
df_subject_id.head()
#putting the race of each subject in each TRR incident in

race_in_trr = {}

for ind in df_subject_id.index:

    trr_str = df_subject_id['list_of_trr_id'][ind]

    race = df_subject_id['race'][ind]

    trr_lizt = trr_str.split(',')[:-1]

    for trr in trr_lizt:

        trr_int = int(trr)

        race_in_trr[trr_int] = race

out = dict(itertools.islice(race_in_trr.items(), N)) 

print(out)
df_trr_id_injured['race'] = ''

df_trr_id['race'] = ''
df_trr_id.head()
#adding the race feature to the main dataframe and the injured dataframe

for ind in df_trr_id.index:

    trr = df_trr_id['trr_id'][ind]

    df_trr_id['race'][ind] = race_in_trr[trr]

for ind in df_trr_id_injured.index:

    trr = df_trr_id_injured['trr_id'][ind]

    df_trr_id_injured['race'][ind] = race_in_trr[trr]
df_trr_id_injured.head()
df_trr_id_injured.head()
passive_resister = df_trr_id.loc[df_trr_id.highest_resistance_level == 0]
passive_race = {}

passive_race['BLACK'] = 0

passive_race['WHITE'] = 0

passive_race['HISPANIC'] = 0

passive_race['OTHER'] = 0



for ind in passive_resister.index:

    race = passive_resister['race'][ind]

    if(race == 'OTHER' or race == 'ASIAN/PACIFIC ISLANDER' or race == 'NATIVE AMERICAN/PACIFIC ISLANDER'):

        passive_race['OTHER'] +=1

    else:

        passive_race[race] +=1
print('PERCENTAGE OF \'PASSIVE RESISTER\' BY RACE: \n')

for race in passive_race.keys():

    percentage = str(passive_race[race]/float(len(passive_resister))*100)

    print('Number of Incidents for '+race+' citizens: '+str(passive_race[race]))

    print('Total Number of Incidents where Passive Resister was Highest Resistance: '+str(len(passive_resister)))

    print('Percentage of '+race+' Subjects who were Passive Resisters: '+percentage+'%')

    print('-'*50)

data = list(passive_race.values())

labels = list(passive_race.keys())

plt.pie(data,labels= labels,autopct='%1.2f%%')

plt.title('Breakdown of Passive Resisters by Race')

plt.axis('equal')

plt.show()
active_resister = df_trr_id.loc[df_trr_id.highest_resistance_level == 1]
active_race = {}

active_race['BLACK'] = 0

active_race['WHITE'] = 0

active_race['HISPANIC'] = 0

active_race['OTHER'] = 0



for ind in active_resister.index:

    race = active_resister['race'][ind]

    if(race == 'OTHER' or race == 'ASIAN/PACIFIC ISLANDER' or race == 'NATIVE AMERICAN/ALASKAN NATIVE'):

        active_race['OTHER'] +=1

    else:

        active_race[race] +=1
print('PERCENTAGE OF \'ACTIVE RESISTER\' BY RACE: \n')

for race in active_race.keys():

    percentage = str(active_race[race]/float(len(active_resister))*100)

    print('Number of Incidents for '+race+' citizens: '+str(active_race[race]))

    print('Total Number of Incidents where Active Resister was Highest Resistance: '+str(len(active_resister)))

    print('Percentage of '+race+' Subjects who were Active Resisters: '+percentage+'%')

    print('-'*50)
data = list(active_race.values())

labels = list(active_race.keys())

plt.pie(data,labels= labels,autopct='%1.2f%%')

plt.title('Breakdown of Active Resisters by Race')

plt.axis('equal')

plt.show()
assailant_assault = df_trr_id.loc[df_trr_id.highest_resistance_level == 2]
assailant_assault_race = {}

assailant_assault_race['BLACK'] = 0

assailant_assault_race['WHITE'] = 0

assailant_assault_race['HISPANIC'] = 0

assailant_assault_race['OTHER'] = 0



for ind in assailant_assault.index:

    race = assailant_assault['race'][ind]

    if(race == 'OTHER' or race == 'ASIAN/PACIFIC ISLANDER' or race == 'NATIVE AMERICAN/ALASKAN NATIVE'):

        assailant_assault_race['OTHER'] +=1

    else:

        assailant_assault_race[race] +=1
print('PERCENTAGE OF ASSAILANT ASSAULT BY RACE: \n')

for race in assailant_assault_race.keys():

    percentage = str(assailant_assault_race[race]/float(len(assailant_assault))*100)

    print('Number of Incidents for '+race+' citizens: '+str(assailant_assault_race[race]))

    print('Total Number of Incidents where Assailant Assault was Highest Resistance: '+str(len(assailant_assault)))

    print('Percentage of '+race+' Subjects who were Assailant Assaulters: '+percentage+'%')

    print('-'*50)
data = list(assailant_assault_race.values())

labels = list(assailant_assault_race.keys())

plt.pie(data,labels= labels,autopct='%1.2f%%')

plt.title('Breakdown of Assailant Assault by Race')

plt.axis('equal')

plt.show()
assailant_ab = df_trr_id.loc[df_trr_id.highest_resistance_level == 3]
assailant_ab_race = {}

assailant_ab_race['BLACK'] = 0

assailant_ab_race['WHITE'] = 0

assailant_ab_race['HISPANIC'] = 0

assailant_ab_race['OTHER'] = 0



for ind in assailant_ab.index:

    race = assailant_ab['race'][ind]

    if(race == 'OTHER' or race == 'ASIAN/PACIFIC ISLANDER' or race == 'NATIVE AMERICAN/ALASKAN NATIVE'):

        assailant_ab_race['OTHER'] +=1

    else:

        assailant_ab_race[race] +=1
print('PERCENTAGE OF ASSAILANT ASSAULT/BATTERY BY RACE: \n')

for race in assailant_ab_race.keys():

    percentage = str(assailant_ab_race[race]/float(len(assailant_ab))*100)

    print('Number of Incidents for '+race+' citizens: '+str(assailant_ab_race[race]))

    print('Total Number of Incidents where Assailant Assault/Battery was Highest Resistance: '+str(len(assailant_ab)))

    print('Percentage of '+race+' Subjects who were Assailant Assaulters/Battery: '+percentage+'%')

    print('-'*50)
data = list(assailant_ab_race.values())

labels = list(assailant_ab_race.keys())

plt.pie(data,labels= labels,autopct='%1.2f%%')

plt.title('Breakdown of Assailant Assault/Battery by Race')

plt.axis('equal')

plt.show()
assailant_battery = df_trr_id.loc[df_trr_id.highest_resistance_level == 4]
assailant_battery_race = {}

assailant_battery_race['BLACK'] = 0

assailant_battery_race['WHITE'] = 0

assailant_battery_race['HISPANIC'] = 0

assailant_battery_race['OTHER'] = 0



for ind in assailant_battery.index:

    race = assailant_battery['race'][ind]

    if(race == 'OTHER' or race == 'ASIAN/PACIFIC ISLANDER' or race == 'NATIVE AMERICAN/ALASKAN NATIVE'):

        assailant_battery_race['OTHER'] +=1

    else:

        assailant_battery_race[race] +=1
print('PERCENTAGE OF ASSAILANT BATTERY BY RACE: \n')

for race in assailant_battery_race.keys():

    percentage = str(assailant_battery_race[race]/float(len(assailant_battery))*100)

    print('Number of Incidents for '+race+' citizens: '+str(assailant_battery_race[race]))

    print('Total Number of Incidents where Assailant Battery was Highest Resistance: '+str(len(assailant_battery)))

    print('Percentage of '+race+' Subjects who were Assailants committing Battery: '+percentage+'%')

    print('-'*50)
data = list(assailant_battery_race.values())

labels = list(assailant_battery_race.keys())

plt.pie(data,labels= labels,autopct='%1.2f%%')

plt.title('Breakdown of Assailant Battery by Race')

plt.axis('equal')

plt.show()
deadly = df_trr_id.loc[df_trr_id.highest_resistance_level == 5]
deadly_race = {}

deadly_race['BLACK'] = 0

deadly_race['WHITE'] = 0

deadly_race['HISPANIC'] = 0

deadly_race['OTHER'] = 0



for ind in deadly.index:

    race = deadly['race'][ind]

    if(race == 'OTHER' or race == 'ASIAN/PACIFIC ISLANDER' or race == 'NATIVE AMERICAN/ALASKAN NATIVE'):

        deadly_race['OTHER'] +=1

    else:

        deadly_race[race] +=1
print('PERCENTAGE OF ASSAILANT DEADLY FORCE BY RACE: \n')

for race in deadly_race.keys():

    percentage = str(deadly_race[race]/float(len(deadly))*100)

    print('Number of Incidents for '+race+' citizens: '+str(deadly_race[race]))

    print('Total Number of Incidents where Assailant Deadly Force was Highest Resistance: '+str(len(deadly)))

    print('Percentage of '+race+' Subjects who were Assailants presenting Deadly Force: '+percentage+'%')

    print('-'*50)
data = list(deadly_race.values())

labels = list(deadly_race.keys())

plt.pie(data,labels= labels,autopct='%1.2f%%')

plt.title('Breakdown of Assailant Deadly Force by Race')

plt.axis('equal')

plt.show()
#BLACK

black_all = df_trr_id.loc[df_trr_id.race == 'BLACK']

black_denom = len(black_all)

#print(black_denom)

black_num = black_all.highest_resistance_level.value_counts()

black_num.sort_index(inplace = True)

#print(black_num)

print('BREAKDOWN OF HIGHEST RESISTANCE LEVEL AMONGST BLACK CITIZENS')

print()

for ind in black_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Incidents at this Resistance: '+ str(black_num[ind]))

    print('Number of Total Incidents Against BLACK Citizens: '+ str(black_denom))

    print('Percentage of Total Inicidents at this Resistnace: '+str(black_num[ind]/float(black_denom)*100)+ '%')

    print('-'*50)
#HISPANIC

hispanic_all = df_trr_id.loc[df_trr_id.race == 'HISPANIC']

hispanic_denom = len(hispanic_all)

#print(hispanic_denom)

hispanic_num = hispanic_all.highest_resistance_level.value_counts()

hispanic_num.sort_index(inplace = True)

#print(hispanic_num)

print('BREAKDOWN OF HIGHEST RESISTANCE LEVEL AMONGST HISPANIC CITIZENS')

print()

for ind in hispanic_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Incidents at this Resistance: '+ str(hispanic_num[ind]))

    print('Number of Total Incidents Against HISPANIC Citizens: '+ str(hispanic_denom))

    print('Percentage of Total Incidents at this Resistance: '+str(hispanic_num[ind]/float(hispanic_denom)*100)+ '%')

    print('-'*50)
#WHITE

white_all = df_trr_id.loc[df_trr_id.race == 'WHITE']

white_denom = len(white_all)

#print(white_denom)

white_num = white_all.highest_resistance_level.value_counts()

white_num.sort_index(inplace = True)

#print(white_num)

print('BREAKDOWN OF HIGHEST RESISTANCE LEVEL AMONGST WHITE CITIZENS')

print()

for ind in white_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Incidents at this Resistance: '+ str(white_num[ind]))

    print('Number of Total Incidents Against WHITE Citizens: '+ str(white_denom))

    print('Percentage of Incidents at this Resistance: '+str(white_num[ind]/float(white_denom)*100)+ '%')

    print('-'*50)
#OTHER

other_all = df_trr_id.loc[df_trr_id.race == 'OTHER']

other_denom = len(other_all)

#print(other_denom)

other_num = other_all.highest_resistance_level.value_counts()

other_num.sort_index(inplace = True)

#print(other_num)

print('BREAKDOWN OF HIGHEST RESISTANCE LEVEL AMONGST \'OTHER\' CITIZENS')

print()

for ind in other_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Incidents at this Resistance: '+ str(other_num[ind]))

    print('Number of Total Incidents Against \'OTHER\' Citizens: '+ str(other_denom))

    print('Percentage of Incidents at this Resistance: '+str(other_num[ind]/float(other_denom)*100)+ '%')

    print('-'*50)
#ASIAN/PACIFIC ISLANDER

asian_all = df_trr_id.loc[df_trr_id.race == 'ASIAN/PACIFIC ISLANDER']

asian_denom = len(asian_all)

#print(asian_denom)

asian_num = asian_all.highest_resistance_level.value_counts()

asian_num.sort_index(inplace = True)

#print(asian_num)

print('BREAKDOWN OF HIGHEST RESISTANCE LEVEL AMONGST ASIAN/PACIFIC ISLANDER CITIZENS')

print()

for ind in asian_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Incidents at this Resistance: '+ str(asian_num[ind]))

    print('Number of Total Incidents Against ASIAN Citizens: '+ str(asian_denom))

    print('Percentage of Incidents at this Resistance: '+str(asian_num[ind]/float(asian_denom)*100)+ '%')

    print('-'*50)
#NATIVE AMERICAN/ALASKAN NATIVE

native_all = df_trr_id.loc[df_trr_id.race == 'NATIVE AMERICAN/ALASKAN NATIVE']

native_denom = len(native_all)

#print(native_denom)

native_num = native_all.highest_resistance_level.value_counts()

for i in range (0,6):

    key = float(i)

    if key not in native_num.keys():

        native_num[key] = 0

native_num.sort_index(inplace = True)

#print(native_num)

print('BREAKDOWN OF HIGHEST RESISTANCE LEVEL AMONGST NATIVE AMERICAN/ALASKAN NATIVE CITIZENS')

print()

for ind in native_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Incidents at this Resistance: '+ str(native_num[ind]))

    print('Number of Total Incidents Against NATIVE AMERICAN Citizens: '+ str(native_denom))

    print('Percentage of Incidents at this Resistance: '+str(native_num[ind]/float(native_denom)*100)+ '%')

    print('-'*50)
#NOTE: 'OTHER' category is now Other, Asian, and Native American combined

#putpercetnage data into dicts for graphing acess

black_dict ={}

hispanic_dict = {}

white_dict = {}

other_dict = {}

for i in range (0,6):

    key = float(i)

    black_dict[key] = black_num[key]/black_denom*100

    hispanic_dict[key] = hispanic_num[key]/hispanic_denom*100

    white_dict[key] = white_num[key]/white_denom*100

    other_dict[key] = (other_num[key]+asian_num[key]+native_num[key])/(other_denom+asian_denom+native_denom)*100

print(black_dict)

print(hispanic_dict)

print(white_dict)

print(other_dict)
#graph above data

n = 6

index = np.arange(n)

width = .15

fig, ax = plt.subplots()

y1 = list(black_dict.values())

y2 = list(hispanic_dict.values())

y3 = list(white_dict.values())

y4 = list(other_dict.values())

rects1 = ax.bar(index, y1, width, color='r')

rects2 = ax.bar(index + width, y2, width, color='y')

rects3 = ax.bar(index + 2*width, y3, width, color='b')

rects4 = ax.bar(index + 3*width, y4, width, color='m')

ax.set_ylabel('Percetange')

ax.set_title('Percentage of TRR Incidents At Each Resistance Level Per Race')

ax.set_xticks(index + width / 2)

#ax.xticks(rotation=90)

ax.set_xticklabels(('Passive Resister', 'Active Resister', 'Assailant Assault', 'Assailant Assault/Battery', 'Assailant Battery', 'Assailant Deadly Force'), rotation = 45)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('BLACK', 'HISPANIC', 'WHITE', 'OTHER'), bbox_to_anchor=(1.28, 1.02))

plt.show()
#breakdown of trr with injuries by race

race_inj_num = df_trr_id_injured.race.value_counts()
#breakdown of incidents with an injury by race

race_inj_denom = df_trr_id.race.value_counts()

print(race_inj_denom)
#printing out the brekadown percetnage of trr's resulting in injury by race

for ind in race_inj_num.index:

    print('Race: ' + ind)

    print('Number of Injuries involving '+ind+' citizens: '+ str(race_inj_num[ind]))

    print('Number of Incidents involving '+ind+' citizens: '+ str(race_inj_denom[ind]))

    print('Percentage of Injuries per Incident: '+str(race_inj_num[ind]/float(race_inj_denom[ind])*100)+ '%')

    print('-'*50)
#BLACK

black_all = df_trr_id.loc[df_trr_id.race == 'BLACK']

black_injured = df_trr_id_injured.loc[df_trr_id_injured.race == 'BLACK']

black_denom = black_all.highest_resistance_level.value_counts()

#print(black_denom)

black_num = black_injured.highest_resistance_level.value_counts()

black_num.sort_index(inplace = True)

#print(black_num)

print('PERCENTAGE OF INJURIES AMONGST BLACK CITIZENS PER RESISTANCE LEVEL')

print()

for ind in black_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Injuries: '+ str(black_num[ind]))

    print('Number of Incidents: '+ str(black_denom[ind]))

    print('Percentage of Injuries per Incident: '+str(black_num[ind]/float(black_denom[ind])*100)+ '%')

    print('-'*50)
#HISPANIC

hispanic_all = df_trr_id.loc[df_trr_id.race == 'HISPANIC']

hispanic_injured = df_trr_id_injured.loc[df_trr_id_injured.race == 'HISPANIC']

hispanic_denom = hispanic_all.highest_resistance_level.value_counts()

#print(hispanic_denom)

hispanic_num = hispanic_injured.highest_resistance_level.value_counts()

hispanic_num.sort_index(inplace = True)

#print(hispanic_num)

print('PERCENTAGE OF INJURIES AMONGST HISPANIC CITIZENS PER RESISTANCE LEVEL')

print()

for ind in hispanic_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Injuries: '+ str(hispanic_num[ind]))

    print('Number of Incidents: '+ str(hispanic_denom[ind]))

    print('Percentage of Injuries per Incident: '+str(hispanic_num[ind]/float(hispanic_denom[ind])*100)+ '%')

    print('-'*50)
#WHITE

white_all = df_trr_id.loc[df_trr_id.race == 'WHITE']

white_injured = df_trr_id_injured.loc[df_trr_id_injured.race == 'WHITE']

white_denom = white_all.highest_resistance_level.value_counts()

#print(white_denom)

white_num = white_injured.highest_resistance_level.value_counts()

white_num.sort_index(inplace = True)

#print(white_num)

print('PERCENTAGE OF INJURIES AMONGST WHITE CITIZENS PER RESISTANCE LEVEL')

print()

for ind in white_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Injuries: '+ str(white_num[ind]))

    print('Number of Incidents: '+ str(white_denom[ind]))

    print('Percentage of Injuries per Incident: '+str(white_num[ind]/float(white_denom[ind])*100)+ '%')

    print('-'*50)
#OTHER

other_all = df_trr_id.loc[df_trr_id.race == 'OTHER']

other_injured = df_trr_id_injured.loc[df_trr_id_injured.race == 'OTHER']

other_denom = other_all.highest_resistance_level.value_counts()

#print(other_denom)

other_num = other_injured.highest_resistance_level.value_counts()

other_num.sort_index(inplace = True)

#print(other_num)

print('PERCENTAGE OF INJURIES AMONGST \'OTHER\' CITIZENS PER RESISTANCE LEVEL')

print()

for ind in other_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Injuries: '+ str(other_num[ind]))

    print('Number of Incidents: '+ str(other_denom[ind]))

    print('Percentage of Injuries per Incident: '+str(other_num[ind]/float(other_denom[ind])*100)+ '%')

    print('-'*50)
#ASIAN/PACIFIC ISLANDER

asian_all = df_trr_id.loc[df_trr_id.race == 'ASIAN/PACIFIC ISLANDER']

asian_injured = df_trr_id_injured.loc[df_trr_id_injured.race == 'ASIAN/PACIFIC ISLANDER']

asian_denom = asian_all.highest_resistance_level.value_counts()

#print(asian_denom)

asian_num = asian_injured.highest_resistance_level.value_counts()

asian_num.sort_index(inplace = True)

#print(asian_num)

print('PERCENTAGE OF INJURIES AMONGST ASIAN/PACIFIC ISLANDER CITIZENS PER RESISTANCE LEVEL')

print()

for ind in asian_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Injuries: '+ str(asian_num[ind]))

    print('Number of Incidents: '+ str(asian_denom[ind]))

    print('Percentage of Injuries per Incident: '+str(asian_num[ind]/float(asian_denom[ind])*100)+ '%')

    print('-'*50)
#NATIVE AMERICAN/ALASKAN NATIVE

native_all = df_trr_id.loc[df_trr_id.race == 'NATIVE AMERICAN/ALASKAN NATIVE']

native_injured = df_trr_id_injured.loc[df_trr_id_injured.race == 'NATIVE AMERICAN/ALASKAN NATIVE']

native_denom = native_all.highest_resistance_level.value_counts()

#print(native_denom)

native_num = native_injured.highest_resistance_level.value_counts()

native_num.sort_index(inplace = True)

#print(native_num)

for i in range (0,6):

    key = float(i)

    if key not in native_num.keys():

        native_num[key] = 0

        native_denom[key] = 0

print('PERCENTAGE OF INJURIES AMONGST NATIVE AMERICAN/ALASKAN NATIVE CITIZENS PER RESISTANCE LEVEL')

print()

for ind in black_num.index:

    print('Level of Resistance: ' + str(rl_dict[ind]))

    print('Number of Injuries: '+ str(native_num[ind]))

    print('Number of Incidents: '+ str(native_denom[ind]))

    if native_denom[ind] == 0:

        print('Percentage of Injuries per Incident: 0%')

    else:

        print('Percentage of Injuries per Incident: '+str(native_num[ind]/float(native_denom[ind])*100)+ '%')

    print('-'*50)
#NOTE: 'OTHER' category is now Other, Asian, and Native American combined

black_dict ={}

hispanic_dict = {}

white_dict = {}

other_dict = {}

for i in range (0,6):

    key = float(i)

    black_dict[key] = black_num[key]/black_denom[key]*100

    hispanic_dict[key] = hispanic_num[key]/hispanic_denom[key]*100

    white_dict[key] = white_num[key]/white_denom[key]*100

    other_dict[key] = (other_num[key]+asian_num[key]+native_num[key])/(other_denom[key]+asian_denom[key]+native_denom[key])*100

print(black_dict)

print(hispanic_dict)

print(white_dict)

print(other_dict)
n = 6

index = np.arange(n)

width = .15

fig, ax = plt.subplots()

y1 = list(black_dict.values())

y2 = list(hispanic_dict.values())

y3 = list(white_dict.values())

y4 = list(other_dict.values())

rects1 = ax.bar(index, y1, width, color='r')

rects2 = ax.bar(index + width, y2, width, color='y')

rects3 = ax.bar(index + 2*width, y3, width, color='b')

rects4 = ax.bar(index + 3*width, y4, width, color='m')

ax.set_ylabel('Percetange')

ax.set_title('Percentage of TRR Incidents Resulting in Injury Based off Subject Resistance')

ax.set_xticks(index + width / 2)

#ax.xticks(rotation=90)

ax.set_xticklabels(('Passive Resister', 'Active Resister', 'Assailant Assault', 'Assailant Assault/Battery', 'Assailant Battery', 'Assailant Deadly Force'), rotation = 45)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('BLACK', 'HISPANIC', 'WHITE', 'OTHER'), bbox_to_anchor=(1.1, 1.05))

plt.show()
beat_injury = df_trr_id_injured.beat.value_counts()
#create counter for the 20 beats with highest number of injuries

top20_injured = {}

for beat in beat_injury.index[:20]:

    top20_injured[beat] = beat_injury[beat]

print(top20_injured)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(top20_injured.keys()), y = list(top20_injured.values()), order = top20_injured, label = 'small')

ax.set_title('20 Beats with HIGHEST Number of Injuries')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.tick_params(labelsize=8.5)

plt.show()
#create counter for the 20 beats with lowest number of injuries

low20_injured = {}

for beat in beat_injury.index[-20:]:

    low20_injured[beat] = beat_injury[beat]

print(low20_injured)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(low20_injured.keys()), y = list(low20_injured.values()), order = low20_injured, label = 'small')

ax.set_title('20 Beats with LOWEST Number of Injuries')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.tick_params(labelsize=8.5)

plt.show()
#sort above GREATEST counter by district

top20_injured_district = {}

for beat in top20_injured:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in top20_injured_district.keys():

        top20_injured_district[district] += 1

    else:

        top20_injured_district[district] = 1

#print(top20_injured_district)

top20_injured_district_sorted_keys = sorted(top20_injured_district, key=top20_injured_district.get, reverse=True)

top20_injured_district_sort ={}

for district in top20_injured_district_sorted_keys:

    top20_injured_district_sort[district] = top20_injured_district[district]

print(top20_injured_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(top20_injured_district_sort.keys()), y = list(top20_injured_district_sort.values()), order = top20_injured_district_sort, label = 'small')

ax.set_title('District with HIGHEST Number of Injuries')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
#sort above LEAST counter by district

low20_injured_district = {}

for beat in low20_injured:

    last2 = beat%100

    beat = beat - last2

    district = int(beat/100)

    if district in low20_injured_district.keys():

        low20_injured_district[district] += 1

    else:

        low20_injured_district[district] = 1

#print(low20_injured_district)

low20_injured_district_sorted_keys = sorted(low20_injured_district, key=low20_injured_district.get, reverse=True)

low20_injured_district_sort ={}

for district in low20_injured_district_sorted_keys:

    low20_injured_district_sort[district] = low20_injured_district[district]

print(low20_injured_district_sort)
#graph of above counter

sns.set(font_scale = 1)

ax = sns.barplot(x = list(low20_injured_district_sort.keys()), y = list(low20_injured_district_sort.values()), order = low20_injured_district_sort, label = 'small')

ax.set_title('Districts with LOWEST Number of Injuries')

ax.set_xticklabels(ax.get_xticklabels())

ax.tick_params(labelsize=8.5)

plt.show()
print('TOP 20 BEATS IN CRIME, ARREST, RATIO (arrest to crime), AND INJURIES\n')

print('CRIME: ')

print(top20_crime_district_sort)

print('-'*60)

print('ARREST: ')

print(top20_arrest_district_sort)

print('-'*60)

print('RATIO: ')

print(top20_ratio_district_sort)

print('-'*60)

print('INJURIES: ')

print(top20_injured_district_sort)

print('-'*60)
print('BOTTOM 20 BEATS IN CRIME, ARREST, RATIO (arrest to crime), AND INJURIES\n')

print('CRIME: ')

print(low20_crime_district_sort)

print('-'*60)

print('ARREST: ')

print(low20_arrest_district_sort)

print('-'*60)

print('RATIO: ')

print(low20_ratio_district_sort)

print('-'*60)

print('INJURIES: ')

print(low20_injured_district_sort)

print('-'*60)
outF = open("outputBeats_EDA_TOP.txt", "w")
df_beat_crime = df_beat.sort_values(by = 'number_of_reported_crimes', ascending = False)

df_beat_crime.head(20)
outF.write("T20 Crimes: \n")

for ind in df_beat_crime.index[:20]:

    beat = str(df_beat_crime['beat'][ind])

    outF.write(beat+" ")

outF.write('\n')
df_beat_arrests = df_beat.sort_values(by = 'arrests_by_beat', ascending = False)

df_beat_arrests.head(20)
outF.write("T20 Arrests: \n")

for ind in df_beat_arrests.index[:20]:

    beat = str(df_beat_arrests['beat'][ind])

    outF.write(beat+" ")

outF.write('\n')
df_beat_ratio = df_beat.sort_values(by = 'arrest_to_crime_ratio', ascending = False)

df_beat_ratio.head(20)
outF.write("T20 Ratio: \n")

for ind in df_beat_ratio.index[:20]:

    beat = str(df_beat_ratio['beat'][ind])

    outF.write(beat+" ")

outF.write('\n')
outF.write("T20 Ratio: \n")

for beat in top20_injured.keys():

    beat = str(beat)

    outF.write(beat+" ")

outF.write('\n')
from IPython.display import FileLink

FileLink(r'outputBeats_EDA_TOP.txt')