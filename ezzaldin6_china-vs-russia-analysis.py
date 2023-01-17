import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd 

import re

import fuzzywuzzy

from fuzzywuzzy import process

import os

plt.style.use('ggplot')

sns.set(style='darkgrid', context='notebook')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
female_chess=pd.read_csv('/kaggle/input/top-women-chess-players/top_women_chess_players_aug_2020.csv')

female_chess.head(5)
female_chess.info()
female_chess.describe(exclude='object')
female_chess.describe(include='object')
female_chess[female_chess['Name']=='Kuznetsova, Olga']
counter=0

for i in female_chess.duplicated(['Fide id']):

    if i==True:

        counter+=1

print('number of repeated rows: ', counter)
cleaned_female_chess=female_chess.copy()

for col in female_chess.columns:

    new=col.strip().lower().replace(' ', '_')

    cleaned_female_chess.rename(columns={col:new}, inplace=True)

cleaned_female_chess.drop('gender',axis=1, inplace=True)

cleaned_female_chess['title']=cleaned_female_chess['title'].fillna('WH')

cleaned_female_chess['rapid_rating']=cleaned_female_chess['rapid_rating'].fillna(0)

cleaned_female_chess['blitz_rating']=cleaned_female_chess['blitz_rating'].fillna(0)

for idx, val in zip(cleaned_female_chess.index, cleaned_female_chess['inactive_flag']):

    if val=='wi':

        cleaned_female_chess.loc[idx, 'inactive_flag']='InActive'

cleaned_female_chess['inactive_flag']=cleaned_female_chess['inactive_flag'].fillna('Active')
countries=cleaned_female_chess['federation'].unique()

china_matches=fuzzywuzzy.process.extract('CHN', countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

print(china_matches)

russia_matches=process.extract('RUS', countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

print(russia_matches)
cleaned_female_chess['first_name']=cleaned_female_chess['name'].str.extract('\w+\,\s(\w+)')

cleaned_female_chess['last_name']=cleaned_female_chess['name'].str.extract('(\w+)\,\s\w+')

cleaned_female_chess['age']=2020-cleaned_female_chess['year_of_birth']
cleaned_female_chess.head()
fig, ax=plt.subplots(figsize=(30,5))

cleaned_female_chess['federation'].value_counts().plot.bar(ax=ax)

ax.set_title('number of players in each federation')

ax.set_xlabel('Country')

ax.set_xticklabels(rotation=90, labels=cleaned_female_chess['federation'].value_counts().index)

ax.text(60, 1000, 'Russia has more than 1.6K Female Players',

        horizontalalignment='center',

        verticalalignment='center',

        fontsize=20)

plt.show()
#Title of Each Player

fig, ax=plt.subplots(figsize=(10,5))

cleaned_female_chess['title'].value_counts().plot.bar(ax=ax)

ax.set_title('Titles of each player')

ax.set_xlabel('Title')

ax.set_xticklabels(rotation=90, labels=cleaned_female_chess['title'].value_counts().index)

plt.show()
# Active and InActive Players

fig, ax=plt.subplots(figsize=(10,5))

ax.pie(cleaned_female_chess['inactive_flag'].value_counts(),

       autopct='%.2f%%')

ax.set_title('percentage of Active/InActive Players')

ax.legend(cleaned_female_chess['inactive_flag'].value_counts().index,

          loc='center right',

          title='player Activity',

          bbox_to_anchor=(1, 0, 0.5, 1))

plt.show()
# distribution of ages

sns.distplot(cleaned_female_chess['age'])

plt.title('Female Players Age Distribution')

plt.axvline(cleaned_female_chess['age'].mean(), color='black', label='mean')

plt.axvline(cleaned_female_chess['age'].median(), color='green', label='median')

plt.axvline(cleaned_female_chess['age'].std(), color='red', label='standard dev')

plt.legend()

plt.show()
grouped_data_1=cleaned_female_chess.groupby('federation')['title'].value_counts()

titles=cleaned_female_chess['title'].unique()

print('Percentages of Chinese Players in each title: \n ---------------')

for i in titles:

    try:

        calc1=round((grouped_data_1['CHN'][i]/len(cleaned_female_chess[cleaned_female_chess['title']==i]))*100,2)

        calc2=round((grouped_data_1['RUS'][i]/len(cleaned_female_chess[cleaned_female_chess['title']==i]))*100,2)

        print('{} in China: {}%\n {} in Russia: {}%\n'.format(i, calc1, i, calc2))

    except:

        print('No Players in both countries\n')
#standard_rating

print('Average Standard Rating : \n')

for i in titles:

    calc1=round(cleaned_female_chess[(cleaned_female_chess['title']==i)&(cleaned_female_chess['federation']=='CHN')]['standard_rating'].mean(),0)

    calc2=round(cleaned_female_chess[(cleaned_female_chess['title']==i)&(cleaned_female_chess['federation']=='RUS')]['standard_rating'].mean(),0)

    print('Avaerage rating of {} in china: {}\nAvaerage rating of {} in russia: {}\n'.format(i, calc1, i, calc2))
#Rapid_rating

print('Average rapid Rating : \n')

for i in titles:

    calc1=round(cleaned_female_chess[(cleaned_female_chess['title']==i)&(cleaned_female_chess['federation']=='CHN')]['rapid_rating'].mean(),0)

    calc2=round(cleaned_female_chess[(cleaned_female_chess['title']==i)&(cleaned_female_chess['federation']=='RUS')]['rapid_rating'].mean(),0)

    print('Avaerage rating of {} in china: {}\nAvaerage rating of {} in russia: {}\n'.format(i, calc1, i, calc2))
#blitz_rating

print('Average Blitz Rating : \n')

for i in titles:

    calc1=round(cleaned_female_chess[(cleaned_female_chess['title']==i)&(cleaned_female_chess['federation']=='CHN')]['blitz_rating'].mean(),0)

    calc2=round(cleaned_female_chess[(cleaned_female_chess['title']==i)&(cleaned_female_chess['federation']=='RUS')]['blitz_rating'].mean(),0)

    print('Avaerage rating of {} in china: {}\nAvaerage rating of {} in russia: {}\n'.format(i, calc1, i, calc2))
print('Number of Players have Blitz Rating: \n')

for i in titles:

    lst=[]

    for j in ['CHN', 'RUS']:

        counter=0

        brs=cleaned_female_chess[(cleaned_female_chess['title']==i)&(cleaned_female_chess['federation']==j)]['blitz_rating']

        for k in brs:

            if k>0:

                counter+=1

        lst.append(counter)

    print('Number of {} Players have BR in china: {}\nNumber of {} Players have BR in russia: {}\n'.format(i, lst[0], i, lst[1]))
sns.distplot(cleaned_female_chess[cleaned_female_chess['federation']=='CHN']['age'], color='red', label='china')

sns.distplot(cleaned_female_chess[cleaned_female_chess['federation']=='RUS']['age'], color='blue', label='russia')

plt.title('Ages Comparison')

plt.legend()

plt.show()
cleaned_female_chess[(cleaned_female_chess['federation']=='CHN')|(cleaned_female_chess['federation']=='RUS')].groupby('federation')['inactive_flag'].value_counts()