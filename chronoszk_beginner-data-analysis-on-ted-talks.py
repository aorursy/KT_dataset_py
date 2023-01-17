import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime

from collections import Counter

import ast

import re

from PIL import Image

from wordcloud import WordCloud

import requests

from io import BytesIO

import warnings

warnings.filterwarnings("ignore")
import os

from os import path
# get data directory (using getcwd() is needed to support running example in generated IPython notebook)

# d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
df = pd.read_csv('../input/ted-talks/ted_main.csv')

df.head()
df.shape
df.isnull().sum()
df.describe()
df.dtypes
df['film_date'] = df['film_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))

df['published_date'] = df['published_date'].apply(lambda x: datetime.date.fromtimestamp(int(x)).strftime('%Y-%m-%d %H:%M:%S'))
df.sample(5)
df['film_date'], df['published_date'] = pd.to_datetime(df['film_date']), pd.to_datetime(df['published_date'])
#Number of ted talks published or filmed by year

pub_year=df['published_date'].dt.year.value_counts().sort_index()

film_year=df['film_date'].dt.year.value_counts().sort_index()



fig, (ax1, ax2) = plt.subplots(2,1,figsize=(15,6))

film_year.plot(kind='bar', ax=ax1)

pub_year.plot(kind='bar', ax=ax2)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

ax1.set_xlabel('Filmed Year')

ax2.set_xlabel('Published Year')

for i, v in enumerate(film_year):

    ax1.text(i-0.25,v+2, str(v),color='black',fontweight='bold')

for i, v in enumerate(pub_year):

    ax2.text(i-0.15,v+2, str(v),color='black',fontweight='bold')

ax1.title.set_text('Number of Ted Talks Filmed By Year')

ax2.title.set_text('Number of Ted Talks Published By Year')

plt.subplots_adjust(bottom=0, top=2)

plt.show()
# Which video received greater number of first level comments

df_comm = df[['main_speaker','title','published_date','comments']].sort_values(by=['comments']).reset_index(drop=True)

fig,ax=plt.subplots(figsize=(15,6))

plt.barh(df_comm['title'].tail(), df_comm['comments'].tail())

for i, v in enumerate(df_comm['comments'].tail()):

    ax.text(v/v,i, str(v),color='white',fontweight='bold')

plt.title('Ted Talks with Most First Level Comments')

plt.xlabel('Number of Comments')

plt.ylabel('Title')

plt.show()



print(df_comm.tail())
#Views by quantile 

com_quantile = pd.qcut(df['views'], q=4).value_counts().sort_index()

plt.figure(figsize=(10,6))

com_quantile.plot(kind='bar')

plt.xticks(rotation=0)

plt.xlabel('Number of views')

plt.title('Quantile on Views')

plt.show()
# Top 10 ted talk events with most published videos

top10_event = df['event'].value_counts().sort_values(ascending=False).head(10)

fig, ax=plt.subplots(figsize=(15,6))

top10_event.plot(kind='bar')

plt.xlabel('Events')

for i, v in enumerate(top10_event):

    ax.text(i-0.05,v+1, str(v),color='black',fontweight='bold')

plt.xticks(rotation=0)

plt.title('Top 10 Ted Talks Events with Most Published Videos')

plt.show()
events = df['event'].value_counts().sort_values(ascending=False)

event_2012 = [(i,v) for i,v in events.iteritems() if('2012' in i)]



event_2012_tag = [tag[0] for tag in event_2012]

event_2012_val = [val[1] for val in event_2012]



fig, ax = plt.subplots(figsize=(15,6))

plt.barh(event_2012_tag,event_2012_val)

plt.xlabel('Events')

for i, v in enumerate(event_2012_val):

    ax.text(v,i, str(v),color='black',fontweight='bold')

plt.title('Ted Talks Events on 2012')

plt.show()



print('Total number of videos published on Ted Talks events with 2012 labelling: ',sum(val for val in event_2012_val))
# What is the top 10 common tags in ted talks

flat_list=[]

for index, row in df.iterrows():

    tag = ast.literal_eval(row['tags'])

    for item in tag:

        flat_list.append(item)



tag_count = Counter(flat_list)

print('Total types of tags:',len(tag_count))
tag_cat = [tag[0] for tag in tag_count.most_common(10)]

tag_val = [tag[1] for tag in tag_count.most_common(10)]



fig, ax = plt.subplots()

plt.pie(tag_val, labels=tag_val, autopct='%1.1f%%', shadow=True, startangle=90)

ax.axis('equal') 

plt.title('Top 10 Most Common Tags in Ted Talks')

plt.legend(tag_cat,bbox_to_anchor=(1.5,1), fontsize=10, bbox_transform=plt.gcf().transFigure)

plt.subplots_adjust(bottom=0, top=1.3)

plt.show()
#Create word cloud of tags



# read the mask image

# taken from https://cdn.freebiesupply.com/images/large/2x/ted-logo-white.png

d = '../input/word-cloud-mask/'

ted_talk_mask = np.array(Image.open(d + "ted-logo-white.png"))



wc = WordCloud(mask=ted_talk_mask, background_color="white",width=800, height=400, contour_width=1).generate_from_frequencies(tag_count)



# show

plt.figure(figsize=(15,8))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title('Word Cloud of Tags')

plt.show()
#Which tag viewed most by audiences

tag_cat_view = []

for tag in tag_count:

    view_counts = 0

    for i in range(len(df)):

        #Match the token

        if(re.search("'"+tag+"'",df['tags'][i])):

            view_counts = view_counts + df['views'][i]

    #Append into list for visualization

    tag_cat_view.append((tag,view_counts))

    

# Sort it in descending order

tag_cat_view.sort(key=lambda x:x[1], reverse=True)
tag_cat_view_cat = [x[0] for x in tag_cat_view[:10]]

tag_cat_view_view = [x[1] for x in tag_cat_view[:10]] 



fig,ax=plt.subplots(figsize=(15,6))

plt.barh(tag_cat_view_cat, tag_cat_view_view)

for i, v in enumerate(tag_cat_view_view):

    ax.text(v/v,i, str(v),color='white',fontweight='bold')

plt.xticks(rotation=0)

plt.ylabel('Categories')

plt.xlabel('Number of Views')

plt.title('Top 10 Categories Viewed By Audiences')

plt.show()
# Who gave the most ted talks

df_most_active_speaker = df.groupby(['main_speaker','speaker_occupation']).agg(

    counts=('speaker_occupation', 'count'), average_views=('views','mean')).reset_index(

).sort_values(by='counts',ascending=False)

print(df_most_active_speaker[df_most_active_speaker['counts'] >= 5])
top_5_active_speaker = df_most_active_speaker[df_most_active_speaker['counts'] >= 5]

fig,ax=plt.subplots(figsize=(15,6))

plt.bar(top_5_active_speaker['main_speaker'], top_5_active_speaker['counts'])

for i, v in enumerate(top_5_active_speaker['counts']):

    ax.text(i-0.05,v/v-0.8, str(v),color='white',fontweight='bold')

plt.xlabel('Speakers')

plt.xticks(rotation=45)

plt.title('Most Active Speakers in Ted Talks')

plt.show()
print(df[df['main_speaker'] == 'Hans Rosling'][['main_speaker','title','tags']])
ignore_process_occupation = ['SURVEILLANCE AND CYBERSECURITY COUNSEL', 'NEUROSCIENCE AND CANCER RESEARCHER', 

                             'FOOD AND AGRICULTURE EXPERT', 'SCULPTOR OF LIGHT AND SPACE', 'PLANETARY AND ATMOSPHERIC SCIENTIST', 

                             'HEALTH AND TECHNOLOGY ACTIVIST', 'ENVIRONMENTAL AND LITERACY ACTIVIST', 

                             'PROFESSOR OF MOLECULAR AND CELL BIOLOGY','HIV/AIDS FIGHTER', '9/11 MOTHERS']

occupations = []

for index, row in df.iterrows():

    speaker_occupation = str(row['speaker_occupation']).upper().strip()

    if(re.search(r'FOUNDER', speaker_occupation)):

        

        speaker_occupation = re.sub(r'COFOUNDER|CO-FOUNDERS','CO-FOUNDER', speaker_occupation)

        if('CO-FOUNDER' in speaker_occupation):

            occupations.append('CO-FOUNDER')

        if('BLOGGER' in speaker_occupation): #BLOGGER; CO-FOUNDER, SIX APART

            occupations.append('BLOGGER')

        if('EXECUTIVE DIRECTOR' in speaker_occupation):

            occupations.append('EXECUTIVE DIRECTOR')

        if('CEO' in speaker_occupation):

            occupations.append('CEO')

        if('DESIGNER' in speaker_occupation):

            occupations.append('DESIGNER')

        if('FOUNDER' in speaker_occupation):

            occupations.append('FOUNDER')

    elif(re.search(r'COO', speaker_occupation)):

        occupations.append(speaker_occupation.split(',')[0])

    elif(re.search(r'DIRECTOR', speaker_occupation)):

        if(' AND ' in speaker_occupation):

            occupations.extend(speaker_occupation.split(' AND '))

        elif('DIRECTOR OF' in speaker_occupation):

            occupations.append(speaker_occupation.split(',')[0])

        elif(',' in speaker_occupation):

            speaker_occupation = re.sub(r'/',', ', speaker_occupation)

            speaker_occupation = re.sub(r';',',', speaker_occupation)

            speaker_occupation = speaker_occupation.replace(', IDEO','')

            speaker_occupation = speaker_occupation.replace(', THE INSTITUTE FOR GLOBAL HAPPINESS','')

            speaker_occupation = speaker_occupation.replace(', NSA','')

            occupations.extend(speaker_occupation.split(','))

    elif(re.search(r' AND |[+;.,/]', speaker_occupation)):

        if(speaker_occupation in ['EXECUTIVE CHAIR, FORD MOTOR CO.']): #SINGER-SONGWRITER

            occupations.append(speaker_occupation.split(',')[0])

        elif(speaker_occupation in ignore_process_occupation):

            occupations.append(speaker_occupation)

        else:

            speaker_occupation = re.sub(r' AND |[/]',', ', speaker_occupation)

            speaker_occupation = speaker_occupation.replace(' + ',', ')

            speaker_occupation = re.sub(r';',',', speaker_occupation)

            speaker_occupation = speaker_occupation.replace(' ...','')

            if('SINGER-SONGWRITER' == speaker_occupation):

                speaker_occupation = speaker_occupation.replace('-',', ')



            occupations.extend(speaker_occupation.split(', '))

        
occupations_counts = Counter(occupations)

print(occupations_counts)
#Counts of Top 10 speakers occupations in ted talks

occupations_counts_cat = [occ[0] for occ in occupations_counts.most_common(10)]

occupations_counts_val = [occ[1] for occ in occupations_counts.most_common(10)]



fig, ax = plt.subplots(figsize=(15,6))

plt.bar(occupations_counts_cat,occupations_counts_val)

for i, v in enumerate(occupations_counts_val):

    ax.text(i-0.1,v/v, str(v),color='white',fontweight='bold')

plt.title('Top 10 Speakers Occupations in Ted Talks')

plt.xlabel('Speakers Occupations')

plt.ylabel('Counts')

plt.show()
#Which speakers have higher average views per talk

top_ten_most_average_views_speaker = df_most_active_speaker.sort_values(by='average_views').tail(10)

fig, ax = plt.subplots(figsize=(15,6))

plt.barh(top_ten_most_average_views_speaker['main_speaker'],top_ten_most_average_views_speaker['average_views'])

for i, v in enumerate(top_ten_most_average_views_speaker['average_views']):

    ax.text(v/v, i, str(v), color='white', fontweight='bold')

plt.xlabel('Views')

plt.ylabel('Speakers')

plt.title('Top 10 Average Views by Speakers')

plt.show()



print(top_ten_most_average_views_speaker.tail(10))
#Top 10 up to date most views videos

df_most_views = df[['main_speaker', 'title', 'views','published_date']].sort_values(

    by='views').reset_index(drop=True)



fig, ax = plt.subplots(figsize=(8,6))

plt.barh(df_most_views['title'].tail(10),df_most_views['views'].tail(10))

for i, v in enumerate(df_most_views['views'].tail(10)):

    ax.text(v/v, i, str(v), color='white', fontweight='bold')

plt.title('Top 10 Most Views Ted Talks')

plt.xlabel('Views')

plt.ylabel('Titles')

plt.show()



df_most_views.tail(10)
df_ext = df.copy()
for index, row in df.iterrows():

    rates = ast.literal_eval(row['ratings'])

    for item in rates:

        if(index == 0): #to create new column

            df_ext[item['name']] = item['count']

        else:

            df_ext[item['name']][index] = item['count']



df_ext.head(5)        
df_ext.columns
df_ext.drop('ratings',axis=1, inplace=True)
#Which ted talks received higher number of ratings

rating_col = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded',

       'Confusing', 'Informative', 'Fascinating', 'Unconvincing', 'Persuasive',

       'Jaw-dropping', 'OK', 'Obnoxious', 'Inspiring']



df_ext['sum_ratings'] = df_ext[rating_col].sum(axis=1)



df_ext_sum_rate_sort = df_ext[['name','sum_ratings']].sort_values(by='sum_ratings').reset_index(drop=True)



fig, ax = plt.subplots(figsize=(10,6))

plt.barh(df_ext_sum_rate_sort['name'][-10:],df_ext_sum_rate_sort['sum_ratings'][-10:])

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

for i, v in enumerate(df_ext_sum_rate_sort['sum_ratings'][-10:]):

    ax.text(v/df_ext_sum_rate_sort['sum_ratings'][i], i, str(v), color='white', fontweight='bold')

plt.title('Ted Talks with Higher Number of Ratings')

plt.xlabel('Total Ratings')

plt.ylabel('Titles')

plt.show()
df_ext['main_rating'] = df_ext[rating_col].idxmax(axis=1)



most_rel_rat = df_ext['main_rating'].value_counts()

fig, ax = plt.subplots(figsize=(15,6))

plt.bar(most_rel_rat.index,most_rel_rat.values)

for i, v in enumerate(most_rel_rat):

    ax.text(i-0.2, v+10, str(v), color='black', fontweight='bold')

plt.title('Ted Talks Main Ratings')

plt.xticks(rotation=45)

plt.ylabel('Count')

plt.xlabel('Main Rating')

plt.show()
# Which ted talks considered as poor rating

for rate in ['Confusing','Obnoxious','Longwinded','Unconvincing']:

    print(df_ext[['main_rating','name',rate,'sum_ratings']][df_ext['main_rating'].isin([rate])].reset_index

          (drop=True))
fig, ax = plt.subplots(figsize=(20,20))

ax = sns.heatmap(df_ext.corr(), annot = True)