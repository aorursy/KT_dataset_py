# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Data import

groups = pd.read_csv('/kaggle/input/meetups-data-from-meetupcom/groups.csv')

cities = pd.read_csv('/kaggle/input/meetups-data-from-meetupcom/cities.csv')

events = pd.read_csv('/kaggle/input/meetups-data-from-meetupcom/events.csv')

categories = pd.read_csv('/kaggle/input/meetups-data-from-meetupcom/categories.csv')

grouptopics = pd.read_csv('/kaggle/input/meetups-data-from-meetupcom/groups_topics.csv', encoding='ISO-8859-1')
#Look at groups

groups.head(3)
#Check different cities

print('The different cities are: ', groups['city'].unique())
#Subset all groups to only those in SF or South SF

sfGroups = groups[(groups['city'] == 'San Francisco') | (groups['city'] == 'South San Francisco')]

sfGroups.head(3)
#Limit the events to only ones where the groups are in SF or South SF

sfEvents = events[events['group_id'].isin(list(sfGroups['group_id']))]
#Check category appearance

categories.head(3)
#Join the categories set to get category names

sfGroupsAndCats = sfGroups.join(categories.set_index('category_id'), on='category_id')

sfGroupsAndCats.head(3)
#Get counts and categories 

vals = sfGroupsAndCats['category.shortname'].value_counts()

cats = list(vals.keys())
#Make the plot

plt.figure(figsize=(12,6))

plt.bar(x=cats, height=vals)

plt.xticks(cats, cats, rotation='vertical');

plt.title('Category Distribution - San Francisco');
#Restrict subtopics to SF

sfgrouptopics = grouptopics[grouptopics['group_id'].isin(list(sfGroups['group_id']))]
#Create new df for SF groups and subtopics

groupsAndSubtopics = sfgrouptopics.join(sfGroups[['group_id', 'category_id']].set_index('group_id'), on='group_id')
#Check the result

groupsAndSubtopics.sample(5)
from collections import Counter

#Set a category

category = 2



#Grab all the subtopics counts in that category, grab the top 'n' most common

subtopicsInCat = Counter(groupsAndSubtopics[groupsAndSubtopics['category_id']==category]['topic_name']).most_common(30)



#List of subtopics and their counts

topics = [e[0] for e in subtopicsInCat]

counts = [e[1] for e in subtopicsInCat]



#Make the plot

plt.figure(figsize=(12,6))

plt.bar(x=topics, height=counts)

plt.xticks(topics, topics, rotation='vertical');

plt.title(list(categories[categories['category_id']==category]['shortname'])[0]+ ' Category Distribution - San Francisco');



#Possibly limit to minimum of 2/3/4/5 occurances?