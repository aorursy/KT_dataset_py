# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# pandas for handling our dataset

import pandas as pd

# numpy for numeric operations

import numpy as np

from collections import defaultdict



# matplotlib for plotting

import matplotlib.pyplot as plt

# use ggplot style

plt.style.use('ggplot')

# seaborn for beautiful visualizations

import seaborn as sns

# regualar expression

import re

# print inline in this notebook

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read the data set using pandas .read_csv() method

df_job_skills = pd.read_csv('../input/job_skills.csv')

# print the top 5 row from the dataframe

df_job_skills.head()
# most popular language list 

programing_language_list = ['python', 'java', 'c++', 'php', 'javascript', 'objective-c', 'ruby', 'perl','c','c#', 'sql','kotlin']
# get our Minimum Qualifications column and convert all of the values to a list

minimum_qualifications = df_job_skills['Minimum Qualifications'].tolist()

# let's join our list to a single string and lower case the letter

miniumum_qualifications_string = "".join(str(v) for v in minimum_qualifications).lower()
# find out which language occurs in most in minimum Qualifications string

wordcount = dict((x,0) for x in programing_language_list)

for w in re.findall(r"[\w'+#-]+|[.!?;’]", miniumum_qualifications_string):

    if w in wordcount:

        wordcount[w] += 1

# print

print(wordcount)
# sort the dict

programming_language_popularity = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
# make a new dataframe using programming_language_popularity for easy use cases

df_popular_programming_lang = pd.DataFrame(programming_language_popularity,columns=['Language','Popularity'])

# Capitalize each programming language first letter

df_popular_programming_lang['Language'] = df_popular_programming_lang.Language.str.capitalize()

df_popular_programming_lang = df_popular_programming_lang[::-1]
# plot

df_popular_programming_lang.plot.barh(x='Language',y='Popularity',figsize=(10,8), legend=False)

# add a suptitle

plt.suptitle("Programming Languages popularity at Google Jobs", fontsize=18)

# set xlabel to ""

plt.xlabel("")

# change xticks fontsize to 14

plt.yticks(fontsize=14)

# finally show the plot

plt.show()
miniumum_qualifications_string = " ".join(str(v) for v in minimum_qualifications)
degree_list = ["BA", "BS", "Bachelor's", "PhD"]
wordcount = dict((x,0) for x in degree_list)

for w in re.findall(r"[\w']+|[.,!?;’]", miniumum_qualifications_string):

    if w in wordcount:

        wordcount[w] += 1

# print

print(wordcount)

degree_popularity = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
df_degree_popular = pd.DataFrame(degree_popularity,columns=['Degree','Popularity'])
df_degree_popular = df_degree_popular[::-1] 

# plot

df_degree_popular.plot.barh(x='Degree',y='Popularity',figsize=(15,10), stacked=True)

# add a suptitle

plt.suptitle("Popularity of academic degree at Google Jobs ", fontsize=18)

# set xlabel to ""

plt.xlabel("")

# change xticks fontsize to 14

plt.yticks(fontsize=18)

# finally show the plot

plt.show()
# this portion of code is taken from https://www.kaggle.com/djcarlos/are-you-experienced-enough-to-work-at-google 

years_exp = defaultdict(lambda: 0)



for w in re.findall(r'([0-9]+) year', miniumum_qualifications_string):

     years_exp[w] += 1

        

print(years_exp)
years_exp = sorted(years_exp.items(), key=lambda kv: kv[1], reverse=True)
df_years_exp = pd.DataFrame(years_exp,columns=['Years of experience','Popularity'])

df_years_exp = df_years_exp[::-1] 
# plot

df_years_exp.plot.barh(x='Years of experience',y='Popularity',figsize=(10, 8), legend=False,stacked=True)

# add a suptitle

plt.title("Years of experiences needed for Google Jobs", fontsize=18)

# set xlabel to ""

plt.xlabel("Popularity", fontsize=14)

plt.ylabel("Years of experiences",fontsize=18)

# change xticks fontsize to 14

plt.yticks(fontsize=18)

# finally show the plot

plt.show()
df_job_skills['Experience'] = df_job_skills['Minimum Qualifications'].str.extract(r'([0-9]+) year')
dff = df_job_skills[['Experience','Category']]

dff = dff.dropna()
plt.figure(figsize=(10,15))

plt.title('Experiences needed in different job category', fontsize=24)

sns.countplot(y='Category', hue='Experience', data=dff, hue_order=dff.Experience.value_counts().iloc[:3].index)

plt.yticks(fontsize=18)

plt.show()
# where is most job located

threshold = 10

location_value_counts = df_job_skills.Location.value_counts()

to_remove = location_value_counts[location_value_counts <= threshold].index

df_job_skills['Location'].replace(to_remove, np.nan, inplace=True)

location_value_counts = df_job_skills.Location.value_counts()

location_value_counts = location_value_counts[::-1]
location_value_counts.plot.barh(figsize=(15, 15))

# add a suptitle

plt.title("Google Jobs Location Popularity", fontsize=24)

# set xlabel to ""

plt.xlabel("Popularity", fontsize=20)

plt.ylabel("Location",fontsize=20)

# change xticks fontsize to 14

plt.yticks(fontsize=24)

# finally show the plot

plt.show()
category_value_counts = df_job_skills.Category.value_counts()

category_value_counts = category_value_counts[::-1]

category_value_counts.plot.barh(figsize=(15, 15))

# add a suptitle

plt.title("What is the most popular job category at Google?", fontsize=24)

# set xlabel to ""

plt.xlabel("Popularity", fontsize=20)

plt.ylabel("Job Category",fontsize=20)

# change xticks fontsize to 14

plt.yticks(fontsize=24)

# finally show the plot

plt.show()
plt.figure(figsize=(10,15))

plt.title('Google job categories popularity in different locations', fontsize=24)

sns.countplot(y='Location', hue='Category', data=df_job_skills, hue_order=dff.Category.value_counts().iloc[:3].index)

plt.yticks(fontsize=18)

plt.show()