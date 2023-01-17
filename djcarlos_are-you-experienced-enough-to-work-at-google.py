# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# pandas for handling our dataset

import pandas as pd

# numpy for numeric operations

import numpy as np

# matplotlib for plotting

import matplotlib.pyplot as plt

# use ggplot style

plt.style.use('ggplot')

# seaborn for beautiful visualizations

import seaborn as sns

# regualar expression

import re

# Defaul Dictionary

from collections import defaultdict



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

programing_language_list = ['python', 'java', 'c++', 'php', 'javascript', 'objective-C', 'ruby', 'perl','c','c#', 'sql', 'swift','scala']
# get our Minimum Qualifications column and convert all of the values to a list

minimum_qualifications = df_job_skills['Minimum Qualifications'].tolist()

# let's join our list to a single string and lower case the letter

miniumum_qualifications_string = " ".join(str(v) for v in minimum_qualifications).lower()
# find out which language occurs in most in minimum Qualifications string

wordcount = dict((x,0) for x in programing_language_list)

for w in re.findall(r"\w+", miniumum_qualifications_string):

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
# plot

df_popular_programming_lang.plot.bar(x='Language',y='Popularity',figsize=(10,5))

# add a suptitle

plt.suptitle("Programming Languages at Google Jobs", fontsize=18)

# set xlabel to ""

plt.xlabel("")

# change xticks fontsize to 14

plt.xticks(fontsize=14)

# finally show the plot

plt.show()
years_exp = defaultdict(lambda: 0)



for w in re.findall(r'([0-9]+) year', miniumum_qualifications_string):

     years_exp[w] += 1
# make a new dataframe 

df_years_exp = pd.DataFrame.from_dict(years_exp, 'index')

df_years_exp = df_years_exp.reset_index()

df_years_exp.columns = ['Years', 'Frequency']

# Convert to int so 10 doesn't come before 2 ;)

df_years_exp = df_years_exp.astype('int')

# Sort by Year

df_years_exp.sort_values(by='Years', inplace=True)
# plot

df_years_exp.plot.bar(x='Years',y='Frequency',figsize=(10,5))

# add a suptitle

plt.suptitle("Minimum Years of Experience for Google Jobs", fontsize=18)

# set xlabel to ""

plt.xlabel("")

# change xticks fontsize to 14

plt.xticks(fontsize=14)

# finally show the plot

plt.show()