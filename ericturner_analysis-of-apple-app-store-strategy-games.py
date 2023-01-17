# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualizations

import re

data_csv = "/kaggle/input/appstore_games.csv"

games_data = pd.read_csv(data_csv)
print("Total Records: " + str(games_data.count()[1]))
sns.set()

sns.countplot(x="Average User Rating", data=games_data);
sns.set()

sns.countplot(x="Age Rating", data=games_data);
# function to find the count of each bin in the above graph and print out the exact values in a nice format

def print_age_rating(rating):

    exact_count = games_data['Age Rating'].str.contains(rating).value_counts()[True]

    total_items = len(games_data.index)

    percent = '{0:.3g}'.format((exact_count/total_items)*100) # three digits

    print('[' + str(exact_count) + " / " + str(total_items) + '] (' + percent + '%) strategy games are in the ' + rating + ' age range. \r\n')



for x in games_data['Age Rating'].unique(): print_age_rating(x) 
# since there is a comma delimited list in languages, lets separate each lang into its own column

languages = pd.DataFrame(games_data['Languages'].str.split(', ',expand=True))

# now lets merge all the columns into one master languages column

languages = pd.DataFrame(languages.values.ravel(), columns = ["Languages"])

# get a total of all the languages and their counts into a df

languages = pd.DataFrame(languages['Languages'].value_counts().reset_index())

# rename columns

languages.columns = ['Language', 'Count']



# grab top 10 (out of 115) most used languages for display

sns.barplot(x="Language", y="Count", data=languages.head(10));
# function to find the count of each bin in the above graph and print out the exact values in a nice format

def print_top_languages(lang):

    # find the language in the table, then grab the index of the row and grab the Count value

    exact_count = languages.loc[languages['Language'] == lang]['Count'][languages.loc[languages['Language'] == lang].index[0]]

    total_items = len(games_data.index)

    percent = '{0:.4g}'.format((exact_count/total_items)*100) # three digits

    print('[' + str(exact_count) + " / " + str(total_items) + '] (' + percent + '%) strategy games use '+lang+'. \r\n')



for x in languages.head(10)['Language']: print_top_languages(x) 
# count the number of commas in each ['Languages'] then add 1 (to count the original Language)

# this will show you how many languages some apps are

multi = games_data['Languages'].str.count(r', ') + 1

multi_lingual = pd.DataFrame(multi.value_counts().reset_index())



multi_lingual.columns = ['Languages', 'Count']



sns.barplot(x="Languages", y="Count", data=multi_lingual.head(10));