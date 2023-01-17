# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import Data
df = pd.read_csv('../input/tmdb_5000_movies.csv')
# First 5 Rows
df.head(5)
df_tidy = df.copy()
df_tidy.head(5)
import time
df_tidy.info()
def tidying_row(dataframe):
    start = time.clock()
    lenght=len(dataframe)
    for row in range(0,lenght):
        new_list = []
        iter_time = len(dataframe.iloc[row].split('\"'))//6
        stry = dataframe.iloc[row].split('\"')
        for i in range(1,iter_time+1):
            new_list.append(stry[(i*6)-1])
        dataframe.iloc[row] = new_list
    print (time.clock() - start)
tidying_row(df_tidy.genres.head(4803))
tidying_row(df_tidy.keywords.head(4803))
tidying_row(df_tidy.production_companies.head(4803))
tidying_row(df_tidy.production_countries.head(4803))
tidying_row(df_tidy.spoken_languages.head(4803))
df_tidy.head(5)
df_tidy_high_vote = df_tidy[df_tidy.vote_average > 8]
df_tidy_high_vote.corr()
plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(df_tidy_high_vote.budget, df_tidy_high_vote.revenue, alpha=0.5)
plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.show()
df_tidy_high_vote.describe()
df_tidy_high_vote.info()
df_tidy_high_vote.homepage.fillna('UNKNOWN',inplace=True)
df_tidy_high_vote.tagline.fillna('UNKNOWN',inplace=True)
df_tidy_high_vote.info()
df_tidy_high_vote.boxplot(column='revenue', figsize=(8,8))
plt.show()
alist = ['Science Fiction']
df_tidy_high_vote[df_tidy_high_vote.genres.apply(lambda x :set(alist).issubset(x))]
df_tidy_high_vote['SciFi'] = np.where(df_tidy_high_vote.genres.apply(lambda x :set(alist).issubset(x)), 'Yes', 'No')
df_tidy_high_vote.head()
df_tidy_high_vote.boxplot(column='revenue', by='SciFi', figsize=(8,8))
plt.show()