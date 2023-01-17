# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt

import scipy.stats as stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load data from database

conn = sqlite3.connect('../input/database.sqlite')



# get score data

score_query = '''

SELECT reviewid, score

FROM reviews

'''

score_df = pd.read_sql_query(score_query, conn)



# get genre data

genre_query = '''

SELECT *

FROM genres

'''

genre_df = pd.read_sql_query(genre_query, conn)



conn.close()



# replace NaN in genre with "Not specified"

genre_df.fillna(value="Not specified", inplace=True)



# convert genre data to sets of all genres associated with a release

grouped = genre_df.groupby('reviewid')

genre_df = grouped.aggregate(lambda x: set(x))



# join reviews and genres

result = score_df.join(genre_df, on='reviewid')

assert len(score_df) == len(result)



# Mean of all scores

popmean = result['score'].mean()

print("Mean of %d reviews: %f" % (result['reviewid'].count(), popmean))



# Standard deviation

print("Standard deviation of reviews: %f" % result['score'].std(ddof=0))
# plot histogram

plt.hist(score_df['score'])

plt.show()



# get genre counts and mean review scores

means_and_counts = result.groupby(result['genre'].apply(tuple))['score'].agg(['count', 'mean'])

assert means_and_counts['count'].sum() == len(result)



# sort by mean

means_and_counts = means_and_counts.sort_values('mean', ascending=False)



# filter out genres with less than 50 reviews

means_and_counts = means_and_counts[means_and_counts['count'] > 50].reset_index()



# print out results

print(means_and_counts)
# one-way ANOVA

data = []

for index, row in means_and_counts.iterrows():

    data.append(result[result['genre'].apply(tuple) == row['genre']].score.tolist())

    # It might be better to create a string from the genre list and create another dataframe. 



(stat, pvalue) = stats.f_oneway(*data)

print("One-way ANOVA on genre values:")

print("F-stat: %f, p-value: %f" % (stat, pvalue))
# t-tests

t_tests_headers = ['genre', 't', 'prob', 'Reject_Null']

t_tests = pd.DataFrame(index=range(0, len(data)), columns=t_tests_headers)



for index in range(len(data)):

    gs = ', '.join(means_and_counts['genre'][index])

    t_tests['genre'][index] = gs

    

    (t, prob) = stats.ttest_1samp(data[index], popmean)

    t_tests['t'][index] = t

    t_tests['prob'][index] = prob

    if prob < 0.05:

        t_tests['Reject_Null'][index] = True

    else:

        t_tests['Reject_Null'][index] = False



print(t_tests.sort_values('t'))