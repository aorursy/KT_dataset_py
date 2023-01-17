# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Python defacto plotting library

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/movie_metadata.csv')

movies.shape
movies.head(n=2).T
movies.duplicated().sum()
movies[movies.duplicated()]['movie_title']
movies.drop_duplicates(inplace=True)

movies.shape
ch_freq = {}

# http://stackoverflow.com/questions/33327540/finding-letter-bigrams-in-text-using-python-regex

for match in movies.movie_title.str.findall('(?=(.{1}))'):

    # print(match)

    for ch in match:

        if ch in ch_freq:

            ch_freq[ch] += 1

        else:

            ch_freq[ch] = 1

# print(ch_freq)

unigrams = pd.DataFrame.from_dict(data=ch_freq, orient='index')

unigrams.columns = ['frequency']

unigrams.index.name = 'unigram'

unigrams.head()
unigrams.sort_values(by='frequency', ascending=False, inplace=True)

unigrams.head(n=10)
unigrams.tail(n=10)
# http://stackoverflow.com/questions/26358200/xticks-by-pandas-plot-rename-with-the-string

n = 50

unigrams['frequency'].head(n).plot(xticks=range(n), logy=True, figsize=(9,4))
n = 50

unigrams['frequency'].tail(n).plot(xticks=range(n), logy=True, figsize=(9,4))