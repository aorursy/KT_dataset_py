# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import sqlite3

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()

c.execute('SELECT DISTINCT year FROM papers ORDER BY year ASC')

print(c.fetchall())
def get_words(title):

    title = title.lower()

    new_words = {}

    candidates = title.split(' ')

    skipwords = ['in', 'for', 'a', 'an', 'of', 'and', 'the', 'with', 'to', 'on', 'from',

                 'by', 'via', 'as', 'is', 'that', 'are', 'this', 'can', 'our', 'which',

                 'not', 'all', 'where', 'such', 'has', 'also', 'any', 'its', 'but']

    for c in candidates:

        if c in skipwords or len(c) < 3:

            continue

        if c in new_words:

            new_words[c] += 1

        else:

            new_words[c] = 1

    return new_words





def merge_words(words, new_words):

    for word, count in new_words.items():

        if word in words:

            words[word] += count

        else:

            words[word] = count



words = {}

c = conn.cursor()

c.execute('SELECT title FROM papers WHERE year = 2016')

titles = c.fetchall()

for title in titles:

    title = title[0]

    new_words = get_words(title)

    merge_words(words, new_words)



for word, count in sorted(words.items(), key=lambda n: n[1], reverse=True):

    print("%i: %s" % (count, word))

    # print(count* ("%s " % word)) # <- use this for https://github.com/amueller/word_cloud

 

# https://github.com/MartinThoma/MartinThoma.github.io/blob/pelican/images/2016/12/nips-2016-wordcloud.png
words = {}

c = conn.cursor()

c.execute('SELECT paper_text FROM papers WHERE year = 2016')

titles = c.fetchall()

for title in titles:

    title = title[0]

    new_words = get_words(title)

    merge_words(words, new_words)



i = 0

for word, count in sorted(words.items(), key=lambda n: n[1], reverse=True):

    print("%i: %s" % (count, word))

    i += 1

    if i == 100:

        break