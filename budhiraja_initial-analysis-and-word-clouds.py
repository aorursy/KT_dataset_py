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
reviews = pd.read_csv('../input/Reviews.csv')

reviews.head()
#Plotting Number of Scores i.e. distribution

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt



scores = list(reviews['Score'])

#scores = list(reviews['Score']).unique() ==> [1,2,3,4,5] => No fractional Ratings like 2.5 

freqs = Counter(scores)

print (freqs)

x = list(freqs.keys())

y = list(freqs.values())

width = 1/1.5

plt.bar( x, y, width, color="red" )

plt.xticks(np.arange(1, 6) + width/2.0,freqs.keys())

plt.show()



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





from wordcloud import WordCloud

import sqlite3

import pandas as pd

import nltk

import string

import matplotlib.pyplot as plt





con = sqlite3.connect('../input/database.sqlite')

total_data = pd.read_sql_query("""

SELECT Score, Summary

FROM Reviews

WHERE Score != 3

""", con)



positive_words = ''

negative_words = ''



def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



Score = total_data['Score']

Score = Score.map(partition)



tmp = total_data

tmp['Score'] = tmp['Score'].map(partition)



intab = string.punctuation

outtab = "                                "

trantab = str.maketrans(intab, outtab)







pos = total_data.loc[total_data['Score'] == 'positive']

pos = pos[0:10000]



neg = total_data.loc[total_data['Score'] == 'negative']

neg = neg[0:10000]



from nltk.corpus import stopwords



for val in pos["Summary"]:

    text = val.lower()

    text = text.translate(trantab)

    tokens = nltk.word_tokenize(text)

    tokens = [word for word in tokens if word not in stopwords.words('english')]

    for words in tokens:

        positive_words = positive_words + words + ' '

        

plt.ion()

print ("Positive Word-Cloud")

wordcloud = WordCloud(max_font_size=40).generate(positive_words)

plt.figure()

plt.imshow(wordcloud)

plt.axis("off")

plt.show()



for val in neg["Summary"]:

    text = val.lower()

    text = text.translate(trantab)

    tokens = nltk.word_tokenize(text)

    tokens = [word for word in tokens if word not in stopwords.words('english')]

    if 'good' in tokens:

        tokens.remove('good')

    for words in tokens:

        negative_words = negative_words + words + ' '

wordcloud = WordCloud(max_font_size=40).generate(negative_words)

plt.figure()

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
