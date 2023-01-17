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
data = pd.read_csv("../input/wordgame_20170628.csv")
data.head()
data.shape
data['author'].unique().shape
data['author'].value_counts().head(10)
data['source'].unique()
data['source'].value_counts()
import numpy as np # linear algebra

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data[data['source']=='wrongplanet']['word1']))

fig = plt.figure(1,figsize=(12,18))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data[data['source']=='gog']['word1']))
fig = plt.figure(1,figsize=(12,18))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data[data['source']=='sas']['word1']))
fig = plt.figure(1,figsize=(12,18))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()