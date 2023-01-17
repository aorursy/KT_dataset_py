import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

corpus = ""

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename[-3:] == 'txt':

            file = open(dirname+'/'+filename, "r")

            corpus = corpus + file.read() 

            file.close()
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import matplotlib.pyplot as plt



a = np.array(Image.open('../input/trumpmmm/trump-fail-004.jpg'))



#wordcloud = WordCloud(stopwords=STOPWORDS, mask =  ).generate(corpus)

wordcloud = WordCloud(stopwords=STOPWORDS, mask=a).generate(corpus)



plt.subplots(figsize=(100,100))

plt.clf()

plt.title('All speeches')

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

import matplotlib.pyplot as plt



a = np.array(Image.open('../input/usamap/USA-States-Color-Map.jpg'))



#wordcloud = WordCloud(stopwords=STOPWORDS, mask =  ).generate(corpus)

wordcloud = WordCloud(stopwords=STOPWORDS, mask=a).generate(corpus)



plt.subplots(figsize=(100,100))

plt.clf()

plt.title('All speeches')

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
file_content = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename[-3:] == 'txt':

            file = open(dirname+'/'+filename, "r")

            file_content.append(file.read())

        

            file.close()

        

file_names = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename[-3:] == 'txt':

            file_names.append(os.path.join(filename))

        

res = {file_names[i]: file_content[i] for i in range(len(file_names))}

df = pd.DataFrame.from_records([res]).T

df
for i in range(0,35):

    wordcloud = WordCloud(stopwords=STOPWORDS).generate(df.iloc[i,0])



    plt.subplots(figsize=(15,15))

    plt.clf()

    plt.title(df.index[i])

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
import nltk

tokens = nltk.word_tokenize(corpus)

tagged = nltk.pos_tag(tokens)



tags = []



for i in range (len(tagged)):

    if tagged[i][1][0] == 'N':

        tags.append(tagged[i][0])

    

str_tags = ' '.join([str(elem) for elem in tags])



wordcloud = WordCloud(stopwords=STOPWORDS, mask=a).generate(str_tags)



plt.subplots(figsize=(100,100))

plt.clf()

plt.title('Most Nouns used')

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
import nltk

tokens = nltk.word_tokenize(corpus)

tagged = nltk.pos_tag(tokens)



tags = []



for i in range (len(tagged)):

    if tagged[i][1][0] == 'V':

        tags.append(tagged[i][0])

    

str_tags = ' '.join([str(elem) for elem in tags])



wordcloud = WordCloud(stopwords=STOPWORDS, mask=a).generate(str_tags)



plt.subplots(figsize=(100,100))

plt.clf()

plt.title('Most Verbs used')

plt.imshow(wordcloud)

plt.axis('off')

plt.show()