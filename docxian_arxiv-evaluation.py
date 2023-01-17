import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import json

import time

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



from tqdm.notebook import tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
my_file = '../input/arxiv/arxiv-metadata-oai-snapshot.json'
# get access to metadata (lazy evaluation)

def get_metadata():

    with open(my_file, 'r') as f:

        for line in f:

            yield line

            

metadata = get_metadata()
# look at titles and abstracts of first few papers

for ind, paper in enumerate(metadata):

    paper = json.loads(paper)

    print(ind)

    print(paper['title'])

    print(paper['abstract'])

    if (ind == 4):

        break
# check full structure of a paper record

paper
text = ''

for ind, paper in tqdm(enumerate(metadata), total=1747307):

    paper = json.loads(paper)

    add_txt = paper['title']

    text = text + add_txt
# number of papers

print('Number of papers: ', ind)
# check lengths of all titles

print('Length of combined title texts: ', len(text))
# wordcloud of all paper titles

stopwords = set(STOPWORDS)



t1 = time.time()

wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

t2 = time.time()
print('Elapsed time for word cloud: ', np.round(t2 - t1,2), 'secs')
metadata = get_metadata()



text = ''

cnt = 0

my_cat = 'cs.AI' # define category of interest



for ind, paper in tqdm(enumerate(metadata), total=1747307):

    paper = json.loads(paper)

    cats = paper['categories']

    if (cats.find(my_cat)>=0):

        # print(cats) # for debugging

        add_txt = paper['title']

        text = text + add_txt

        cnt = cnt + 1
# number of papers

print('Number of papers: ', cnt)
# check lengths of all titles

print('Length of combined title texts: ', len(text))
# wordcloud of selected paper titles

stopwords = set(STOPWORDS)



t1 = time.time()

wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

t2 = time.time()
print('Elapsed time for word cloud: ', np.round(t2 - t1,2), 'secs')
metadata = get_metadata()



text = ''

cnt = 0

my_cat = 'stat.ML' # define category of interest



for ind, paper in tqdm(enumerate(metadata), total=1747307):

    paper = json.loads(paper)

    cats = paper['categories']

    if (cats.find(my_cat)>=0):

        # print(cats) # for debugging

        add_txt = paper['title']

        text = text + add_txt

        cnt = cnt + 1
# number of papers

print('Number of papers: ', cnt)
# check lengths of all titles

print('Length of combined title texts: ', len(text))
# wordcloud of selected paper titles

stopwords = set(STOPWORDS)



t1 = time.time()

wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,

                      width = 600, height = 400,

                      background_color="white").generate(text)

plt.figure(figsize=(12,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

t2 = time.time()
print('Elapsed time for word cloud: ', np.round(t2 - t1,2), 'secs')