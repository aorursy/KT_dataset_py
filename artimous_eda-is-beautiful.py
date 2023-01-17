from wordcloud import WordCloud

from wordcloud import STOPWORDS

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import random

import string

import os

import re
def clean_file(train):

    train = train.split(".")

    train = [re.sub('['+string.punctuation+']', '', x) for x in train]

    return train
LyricsData = []

SongLength = []



df = pd.read_csv("../input/Lyrics1.csv")

df = df.sample(15000)

song = ['.'.join(x.splitlines()) for x in df["Lyrics"]]

LyricsData = '.'.join(song)

song = [len(re.findall('\w+', x)) for x in df["Lyrics"]]

SongLength = song



train  = clean_file(LyricsData)

Length = [len(x.split(" ")) for x in train]

full   = ' '.join(train)

train_copy = train
plt.figure(figsize=(15, 15))

plt.subplot(2, 1, 1)

sns.set_style("darkgrid")

plt.plot(SongLength)

plt.title('A tale of lengths')

plt.ylabel('Song lengths')



plt.subplot(2, 1, 2)

plt.plot(Length, 'r.-')

plt.xlabel('Songs')

plt.ylabel('Line lengths')

plt.show()
plt.figure(figsize=(15, 15))

plt.subplot(2, 1, 1)

sns.distplot(np.asarray(SongLength));



plt.subplot(2, 1, 2)

sns.kdeplot(np.asarray(Length), shade = True);

plt.show()
stopwords = set(STOPWORDS)

wordcloud_hc = WordCloud(width=900, height=900, relative_scaling=.9,stopwords=stopwords).generate(full)

plt.figure(figsize=(15, 15))

plt.imshow(wordcloud_hc)

plt.axis("off")

plt.show()
words = full.split(" ")

words = filter(None, words)

ups = [w for w in words if w[0].isupper()]
wordcloud_hc = WordCloud(width=900, height=900, relative_scaling=.8,stopwords=stopwords).generate(" ".join(ups))

plt.figure(figsize=(10, 10))

plt.imshow(wordcloud_hc)

plt.axis("off")

plt.show()
data = []

df = pd.read_csv("../input/Lyrics1.csv")

df = df[df["Band"]=="Pink Floyd"]

data.append(df)

df = pd.read_csv("../input/Lyrics2.csv")

df = df[df["Band"]=="Pink Floyd"]

data.append(df)



floyd = pd.concat(data)
FloydLyrics = []

FloydLength = []



song = ['.'.join(x.splitlines()) for x in floyd["Lyrics"]]

FloydLyrics = '.'.join(song)

song = [len(re.findall('\w+', x)) for x in floyd["Lyrics"]]

FloydLength = song



train  = clean_file(FloydLyrics)



Length = []

for lyric in floyd["Lyrics"]:

    LineLengths = [len(x.split(" ")) for x in lyric.splitlines()]

    Length.append(int(sum(LineLengths)/len(LineLengths)))

full   = ' '.join(train)
with sns.axes_style("white"):

     sns.jointplot(x=np.asarray(Length), y=np.asarray(FloydLength), kind="hex", color="k");
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(np.asarray(Length), np.asarray(FloydLength), cmap=cmap, n_levels=60, shade=True);
stopwords = set(STOPWORDS)

wordcloud_hc = WordCloud(width=900, height=900, relative_scaling=.8,stopwords=stopwords).generate(full)

plt.figure(figsize=(13, 13))

plt.imshow(wordcloud_hc)

plt.axis("off")

plt.show()
stopwords.add("Live")

stopwords.add("Version")

stopwords.add("BBC")

stopwords.add("Radio")

stopwords.add("Roger")

stopwords.add("Waters")

stopwords.add("Excerpt")

stopwords.add("Original")

stopwords.add("Pt")

stopwords.add("Tour")

stopwords.add("Band")

stopwords.add("Demo")

stopwords.add("Mix")

stopwords.add("Session")

stopwords.add("Stereo")

stopwords.add("Take")

full = " ".join(floyd["Song"])

wordcloud_hc = WordCloud(width=900, height=900, relative_scaling=.8,stopwords=stopwords).generate(full)

plt.figure(figsize=(13, 13))

plt.imshow(wordcloud_hc)

plt.axis("off")

plt.show()
train = train_copy
train = [x.lower() for x in train]

phrases3 = []

phrases4 = []

phrases5 = []

for i in range(len(train)):

    words = train[i].split(" ")

    for j in range(len(words)-2):

        phrases3.append(words[j] + " " + words[j+1] + " " + words[j+2])

    for j in range(len(words)-3):

        phrases4.append(words[j] + " " + words[j+1] + " " + words[j+2]+ " " + words[j+3])

    for j in range(len(words)-4):

        phrases5.append(words[j] + " " + words[j+1] + " " + words[j+2]+ " " + words[j+3]+ " " + words[j+4])
from collections import Counter, OrderedDict

phrase3_counts = OrderedDict(Counter(phrases3).most_common(30))

phrase4_counts = OrderedDict(Counter(phrases4).most_common(30))

phrase5_counts = OrderedDict(Counter(phrases5).most_common(30))
del phrase3_counts['  ']

del phrase4_counts['   ']

del phrase5_counts['    ']
df = pd.DataFrame.from_dict(phrase3_counts, orient='index')

plt = df.plot(kind='bar', figsize = (13,13), legend = False)
df = pd.DataFrame.from_dict(phrase4_counts, orient='index')

plt = df.plot(kind='bar', figsize = (13,13), legend = False)
df = pd.DataFrame.from_dict(phrase5_counts, orient='index')

plt = df.plot(kind='bar', figsize = (13,13), subplots=True, legend = False)
phrases4 = [x for x in phrases4 if (x.split(" ")[0] != x.split(" ")[1]

                                    and x.split(" ")[1] != x.split(" ")[2]

                                    and x.split(" ")[2] != x.split(" ")[3]

                                   )]
phrase4_counts = OrderedDict(Counter(phrases4).most_common(30))
df = pd.DataFrame.from_dict(phrase4_counts, orient='index')

plt = df.plot(kind='pie', figsize = (13,13),subplots=True, legend = False)