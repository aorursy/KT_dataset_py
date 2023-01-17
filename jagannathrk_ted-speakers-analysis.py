import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import operator

from string import punctuation

import re

from collections import Counter



ts = pd.read_csv("../input/transcripts.csv")

ts.head(10)
def removeMusicAndApplause(transcript):

    r = re.compile(r"(\(Music.*?\))|(\(Applause.*?\))|(\(Video.*?\))|(\(Sing.*?\))|(\(Cheer.*?\))", re.IGNORECASE)

    return r.sub("", transcript).strip()



ts['raw_transcript'] = ts['transcript']

ts['transcript'] = ts['transcript'].apply(removeMusicAndApplause)



ts.head()
punctuation_table = str.maketrans({key: None for key in punctuation})

ts['first_word'] = ts['transcript'].map(lambda x: x.split(" ")[0]).map(lambda x: x.translate(punctuation_table))

opening_counts = ts.first_word.value_counts()

opening_counts.head(30).plot(kind='barh', figsize=(6,6), title='Most Common Opening Words', color='g')
r = re.compile(r'[.!?:-]'.format(re.escape(punctuation)))

ts['first_250'] = ts['transcript'].map(lambda x: x[0:250])

phrases = []

for x in ts['first_250'].tolist():

    openings = r.split(x)

    phrases.append(openings[0])

    

phrase_count = Counter(phrases)

phrase_count = sorted(phrase_count.items(), key=operator.itemgetter(1))



phrase, count = zip(*phrase_count)

phrase = [x for _,x in sorted(zip(count,phrase), reverse=True)]

count = sorted(count, reverse=True)

y_pos = np.arange(len(phrase))



number_of_phrases = 20



plt.figure(figsize=(6,6))

plt.title('Most Common Opening Phrases')

plt.yticks(y_pos[:number_of_phrases], phrase[:number_of_phrases])

plt.xticks(np.arange(25))

plt.barh(y_pos[:number_of_phrases], count[:number_of_phrases],color='g')

plt.show()
ts['speaker'] = ts['transcript'].map(lambda x: x.split(":")[0])

mask = ts['speaker'].map(lambda x: x.split(" ")).map(lambda x: len(x)) >= 4

ts.loc[mask, 'speaker'] = 'unknown'



ts[ts['speaker'] != "unknown"]['speaker'].head(10)
# Sanity check that there we caught the 20 rows with Chris Anderson as the annotated speaker.

by_Chris = ts[ts['speaker'] == "Chris Anderson"]

len(by_Chris)
r = re.compile(r'[.!?:-]'.format(re.escape(punctuation)))

raw_openings = by_Chris['transcript'].apply(lambda x: x[0:250])

phrases = []

for x in raw_openings.tolist():

    openings = r.split(x)

    phrases.append(openings[1]) # Skip the "Chris Anderson:"

    

phrase_count = Counter(phrases)

phrase_count = sorted(phrase_count.items(), key=operator.itemgetter(1))



phrase, count = zip(*phrase_count)

phrase = [x for _,x in sorted(zip(count,phrase), reverse=True)]

count = sorted(count, reverse=True)

y_pos = np.arange(len(phrase))



number_of_phrases = 21



plt.figure(figsize=(6,6))

plt.title('Chris Anderson\'s Most Common Opening Phrases')

plt.yticks(y_pos[:number_of_phrases], phrase[:number_of_phrases])

plt.xticks(np.arange(25))

plt.barh(y_pos[:number_of_phrases], count[:number_of_phrases],color='g')

plt.show()