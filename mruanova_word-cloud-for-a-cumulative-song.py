import collections

import re

import matplotlib.pyplot as plt

# % matplotlib inline

file = open('../input/cumulativesong/la-feria-de-cepillin.txt', 'r')

file = file.read()

stopwords = set(line.strip() for line in open('../input/cumulativesong/stopwords.txt'))

stopwords = stopwords.union(set(['a', 'i', 'mr', 'ms', 'mrs', 'one', 'two', 'said']))

wordcount = collections.defaultdict(int)

# \W is regex for characters that are not alphanumerics.

pattern = r"\W"

for word in file.lower().split():

    # all non-alphanumerics are replaced with a blank space using re.sub

    word = re.sub(pattern, '', word)

    if word not in stopwords:

        wordcount[word] += 1

most_common = sorted(wordcount.items(), key=lambda k_v: k_v[1], reverse=True)

for word, count in most_common:

    print(word, ":", count)

# Draw the bart chart

most_common = dict(most_common)

names = list(most_common.keys())

values = list(most_common.values())

plt.xticks(rotation='vertical')

plt.bar(range(len(most_common)),values,tick_label=names)

plt.savefig('bar.png')

plt.show()
from wordcloud import WordCloud

wc = WordCloud().generate_from_frequencies(wordcount)

plt.figure()

plt.imshow(wc, interpolation="bilinear")

plt.axis("off")

plt.show()