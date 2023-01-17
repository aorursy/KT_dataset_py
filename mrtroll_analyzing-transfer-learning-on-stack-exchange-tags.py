import numpy as np

import pandas as pd

import nltk as nlt

from nltk.tokenize import word_tokenize

from subprocess import check_output

from nltk.stem import WordNetLemmatizer

print(check_output(["ls", "../input"]).decode("utf8"))
biology = pd.read_csv('../input/biology.csv')

cooking = pd.read_csv('../input/cooking.csv')

crypto = pd.read_csv('../input/crypto.csv')

diy = pd.read_csv('../input/diy.csv')

robotics = pd.read_csv('../input/robotics.csv')

travel = pd.read_csv('../input/travel.csv')



corpora = [biology,cooking,crypto,diy,robotics,travel]
biology.head()
print(biology.title[0])

print(biology.content[0])

print(biology.tags[0])
def tag_clean(data):

    tags = data.tags



    print('tokenize')

    tags = tags.apply(lambda x: word_tokenize(x))



    print('Lemmatizing')

    wordnet_lemmatizer = WordNetLemmatizer()

    tags = tags.apply(lambda x: [wordnet_lemmatizer.lemmatize(i) for i in x])

    tags = tags.apply(lambda x: [i for i in x if len(i)>2])

    return(tags)
tags_lemmatized = tag_clean(biology)
tags_lemmatized = tags_lemmatized.as_matrix()

tags_lemmatized = np.concatenate(tags_lemmatized)

tags_frequency_lemmatized = nlt.FreqDist(tags_lemmatized)

len(tags_frequency_lemmatized)
tags = biology.tags.apply(lambda x: word_tokenize(x))

tags = tags.as_matrix()

tags = np.concatenate(tags)

tags_frequency = nlt.FreqDist(tags)

len(tags_frequency)
corpora_tags_frequency = []

for corpus in corpora:

    tags = corpus.tags.apply(lambda x: word_tokenize(x))

    tags = tags.as_matrix()

    tags = np.concatenate(tags)

    tags_frequency = nlt.FreqDist(tags)

    corpora_tags_frequency.append(tags_frequency)
import operator

corpora_tags_frequency_sorted = []

for corpus_tags in corpora_tags_frequency:

    tags_sorted = sorted(corpus_tags.items(), key=operator.itemgetter(1), reverse=True)

    corpora_tags_frequency_sorted.append(tags_sorted)
for corpus_tags in corpora_tags_frequency_sorted:

    print(corpus_tags[0:3])
for key in corpora_tags_frequency[0]:

    found = True

    for corpus_tags in corpora_tags_frequency[1:]:

        if key not in corpus_tags:

            found = False

            break

    if found == True:

        print(key)

     
tags_shared = []

for key in corpora_tags_frequency[0]:

    found = True

    for corpus_tags in corpora_tags_frequency[1:]:

        if key in corpus_tags:

            tags_shared.append(key)

tags_shared = nlt.FreqDist(tags_shared)
tags_shared_sorted = sorted(tags_shared.items(), key=operator.itemgetter(1), reverse=True)

print(len(tags_shared_sorted))
tags_shared_sorted[0:10]