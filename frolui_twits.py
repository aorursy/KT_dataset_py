# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gensim

from gensim import corpora, models

from pprint import pprint
import pandas as pd

fifa = pd.read_csv("../input/world-cup-2018-tweets/FIFA.csv")
fifa.head(5)
like_sort = fifa.sort_values(by='Likes', ascending=False)

like_sort.head(5)
data = like_sort.head(1000)
document = [data['Tweet'].values[i] for i in range(0,1000)]
document = [str(x) for x in document]
text = [[text for text in x.split()] for x in document]
diction = corpora.Dictionary(text)
print(diction)
corps = [diction.doc2bow(doc, allow_update=True) for doc in text]
# Save the Dict and Corpus

diction.save('fifa2018.dict')  # save dict to disk

corpora.MmCorpus.serialize('fifa2018corpus.mm', corps)  # save corpus to disk
tfidf = models.TfidfModel(corps, smartirs='ntc')

for doc in tfidf[corps]:

    print([[diction[id], np.around(freq, decimals=3)] for id, freq in doc])
bigram = gensim.models.phrases.Phrases(text, min_count=3, threshold=1)

print(bigram[text[0]])
for i in range(0,10):

    print(bigram[text[i]])
counter = 0



for i in document:

    counter += 1 

    if 'face Russia' in i:

        print(str(counter) + ' - '+i)        
# Build the trigram models

trigram = gensim.models.phrases.Phrases(bigram[text], threshold=10)

# Construct trigram

print(trigram[bigram[text[0]]])
from re import fullmatch
lst = []



for x in range(0, 1000):

    for i in trigram[bigram[text[x]]]:

        if fullmatch('\w+_\w+_\w+', i) != None:

            if i not in lst:

                lst.append(i)

                print(i)

        

        
from gensim.models.word2vec import Word2Vec

from multiprocessing import cpu_count
w2v_model = Word2Vec(text, min_count = 0, workers=cpu_count())
w2v_model['football']
w2v_model.most_similar('football')
data_new = like_sort.head(100000)

document_new = [str(data_new['Tweet'].values[i]) for i in range(0,100000)]

text_new = [[text for text in x.split()] for x in document_new]
w2v_model.build_vocab(text_new, update=True)

w2v_model.train(text_new, total_examples=w2v_model.corpus_count, epochs=w2v_model.iter)
w2v_model.most_similar('football')
w2v_model.save('fifa18_w2v')