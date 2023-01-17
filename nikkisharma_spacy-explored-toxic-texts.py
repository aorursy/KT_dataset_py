# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
df.head()
df['comment_text'][8]
import pprint
doc = nlp('The ranchers seem motivated by mostly by greed; no one should have the right to allow their animals destroy public land.')



pprint.pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
doc1=nlp("My Name is Nikki Sharma. I love Nature Languge Processing.")

for word in doc1.ents:

    print(word)

text= df['comment_text'][19]

text
article = nlp(text)

len(article.ents)
labels = [x.label_ for x in article.ents]

Counter(labels)
doc = nlp(text)

pprint.pprint([(X.text, X.label_) for X in doc.ents])
sentence = df['comment_text'][119]

sentence
displacy.render(nlp(str(sentence)), jupyter=True, style='ent')
sentence_1 = df['comment_text'][350]

sentence_1
displacy.render(nlp(str(sentence_1)), jupyter=True, style='ent')
sentence_2 = df['comment_text'][970]

sentence_2
displacy.render(nlp(str(sentence_2)), jupyter=True, style='ent')
displacy.render(nlp(str(sentence)), style='dep', jupyter = True, options = {'distance': 120})
[(x.orth_,x.pos_, x.lemma_) for x in [y 

                                      for y

                                      in nlp(str(sentence)) 

                                      if not y.is_stop and y.pos_ != 'PUNCT']]
dict([(str(x), x.label_) for x in nlp(str(sentence_2)).ents])
article = nlp('By the time Prime Minister Boris Johnson finished taking questions in Parliament on Wednesday, he had ushered in a new season of political mayhem in Britain, one in which the voters are now as likely as their feuding leaders to resolve the questions over how and when Britain should leave the European Union. The raucous spectacle in the House of Commons illustrated the obstacles Mr. Johnson will face as he tries to lead Britain out of the European Union next month. On Wednesday, Parliament handed he prime minister two stinging defeats.It first blocked his plans to leave the union with or without an agreement. And it then stymied his bid, at least for the moment, to call an election for Oct. 15, out of fear he could secure a new majority in favor of breaking with Europe, deal or no deal.')
len(article.ents)
sentences = [x for x in article.sents]
print(sentences)
displacy.render(nlp(str(sentences)), jupyter=True, style='ent')
