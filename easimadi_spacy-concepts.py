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
import spacy

import en_core_web_lg

from IPython.display import SVG, display

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

%matplotlib inline



nlp = en_core_web_lg.load()
pd.DataFrame([annotators for annotators in nlp.pipeline], columns = ['annotator','object'])


text ="""I was unaware and disappointed that I would have to pay for the engineer myself 

considering how long I am a customer and how much I pay each month. 

However the advisor was very pleasant. I must now wait until I can afford the advisor."""



text_example = ('GOP Sen. Rand Paul was assaulted in his home in Bowling Green, Kentucky, on Friday, '

        'according to Kentucky State Police. State troopers responded to a call to the senator\'s '

        'residence at 3:21 p.m. Friday. Police arrested a man named Rene Albert Boucher, who they '

        'allege "intentionally assaulted" Paul, causing him "minor injury". Boucher, 59, of Bowling '

        'Green was charged with one count of fourth-degree assault. As of Saturday afternoon, he '

        'was being held in the Warren County Regional Jail on a $5,000 bond.')



doc = nlp(text_example)
for x in doc.sents:

    print(str(x) + "\n")
pd.DataFrame([(token.text, token.ent_type_, token.idx, token.idx+len(token.text)) for token in doc]

             ,columns=['word','ent','start','end']).head(20)
# show lemmas simple example

doc_lemma = nlp('disappointed disappoint disappoints')



pd.DataFrame([(token.text, token.lemma_)  for token in doc_lemma],columns=['word','lemma'])
pd.DataFrame([(token.text, token.lemma_)  for token in doc]

             ,columns=['word','lemma']).head(10)
spacy.explain('PRON')


pd.DataFrame([(token.text,token.tag_,spacy.explain(token.tag_)) for token in doc], 

             columns=['word','part of speech','description']).head(20)
text_example
list(doc.noun_chunks)[:7]
text_example
# Let's find named entities in the text.

from spacy import displacy

displacy.render([doc], jupyter=True, style='ent')
print ("GPE -> '{}'".format(spacy.explain('GPE')))
# show Tree parsing

from spacy import displacy

doc1 = nlp("""The call center agent was great""")

doc6 = nlp("I don't think he was able to answer my question")

displacy.render([doc6], jupyter=True, style='dep')
pd.DataFrame([['acomp',spacy.explain('acomp')],['nsubj',spacy.explain('nsubj')],['det',spacy.explain('det')]],

             columns=['abbreviation','description'])
cat = nlp("cat")

dog = nlp("dog")

ani = nlp("animal")

car = nlp("car")





print(cat.similarity(dog))

print(cat.similarity(ani))

print(cat.similarity(car))



dog.vector