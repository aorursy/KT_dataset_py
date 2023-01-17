# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from IPython.core.display import display, HTML

from IPython.display import Image

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

from spacy import displacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import normalize





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nlp = spacy.load('en')



text="This is one of the greatest films ever made. Brilliant acting by George C. Scott and Diane Riggs. This movie is both disturbing and extremely deep. Don't be fooled into believing this is just a comedy. It is a brilliant satire about the medical profession. It is not a pretty picture. Healthy patients are killed by incompetent surgeons, who spend all their time making money outside the hospital. And yet, you really believe that this is a hospital. The producers were very careful to include real medical terminology and real medical cases. This movie really reveals how difficult in is to run a hospital, and how badly things already were in 1971. I loved this movie."

print(text)
doc = nlp(text)#, disable = ['parser', 'tagger', 'ner'])



print(dir(doc))
print([token.text for token in nlp("let's go to N.Y.!")])

print([token.text for token in nlp("Some\nspaces and\ttab characters")])
from spacy.lang.en.stop_words import STOP_WORDS



print(f'{list(STOP_WORDS)[0:10]}')
#word_list = ['feet', 'foot', 'foots', 'footing']

word_list=['organize', 'organizes', 'organizing']



from nltk import stem

wnl = stem.WordNetLemmatizer()

porter = stem.porter.PorterStemmer()



print([porter.stem(word) for word in word_list])

print([wnl.lemmatize(word) for word in word_list])
#Lemmatiztion using spacy

for token in nlp(" ".join(word_list)):

    print(token.text, token.lemma_)
print(doc.is_tagged)
# print column headers

print('{:15} | {:15} | {:8} | {:8} | {:11} | {:8} | {:8} | {:8} | '.format(

    'TEXT','LEMMA_','POS_','TAG_','DEP_','SHAPE_','IS_ALPHA','IS_STOP'))



# print various SpaCy POS attributes

for token in doc:

    print('{:15} | {:15} | {:8} | {:8} | {:11} | {:8} | {:8} | {:8} |'.format(

          token.text, token.lemma_, token.pos_, token.tag_, token.dep_

        , token.shape_, token.is_alpha, token.is_stop))
# Explanations of tags

print(spacy.explain('JJ'), spacy.explain('IN'), spacy.explain('VBZ'), spacy.explain('DT'))
previous_token = doc[0]  # set first token



for token in doc[1:]:    

    # identify adjective noun pairs

    if previous_token.pos_ == 'ADJ' and token.pos_ == 'NOUN':

        print(f'{previous_token.text}_{token.text}')

    

    previous_token = token
for token in doc[0:20]:

    print(f'{token.text}_{token.pos_}')
ner_text = "When I told John that I wanted to move to Alaska, he warned me that I'd have trouble finding a Starbucks there."

ner_doc = nlp(ner_text)
print('{:10} | {:15}'.format('LABEL','ENTITY'))



for ent in ner_doc.ents[0:20]:

    print('{:10} | {:50}'.format(ent.label_, ent.text))
print(spacy.explain('GPE'))



displacy.render(docs=ner_doc, style='ent', jupyter=True)
displacy.render(docs=doc, style='ent', jupyter=True)
sentence = "This is a sentence. This is another sentence. let's go to N.Y.!"

print(sentence.split('.'))



doc = nlp(sentence)

for sent in doc.sents:

    print(sent.text)
tokens = ['i', 'want', 'to', 'go', 'to', 'school'] # "i_want", want_to



def ngrams(tokens, n):

    length = len(tokens)

    grams = []

    for i in range(length - n + 1):

        grams.append("_".join(tokens[i:i+n]))

    return grams

print(ngrams(tokens, 3)) # Bigram = 2, trigram = 3
from collections import defaultdict,Counter



d = defaultdict(int)



d['reveals'] = 1

print(d)
# create an instance of the CountVectorizer

vect = CountVectorizer()

print(vect)
text=['This is good',

     'This is bad',

     'This is awesome']

vect.fit(text)
print(vect.get_feature_names())
pd.DataFrame(vect.transform(text).toarray(), columns=['awesome', 'bad', 'good', 'is', 'this'])[ ['this','is','good','bad','awesome']]
example_text = ['again we observe a document'

               , 'the second time we have see this text']



vect = CountVectorizer()

vect.fit_transform(text).toarray()
text = ['This is the first document'

        , 'This is the second second document'

        , 'And the third one'

        , 'Is it the first document again']

vect = CountVectorizer(lowercase = False)

vect.fit(text)

print(vect.get_feature_names())
from sklearn.feature_extraction.text import TfidfVectorizer

text