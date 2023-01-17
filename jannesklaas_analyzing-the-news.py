# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
a1 = pd.read_csv('../input/articles1.csv',index_col=0)
a2 = pd.read_csv('../input/articles2.csv',index_col=0)
a3 = pd.read_csv('../input/articles3.csv',index_col=0)
df = pd.concat([a1,a2,a3])
# save memory
del a1, a2, a3
df.head()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
df.publication.value_counts().plot(kind='bar')
doc = df.loc[0,'content']
import spacy
from spacy import displacy
nlp = spacy.load('en')
doc = nlp(doc)
displacy.render(doc,style='ent',jupyter=True)
from tqdm import tqdm, tqdm_notebook
nlp = spacy.load('en',
                 disable=['parser', 
                          'tagger',
                          'textcat'])
frames = []
for i in tqdm_notebook(range(1000)):
    doc = df.loc[i,'content']
    text_id = df.loc[i,'id']
    doc = nlp(doc)
    ents = [(e.text, e.start_char, e.end_char, e.label_) 
            for e in doc.ents 
            if len(e.text.strip(' -—')) > 0]
    frame = pd.DataFrame(ents)
    frame['id'] = text_id
    frames.append(frame)
npf = pd.concat(frames)
npf.head()
npf.columns = ['Text','Start','Stop','Type','id']
plt.figure(figsize=(10,7))
npf.Type.value_counts().plot(kind='bar')
orgs = npf[npf.Type == 'ORG']
plt.figure(figsize=(10,7))
orgs.Text.value_counts()[:15].plot(kind='bar')
orgs.groupby(['id','Text']).size()
nlp = spacy.load('en')
doc = 'Google to buy Apple'
doc = nlp(doc)
displacy.render(doc,style='dep',jupyter=True, options={'distance':120})
for chunk in doc.noun_chunks:
    print(chunk.text,'|' , chunk.root.text,'|', chunk.root.dep_,'|',
          chunk.root.head.text)
for token in doc:
    print(token.text,'|', token.lemma_,'|', token.pos_,'|', token.tag_,'|', token.dep_,'|',
          token.shape_,'|', token.is_alpha,'|', token.is_stop)
import spacy
import random
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]
nlp = spacy.load('en')
# create the built-in pipeline components and add them to the pipeline
# nlp.create_pipe works for built-ins that are registered with spaCy
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
# otherwise, get it so we can add labels
else:
    ner = nlp.get_pipe('ner')
# add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

n_iter = 5

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp._optimizer 
    if not nlp._optimizer:
        optimizer = nlp.begin_training()
    
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)
# test the trained model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en')
matcher = Matcher(nlp.vocab)
pattern = [{'LOWER': 'hello'}, {'IS_PUNCT': True}, {'LOWER': 'world'}]


matcher.add('HelloWorld', None, pattern)

doc = nlp(u'Hello, world! Hello world!')
matches = matcher(doc)
matches
doc[0:3]

df.title = df.title.fillna('')
np.where(df.content.str.contains('iPhone'))
df.loc[14]
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en')
matcher = Matcher(nlp.vocab)
# Get the hash of the word 'PRODUCT'. This is required to set an entity.
PRODUCT = nlp.vocab.strings['PRODUCT']

def add_product_ent(matcher, doc, i, matches):
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)
    match_id, start, end = matches[i]
    doc.ents += ((PRODUCT, start, end),)

pattern1 = [{'LOWER': 'iphone'}]
pattern2 = [{'ORTH': 'iPhone'}, {'IS_DIGIT': True}]

matcher.add('iPhone', add_product_ent,pattern1, pattern2)
matches = matcher(doc)
displacy.render(doc,style='ent',jupyter=True)
def matcher_component(doc):
    matches = matcher(doc)
    return doc
nlp.add_pipe(matcher_component,last=True)
doc = nlp(df.content.iloc[14])
#matcher(doc)
displacy.render(doc,style='ent',jupyter=True)
import re
pattern = 'NL[0-9]{9}B[0-9]{2}'
my_string = 'ING Bank N.V. BTW:NL003028112B01'
re.findall(pattern,my_string)
match = re.search(pattern,my_string)
match.span()
