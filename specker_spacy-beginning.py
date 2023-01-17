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
import spacy
nlp = spacy.load("en")
doc = nlp("Let's go to N.Y.!")
doc.text.split()
[token.orth_ for token in doc]
[(token, token.orth_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha) for token in doc]
#when orth or any method is used without underscore then the hash value is shown because spacy does not 
#store the words but stores the hashes of the words. except methods starting with "is"
[(token,token.is_stop) for token in doc]
#stop words are common words that dont have any influence over the context.
prac = "practice practicing practiced"
nlp_p = nlp(prac)
[word.lemma_ for word in nlp_p]
#lemma brings down the word to its root word
doc
print([token for token in doc[5].lefts])
doc
nlp = spacy.load('en_core_web_sm')
doc = nlp(u"Credit and mortgage account holders must submit their requests")

root = [token for token in doc if token.head == token][0]
subject = list(root.lefts)[0]
for descendant in subject.subtree:
    assert subject is descendant or subject.is_ancestor(descendant)
    print(descendant.text, descendant.dep_, descendant.n_lefts,
          descendant.n_rights,
          [ancestor.text for ancestor in descendant.ancestors])
from spacy import displacy
displacy.render(doc, style='dep', jupyter=True)
#Visualising the dependencies (can also pass a list of docs in .render())
#Disabling the parser will make spaCy load and run much faster.
#If you want to load the parser, but need to disable it for specific documents, 
#you can also control its use on the nlp object.
#nlp = spacy.load('en', disable=['parser'])
#doc = nlp("I don't want to be parsed", disable=['parser'])
doc
#named entities are available as .ents property of the doc.
for ent in doc.ents:
    print(ent, ent.start_char, ent.end_char, ent.label_)
#does'nt prints anything because does not recognize the entities or maybe the sentence doesn't have any.
from spacy.tokens import Span

nlp = spacy.load('en_core_web_sm')
doc = nlp(u"FB is hiring a new Vice President of global policy")
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('Before', ents)
# the model didn't recognise "FB" as an entity :(

ORG = doc.vocab.strings[u'ORG']  # get hash value of entity label
fb_ent = Span(doc, 0, 1, label=ORG) # create a Span for the new entity
doc.ents = list(doc.ents) + [fb_ent]

ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print('After', ents)
from spacy.attrs import ENT_IOB, ENT_TYPE

nlp = spacy.load('en_core_web_sm')
doc = nlp.make_doc(u'London is a big city in the United Kingdom.')
print('Before', list(doc.ents))  # []

header = [ENT_IOB, ENT_TYPE]
attr_array = np.zeros((len(doc), len(header)))
attr_array[0, 0] = 3  # B
attr_array[0, 1] = doc.vocab.strings[u'GPE']
doc.from_array(header, attr_array)
print('After', list(doc.ents))  # [London]
#In this case, "FB" is token (0, 1) – but at the document level, the entity will have the start and end indices (0, 2).
from spacy.lang.en import English
from spacy.pipeline import SentenceSegmenter

def split_on_newlines(doc):
    start = 0
    seen_newline = False
    for word in doc:
        if seen_newline and not word.is_space:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text == '\n':
            seen_newline = True
        if start < len(doc):
            yield doc[start:len(doc)]

nlp = English()
sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
nlp.add_pipe(sbd)
doc = nlp(u"This is a sentence.\nThis is another one.\nand more")
for sent in doc.sents:
    print([token.text for token in sent])
