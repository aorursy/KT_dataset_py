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
convo1text = open("../input/conversation1.txt", "r").readlines()
convo1text
newConvo = []

for sent in convo1text:

    sent = sent.strip("\n")

    if not sent is '':

        newConvo.append(sent)

    

convo1text = newConvo

print(convo1text)
for index,sent in enumerate(convo1text):

    if sent.count(':') > 0:

        previousSpeaker = sent.split(':')[0]

    if sent.count(':') == 0:

        convo1text[index] = previousSpeaker + ':' + sent

        

convo1text



conversation1 = []

for sent in convo1text:

    convo = sent.split(':')

    if len(convo) > 2:

        convo[1] += ('-').join(convo[2:])

    convo = convo[:2]

    convo[1] = convo[1].strip()

    if not convo[1] is '':

        conversation1.append([c for c in convo])
conversation1
import json

with open("../input/appos.json", "r") as read_file:

    data = json.load(read_file)

    

appos = data['appos']

appos
import re

processedConversation = []

for speaker, dialog in conversation1:

    for (key, val) in appos.items():

        dialog = dialog.lower()

        dialog = dialog.replace(key, val)

        dialog = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", dialog)

        dialog = ' ' + dialog + ' '

        dialog = re.sub(' +', ' ',dialog)

        

    processedConversation.append([speaker, dialog])

        

processedConversation
with open("../input/thirdperson.json", "r") as read_file:

    data = json.load(read_file)

thirdperson = {}

for (key, value) in data['thirdperson'].items():

    thirdperson[' ' + key + ' '] = ' ' + value + ' '





    

thirdperson
finalConversation = []

previousSpeaker = ''

for speaker, dialog in processedConversation:

    for (key, val) in thirdperson.items():

        dialog = dialog.lower()

        dialog = dialog.replace(key, val)

        dialog = dialog.replace(' i ', ' ' +speaker + ' ')

        dialog = dialog.replace('i ', ' ' +speaker + ' ')

        dialog = dialog.replace('you', previousSpeaker if not previousSpeaker is '' else 'them' )

    finalConversation.append([speaker, dialog])

    

finalConversation
import spacy
nlp = spacy.load("en_core_web_sm")
summary = ''

summarySpeaker = None

for speaker, dialog in finalConversation:

    if summarySpeaker is None:

        doc = nlp(dialog)

        for token in doc:

            if token.lemma_ == 'summarize':

                # this is the part where we hear someone summarize

                # point following this must be a part of the summarization

                summarySpeaker = speaker

    else:

        if speaker == summarySpeaker:

            summary = summary + dialog

        else:

            summarySpeaker = None

        

summary

            
doc = nlp(summary)
spacy.displacy.render(doc, style='ent',jupyter=True)
from nltk.corpus import stopwords

from string import punctuation
_stopwords = set(stopwords.words('english') + list(punctuation) + list(' '))

_stopwords
word_sent = [word.text for word in doc if word.text not in _stopwords]

word_sent
from nltk.probability import FreqDist
main_sent = []

for speaker, dialog in finalConversation:

    dialogdoc = nlp(dialog)

    for word in dialogdoc:

        if word.text not in _stopwords:

            main_sent.append(word.text)
fd = {word:freq for word,freq in FreqDist(main_sent).items() if word in word_sent}
from heapq import nlargest
imp_words = nlargest(10, fd, key=fd.get)
from collections import defaultdict
sents = []

for speaker, dialog in finalConversation:

    sents.append(dialog)
ranking = defaultdict(int)

for i, sent in enumerate(sents):

    doc = nlp(sent)

    for w in doc:

        if w.text in freq: 

            ranking[i] += freq[w.text]
ranking
sent_idx = nlargest(5, ranking, key=ranking.get)

sent_idx
[sents[i] for i in sent_idx]