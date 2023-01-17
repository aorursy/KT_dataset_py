import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk.chunk import ne_chunk
#Example Sentence



ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
#Apply word tokenization and part-of-speech tagging to the sentence.

def preprocess(sent):

    sent = nltk.word_tokenize(sent)

    sent = nltk.pos_tag(sent)

    return sent
#Sentence filtered with Word Tokenization



sent = preprocess(ex)

print("POS_Tags for Sentence")

sent
#Chunking Pattern 



pattern = 'NP: {<DT>?<JJ>*<NN>}'
#create a chunk parser and test it on our sentence.

cp = nltk.RegexpParser(pattern)

cs = cp.parse(sent)

print(cs)
#Chunker formed for sentence



NPChunker = nltk.RegexpParser(pattern) 

result = NPChunker.parse(sent)

#result.draw()
import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()
doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')

print([(X.text, X.label_) for X in doc.ents])
print([(X, X.ent_iob_, X.ent_type_) for X in doc])
from bs4 import BeautifulSoup

import requests

import re
def url_to_string(url):

    res = requests.get(url)

    html = res.text

    soup = BeautifulSoup(html, 'html5lib')

    for script in soup(["script", "style", 'aside']):

        script.extract()

    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')

article = nlp(ny_bb)

print(article.ents)

len(article.ents)
labels = [x.label_ for x in article.ents]

Counter(labels)
#most common used items



items = [x.text for x in article.ents]

Counter(items).most_common(3)
sentences = [x for x in article.sents]

print(sentences[20])
#Displaying detecing the name entities, Marking down.



displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')