import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm



import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag



from nltk.chunk import conlltags2tree, tree2conlltags

from pprint import pprint

from nltk.chunk import ne_chunk



from bs4 import BeautifulSoup

import requests

import re



import warnings

warnings.filterwarnings("ignore")
ex = "European authorities fined Google a record $5.1 billion on Wednesday"\

" for abusing its power in the mobile phone market and ordered the company to alter its practices"
def preprocess(sent):

    sent = nltk.word_tokenize(sent)

    sent = nltk.pos_tag(sent)

    return sent



sent = preprocess(ex)

sent
pattern = 'NP: {<DT>?<JJ>*<NN>}'

cp      = nltk.RegexpParser(pattern)

cs       = cp.parse(sent)



print(cs)
iob_tagged = tree2conlltags(cs)

pprint(iob_tagged)
ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))

print(ne_tree)
nlp = en_core_web_sm.load()
doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')

pprint([(X.text, X.label_) for X in doc.ents])
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
def url_to_string(url):

    res = requests.get(url)

    html = res.text

    soup = BeautifulSoup(html, 'html5lib')

    for script in soup(["script", "style", 'aside']):

        script.extract()

    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
url = "https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html"\

"?hp&action=click&pgtype=Homepage&clickSource=story-heading&module="\

"first-column-region&region=top-news&WT.nav=top-news"



ny_bb   = url_to_string(url)
f = open("../input/kernel.txt","r",encoding='UTF-8',errors='ignore')

ny_bb = f.read()
article = nlp(ny_bb)

len(article.ents)
labels = [x.label_ for x in article.ents]

Counter(labels)
items = [x.text for x in article.ents]

Counter(items).most_common(3)
sentences = [x for x in article.sents]

print(sentences[20])
displacy.render(nlp(str(sentences[20])), jupyter=True, style='ent')
displacy.render(nlp(str(sentences[20])), style='dep', jupyter = True, options = {'distance': 120})
[(x.orth_,x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[20])) 

                                      if not y.is_stop and y.pos_ != 'PUNCT']]
dict([(str(x), x.label_) for x in nlp(str(sentences[20])).ents])