import numpy as np
import os
from random import shuffle
import re
import zipfile
import lxml.etree
#download the data
#urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", "ted_en-20160408")
# extract subtitle
#with zipfile.ZipFile('../input/ted_en-20160408/', 'r') as z:
doc = lxml.etree.parse(open('../input/ted_en-20160408/ted_en-20160408.xml', 'r'))
input_text = '\n'.join(doc.xpath('//content/text()'))
# remove parenthesis 
input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
# store as list of sentences
sentences_strings_ted = []
for line in input_text_noparens.split('\n'):
    m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
    sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
# store as list of lists of words
sentences_ted = []
for sent_str in sentences_strings_ted:
    tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
    sentences_ted.append(tokens)
#import the WORD2VEC model
from gensim.models import Word2Vec
model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)
#extract the most similar words to the word 'man'
model_ted.wv.most_similar('man')
#extract the most similar words to the word 'they'
model_ted.wv.most_similar('they')
