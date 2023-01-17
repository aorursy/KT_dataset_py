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
from datetime import date, time, datetime 
import pytz
IST = pytz.timezone('Asia/Kolkata')
IST_time = datetime.now(IST)
print("Current Date Time (IST): ",IST_time.strftime('%Y-%B-%d  %H:%M %p'))
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
df_orig = df
df.columns
sentence = df.iloc[3].abstract
sentence
for i in range(5):
    print(df.iloc[i],"\n\n")
import jsonschema
f = open('/kaggle/input/CORD-19-research-challenge/json_schema.txt',"r")
schema = f.read()
print(schema)
# Python program to read 
# json file 


import json 

# Opening JSON file 
f = open('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/7db22f7f81977109d493a0edf8ed75562648e839.json',) 

# returns JSON object as 
# a dictionary 
data = json.load(f) 

# Iterating through the json 
# list 
for i in data['body_text']: 
	print(i) 

# Closing file 
f.close() 

#Take an example abstract for record 3
token_sequence = str.split(sentence)
vocab = sorted(set(token_sequence))
token_sequence
vocab
', '.join(vocab)
num_tokens = len(token_sequence)
vocab_size = len(vocab)
onehot_vectors = np.zeros((num_tokens,vocab_size),int)
onehot_vectors
for i,word in enumerate(token_sequence):
    onehot_vectors[i,vocab.index(word)] = 1
' '.join(vocab)
for i in onehot_vectors:
    print(i,"\n\n")
df = pd.DataFrame(onehot_vectors,columns=vocab)
df[df == 0] = ''
df
df.describe()
token_sequence
#Sentence Bag of Words

sentence_bow = {}

for token in sentence.split():
    sentence_bow[token] = 1

sentence_bow
type(sentence_bow)
sorted(sentence_bow.items())
#sparse versus dense bow
df1 = pd.DataFrame(pd.Series(dict([(token,1) for token in sentence.split()])), columns=['sent']).T
df1
sentence2 = df_orig.iloc[11].abstract
sentence2 += df_orig.iloc[15].abstract
sentence2 += df_orig.iloc[16].abstract
sentence2
sentence += df_orig.iloc[4].abstract
sentence += df_orig.iloc[5].abstract
sentence += df_orig.iloc[6].abstract
sentence += df_orig.iloc[7].abstract
sentence += df_orig.iloc[8].abstract

sentence
corpus = {} # define corpus as an empty dictionary

for i,sent in enumerate(sentence.split('.')):
    #print('sent{}'.format(i))
    corpus['sent{}'.format(i)] = dict((tok,1) for tok in sent.split())
df_multi = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
df_multi
df_multi[df_multi.columns[:10]]
#overlap of word counts for two bag of words vectors
df_multi = df_multi.T
df_multi.sent0.dot(df_multi.sent1)
df_multi.sent4.dot(df_multi.sent9)
[(k, v) for (k, v) in (df_multi.sent0 & df_multi.sent1).items() if v]
import re
tokens = re.split(r'[-\s.,;!?]+', sentence)
tokens
pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
tokens
[x for x in tokens if x and x not in '- \t\n.,;!?']
!pip install regex
# Now we use spaCy and NLTK and Stanford CoreNLP
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
tokenizer.tokenize(sentence)
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
tokenizer.tokenize(sentence)
pattern = re.compile(r"([-\s.,;!?])+")
tokens = pattern.split(sentence)
tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']
tokens
# we use 2-gram
from nltk.util import ngrams
list(ngrams(tokens,2))
list(ngrams(tokens,3))
two_grams = list(ngrams(tokens,2))
[" ".join(x) for x in two_grams]
stop_words = ['a', 'an', 'the', 'on', 'of', 'off', 'this', 'is']
tokens_without_stopwords = [x for x in tokens if x not in stop_words]
tokens_without_stopwords
#using stopwords from nltk
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
len(stop_words)
stop_words
# single length stopwords
[sw for sw in stop_words if len(sw) ==1]
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
len(sklearn_stop_words)
len(set().union(stop_words,sklearn_stop_words))
len(set().intersection(set(stop_words),set(sklearn_stop_words)))
print(sklearn_stop_words,"\n",stop_words)
# sets cannot contain duplicate members
set_1 = set(stop_words)
set_2 = set(sklearn_stop_words)
len(set_1 | set_2) #union

len(set_1 & set_2) # intersection 
#case folding
normalized_tokens = [x.lower() for x in tokens]
normalized_tokens
# stemming 
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(nw).strip("'") for nw in normalized_tokens] # we lose important information like disease lupus is incorrectly changed to lupu
stemmed_tokens
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("aged","n")

lemmatized_tokens_n = [lemmatizer.lemmatize(token,pos="n") for token in normalized_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(token,pos="a") for token in lemmatized_tokens_n]
lemmatized_tokens
stemmed_tokens = [stemmer.stem(nw).strip("'") for nw in lemmatized_tokens]
stemmed_tokens # stemming loses some vital information. So with our future analysis, we use only the lemmatized tokens.
#TF-IDF
from nltk.tokenize import TreebankWordTokenizer
sentence
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(sentence.lower())
tokens
from collections import Counter
bag_of_words = Counter(normalized_tokens)
bag_of_words
bag_of_words.most_common(100)
#Specifically, the number of times a word occurs in a given document is called the term frequency, commonly abbreviated TF. In some examples you may see the count of word occurrences normalized (divided) by the number of terms in the document.
# calculate TF
times_infection_appears = bag_of_words.get('infection')
number_unique_words = len(bag_of_words)
normalized_tf = times_infection_appears/number_unique_words
round(normalized_tf,4)
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
tokens = [token for token in normalized_tokens if token not in stopwords]
counts = Counter(tokens)
counts
# document vector
document_vector = []
doc_length = len(tokens)
for key, value in counts.most_common(20):
    print(key," ",value)
    document_vector.append(value/doc_length)
    
document_vector
#lexicon for this corpus containing three documents (3 different abstracts)
sentence2 = df_orig.iloc[11].abstract
sentence2 += df_orig.iloc[17].abstract
sentence2 += df_orig.iloc[19].abstract
sentence2
tokenizer = TreebankWordTokenizer()
doc2_tokens = sorted(tokenizer.tokenize(sentence2.lower()))
doc2_tokens = [x for x in tokens if x and x not in '- \t\n.,;!?']

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
doc2_tokens = [token for token in doc2_tokens if token not in stopwords]

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemmatized_tokens_n2 = [lemmatizer.lemmatize(token,pos="n") for token in doc2_tokens]
lemmatized_tokens2 = [lemmatizer.lemmatize(token,pos="a") for token in lemmatized_tokens_n2]
lemmatized_tokens2
counts2 = Counter(lemmatized_tokens)
counts2

len(lemmatized_tokens2)
lexicon = sorted(set(lemmatized_tokens2))
lexicon

from collections import OrderedDict
zero_vector = OrderedDict((token,0) for token in lexicon)
zero_vector

import copy
docs = [x for x in [df_orig.iloc[11].abstract,df_orig.iloc[17].abstract,df_orig.iloc[19].abstract]]
doc_vectors = []

for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key,value in token_counts.items():
        vec[key] = value/len(lexicon)
    doc_vectors.append(vec)
docs[0]
doc_vectors
# We choose five papers as follows for our further analysis
paper1 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/25621281691205eb015383cbac839182b838514f.json'
paper2 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/7db22f7f81977109d493a0edf8ed75562648e839.json'
paper3 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/a137eb51461b4a4ed3980aa5b9cb2f2c1cf0292a.json'
paper4 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/6c3e1a43f0e199876d4bd9ff787e1911fd5cfaa6.json'
paper5 = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/2ce201c2ba233a562ee605a9aa12d2719cfa2beb.json'

# Python program to read 
# json file 


import json 

# Opening JSON file 
f1 = open(paper1,) 
f2 = open(paper2,) 
f3 = open(paper3,) 
f4 = open(paper4,) 
f5 = open(paper5,) 


# returns JSON object as 
# a dictionary 
data1 = json.load(f1) 
data2 = json.load(f2) 
data3 = json.load(f3) 
data4 = json.load(f4) 
data5 = json.load(f5) 

# Iterating through the json 
# list 
for i in data1['body_text']: 
	print(i) 
print('\n\n\n ============ \n\n\n')
for i in data2['body_text']: 
	print(i) 
print('\n\n\n ============ \n\n\n')
for i in data3['body_text']: 
	print(i) 
print('\n\n\n ============ \n\n\n')
for i in data4['body_text']: 
	print(i) 
print('\n\n\n ============ \n\n\n')

for i in data5['body_text']: 
	print(i) 
    
    
    
    
    
# Closing file 
f1.close() 
f2.close() 
f3.close() 
f4.close() 
f5.close() 




print(data1.keys())
print(type(data1['body_text']))
data1['body_text'][0]
data1['body_text'][1]
# We convert list to string
d1 = ' '.join(map(str, data1['body_text'])) 
d2 = ' '.join(map(str, data2['body_text'])) 
d3 = ' '.join(map(str, data3['body_text'])) 
d4 = ' '.join(map(str, data4['body_text'])) 
d5 = ' '.join(map(str, data5['body_text'])) 
d1
#lexicon for this corpus containing three documents (3 different abstracts)
docs5 = d1 + d2 + d3 + d4 +d5

from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

docs5_tokens = sorted(tokenizer.tokenize(docs5.lower()))
docs5_tokens = [x for x in docs5_tokens if x and x not in '- \t\n.,;!?']

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
docs5_tokens = [token for token in docs5_tokens if token not in stopwords]

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemmatized_tokens_docs5a = [lemmatizer.lemmatize(token,pos="n") for token in docs5_tokens]
lemmatized_tokens_docs5b = [lemmatizer.lemmatize(token,pos="a") for token in lemmatized_tokens_docs5a]

from collections import Counter
countsdocs5 = Counter(lemmatized_tokens_docs5b)

len(lemmatized_tokens_docs5b)
lexicon5 = sorted(set(lemmatized_tokens_docs5b))
tokens5 = lemmatized_tokens_docs5b

tokens5
lexicon5
print('token size: ',len(tokens5))
print('lexicon size',len(lexicon5))
len(countsdocs5)
len(docs5_tokens)
# We use scispacy by allen.ai https://github.com/allenai/scispacy
!pip install scispacy
import spacy

!python -m scispacy download en_core_sci_lg
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz

import spacy

from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_sci_lg")
import spacy

from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_sci_lg")

# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
import scispacy
nlp = spacy.load("en_core_sci_lg")
doc = nlp("Alterations in the hypocretin receptor 2 and preprohypocretin genes produce narcolepsy in some animals.")
import spacy

from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_sci_sm")

# Add the abbreviation pipe to the spacy pipeline.
abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

doc = nlp("Spinal and bulbar muscular atrophy (SBMA) is an \
           inherited motor neuron disease caused by the expansion \
           of a polyglutamine tract within the androgen receptor (AR). \
           SBMA can be caused by this easily.")

print("Abbreviation", "\t", "Definition")
for abrv in doc._.abbreviations:
	print(f"{abrv} \t ({abrv.start}, {abrv.end}) {abrv._.long_form}")
