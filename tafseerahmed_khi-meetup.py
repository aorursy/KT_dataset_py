docs = ["Kaggle provides notebooks for python.",

       "Python is an easy language.",

       "Kaggle provides many datasets."]
import spacy

# Creting List of Stop Words

from spacy.lang.en.stop_words import STOP_WORDS

stop_words = spacy.lang.en.stop_words.STOP_WORDS



# Creating list of punctuation marks

import string

punctuations = string.punctuation



print(stop_words)

print("\n===================\n")

print(punctuations)
nlp = spacy.load('en_core_web_sm')

prc_docs = [nlp(doc) for doc in docs]

print(prc_docs)

token_docs = [ [tok.lemma_.lower().strip() for tok in prc_doc] for prc_doc in prc_docs]

print("Before: ", prc_docs)

print("\nAfter: ", token_docs)
token_docs = [ [tok for tok in token_doc if (tok not in stop_words and tok not in punctuations)] for token_doc in token_docs] 

print("Before: ", token_docs)

print("\nAfter: ",token_docs)
s = ''

docs = []

for token_doc in token_docs: 

    for token in token_doc:

        s += token + ' '

    docs.append(s)

    s = ''



print(docs)
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names())

print(X.todense())
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names())



print(X.todense())
import gensim

import nltk



reader = nltk.corpus.PlaintextCorpusReader("/kaggle/input/",'.*\.txt')

text = reader.raw()

print(reader.fileids())
import re



sentences = re.split(r'[۔؟]',text)



print(len(sentences))

print(sentences[6])

from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()





s = []

for i in range(len(sentences)):

    s.append(word_punct_tokenizer.tokenize(sentences[i]))



print(s[6])
model = gensim.models.Word2Vec(s, min_count=5, size = 20)

model.wv.most_similar('سورج')
model.wv.most_similar('فجر')
model.wv.doesnt_match(["رمضان" ,"شوال" ,"رجب" ,"جمعہ"])
