import nltk

import string

from sklearn.feature_extraction.text import TfidfVectorizer

#nltk.download('punkt') to download the tokenizer, only if required!



import gensim # for bag of words model

from nltk.tokenize import word_tokenize

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import spacy

import Levenshtein
text1 = "Browns 24, Bengals 3: Cleveland dominates from the start - Columbus Dispatch"

text2 = "Browns dominate Bengals 24-3 through 3 quarters - SFGate" 

text3 = "Browns take down the Bengals in prime time 24-3 - 19 Action News "

text4 = "Browns Move Into 1st With 24-3 Win Over Bengals - ABC News"

text5 = "Man charged in Philly, Virginia abductions - Hickory Daily Record"

text6 = "Man charged in Philly, Virginia abductions has history of violence - Fox News"

text7 = "AC/DC drummer Phil Rudd murder plot charge dropped - BBC News"

text8 = "AC/DC's Phil Rudd: Charge to procure murder laid against drummer withdrawn ... - ABC Online"

text9 = "Charge dropped against AC/DC drummer Phil Rudd"

text10 = "AC/DC drummer Phil Rudd 'attempted to procure murders"



combined_text = (text1, text2, text3, text4, text5, text6, text7, text8, text9, text10) 
combined_text # corpus
combined_text
def stem_tokens(tokens):

    return [stemmer.stem(item) for item in tokens]



def normalize(text):

    return stem_tokens(word_tokenize(text.lower().translate(punctuation_map)))



punctuation_map = dict((ord(char), None) for char in string.punctuation)

stemmer = nltk.stem.porter.PorterStemmer()

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')



def similarity_tfidf(text1, text2):

    print(text1)

    print(text2)

    tfidf = vectorizer.fit_transform([text1, text2])

    score = ((tfidf * tfidf.T).A)[0,1]

    return score
similarity_tfidf(text1, text2)
similarity_tfidf(text1, text3)
similarity_tfidf(text1, text4)
similarity_tfidf(text1, text6)
similarity_tfidf(text5, text7)
similarity_tfidf(text8, text9)
similarity_tfidf(text8, text10)
similarity_tfidf(text9, text10)
tfidf = vectorizer.fit_transform(combined_text)

print(cosine_similarity(tfidf[9], tfidf)[0][0])

combined_text
vectorizer_no_preprocessing = TfidfVectorizer()

tfidf_no_preprocessing = vectorizer_no_preprocessing.fit_transform(combined_text)

print(cosine_similarity(tfidf_no_preprocessing[9], tfidf_no_preprocessing)[0])

combined_text
# nlp = spacy.load('en')

# doc1 = nlp(text1)

# doc2 = nlp(text2)

# doc3 = nlp(text3)

# doc4 = nlp(text4)



# print (doc1.similarity(doc2))

# print (doc1.similarity(doc3))

# print (doc1.similarity(doc4))
for text in combined_text:

    print(Levenshtein.ratio(text10, text))

combined_text
import difflib

for text in combined_text:

    print(difflib.SequenceMatcher(None, text10, text).ratio())

combined_text
gen_docs = [[w.lower() for w in word_tokenize(text)] 

            for text in combined_text]

print(gen_docs)
dictionary = gensim.corpora.Dictionary(gen_docs)

print(dictionary.token2id['browns'])

print("Number of words in dictionary:",len(dictionary))

for i in range(len(dictionary)):

    print(i, dictionary[i])
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

corpus
tf_idf = gensim.models.TfidfModel(corpus)

print(tf_idf)

s = 0

for i in corpus:

    s += len(i)

print(s)
sims = gensim.similarities.Similarity('/',tf_idf[corpus],

                                      num_features=len(dictionary))

print(sims)

print(type(sims))
query_doc = [w.lower() for w in word_tokenize("Browns 24, Bengals 3: ominates from the start - Columbus Dispatch")]

print(query_doc)

query_doc_bow = dictionary.doc2bow(query_doc)

print(query_doc_bow)

query_doc_tf_idf = tf_idf[query_doc_bow]

query_doc_tf_idf
# sims[query_doc_tf_idf]