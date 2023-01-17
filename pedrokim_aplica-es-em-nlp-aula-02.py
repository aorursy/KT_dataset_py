import nltk

nltk.download("gutenberg")
hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
print(hamlet_raw[:1000])
from nltk.tokenize import sent_tokenize

sentences = sent_tokenize(hamlet_raw)

print(sentences[:10])

from nltk.tokenize import word_tokenize

words = word_tokenize(sentences[0])

print(words)
from nltk.corpus import stopwords

stopwords_list = stopwords.words('english')

print(stopwords_list)

#stopwords_list = stopwords.words('portuguese')

#print(stopwords_list)
non_stopwords = [w for w in words if not w.lower() in stopwords_list]
print(non_stopwords)
import string
punctuation = string.punctuation
print(punctuation)
non_punctuation = [w for w in non_stopwords if not w in punctuation]


print(non_punctuation)
from nltk import pos_tag

pos_tags = pos_tag(words)

print(pos_tags)
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')

sample_sentence = "He has already gone"
sample_words = word_tokenize(sample_sentence)

stems = [stemmer.stem(w) for w in sample_words]

print(stems)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

pos_tags = nltk.pos_tag(sample_words)

lemmas = []
for w in pos_tags:
    if w[1].startswith('J'):
        pos_tag = wordnet.ADJ
    elif w[1].startswith('V'):
        pos_tag = wordnet.VERB
    elif w[1].startswith('N'):
        pos_tag = wordnet.NOUN
    elif w[1].startswith('R'):
        pos_tag = wordnet.ADV
    else:
        pos_tag = wordnet.NOUN
        
    lemmas.append(lemmatizer.lemmatize(w[0], pos_tag))
    
print(lemmas)
from nltk import word_tokenize

frase = 'o cachorro correu atrás do gato'


ngrams = ["%s %s %s" % (nltk.word_tokenize(frase)[i], \
                      nltk.word_tokenize(frase)[i+1], \
                      nltk.word_tokenize(frase)[i+2]) \
          for i in range(len(nltk.word_tokenize(frase))-2)]

print(ngrams)

non_punctuation = [w for w in words if not w.lower() in punctuation]

n_grams_3 = ["%s %s %s"%(non_punctuation[i], non_punctuation[i+1], non_punctuation[i+2]) for i in range(0, len(non_punctuation)-2)]

print(n_grams_3)
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(ngram_range=(2,3))

import numpy as np

sent2 = ['o cachorro correu atrás do gato que correu atrás do rato',
        'o gato correu atrás do rato',
        'o rato comeu a ração',
        'o cachorro comeu o rato que comeu a ração',
        'o gato comeu o rato']

arr = np.array(sent2)

print(arr)

n_gram_counts = count_vect.fit_transform(arr)

print(n_gram_counts.toarray())

print(count_vect.vocabulary_)
arr = np.array(sentences)

n_gram_counts = count_vect.fit_transform(arr)

print(n_gram_counts.toarray()[:20])

print([k for k in count_vect.vocabulary_.keys()][:20])
from nltk.corpus import reuters
articles = reuters.fileids()
reuters_list = ""
for name in articles:
    reuters_list =  reuters_list + nltk.corpus.reuters.raw( name ) 
reuters_list

reuters = nltk.corpus.reuters.raw('test/14826')

sentences = sent_tokenize(reuters)

words = word_tokenize(reuters_list)
#words = word_tokenize(sentences[0])
#words = word_tokenize(reuters)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

pos_tags = nltk.pos_tag(words)

lemmas = []
for w in pos_tags:
    if w[1].startswith('J'):
        pos_tag = wordnet.ADJ
    elif w[1].startswith('V'):
        pos_tag = wordnet.VERB
    elif w[1].startswith('N'):
        pos_tag = wordnet.NOUN
    elif w[1].startswith('R'):
        pos_tag = wordnet.ADV
    else:
        pos_tag = wordnet.NOUN
        
    lemmas.append(lemmatizer.lemmatize(w[0], pos_tag))
    
#print(lemmas)
### Remover stop words e pontuações
from nltk.corpus import stopwords
import string

stopwords_list = stopwords.words('english')
punctuation = string.punctuation

lemmas_non_stopwords = [w for w in lemmas if not w.lower() in stopwords_list]
lemmas_non_stopwords_punctuation = [w for w in lemmas_non_stopwords if not w in punctuation]

word_list = {}

for key in lemmas_non_stopwords_punctuation:

    # python check if key in dict using "in"
    if key in word_list:
        #print(f"Yes, key: '{key}' exists in dictionary")]
        word_list[key] = word_list[key]+1
    else:
        #print(f"No, key: '{key}' does not exists in dictionary")
        word_list[key] = 1
word_list

sort_words = sorted(word_list.items(), key=lambda x: x[1], reverse=True)

for i in sort_words:
    print(i[0], i[1])
