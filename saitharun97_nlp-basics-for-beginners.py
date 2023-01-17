# !pip install pandas

# !pip install numpy

!pip install nltk

# !pip install sklearn

# !pip install textblob

# !pip install --upgrade gensim

# !pip3 install gTTS
import pandas as pd

import numpy  as np



# library to clean data 

import re  

  

# Natural Language Tool Kit 

import nltk  

  

nltk.download('stopwords') 

  

# to remove stopword 

from nltk.corpus import stopwords 

  

# for Stemming purpose  

from nltk.stem.porter import PorterStemmer 



# for Lemmatization purpose

from nltk.stem import WordNetLemmatizer 

  

# Initialize empty array 

# to append clean text  

corpus = []  
# Lets say we have a text,here i have taken random text

#first step is to get all the sentences in the paragragh and from sentences get all the words



text_rand = """Good Morning Ladies and Gentleman – I hope this day finds you in the best of spirits.I warmly welcome you all here in today’s speech ceremony on the subject called Leadership. I, Vaishali Rawat, your host for today, will address this topic. Leadership is quite a word in itself and I can very much relate with it because I am myself holding a position of senior manager in my present company. If given a chance, everyone would want to lead and be followed. But has anyone ever realized how difficult this task is and what roles and responsibilities come with it?

First of all, please understand that leadership doesn’t involve domination or subjugation of the weaker sex. The world is already full with such people who have an ardent desire to rule and take charge of other peoples’ lives. But this is not the trait of a good leader.

The true leader is someone who earns respect through his rightful actions and mass following without any dictatorship. He inspires others to follow his footsteps and become the guiding light for the humanity. The great leader is someone who carries the torch of wisdom and enlightens the society thereby leading people to the path of progress and growth. Besides, the true meaning of leadership is having the requisite ability to enable people want to follow you while being under no compulsion as such to do so. Leaders are those people who set certain benchmarks and try to achieve those benchmarks by allowing people to judge them according to their actions and endeavors. The goals are set and all might is put towards achieving those goals, but without compromising with the ethics and morals – this is the true mark of a great leader.

Leaders who possess great leadership qualities effectively channelize their energy and devote themselves for the growth and progress of humanity. The restrictions or obligations that he/she imposes on himself/herself enable him/her to rise against all odds and never bow down to the circumstances. Always remember that the love of supreme excellence is found in a great leader. Thus, a true leader is someone who is able to establish a connection with the almighty and realizes by faith that he/she is a mere instrument in the hands of Him and dedicates his entire life to become an inspirer and guide of the higher sentiments and ambitions of the people.

He/she who is a leader in the true sense of the term has to pay the price for his forbearance and moral restraints. He/she does good to the society selflessly, i.e. without expecting anything in return. This leads to further enhancement or cleansing of his/her soul and keeping a check on his/her personal desires, which in turn allows him/her to become an extraordinary being.

There is an old-saying which is, “To be first in place, one must be first in merit as well.” Thus, an individual actually becomes a leader when he/she has the ability to lead mankind on the path of progress without any selfish reason."""
# Get all the sentences in a paragraph

sentences = nltk.sent_tokenize(text_rand)
sentences_df = pd.DataFrame(sentences)

sentences_df.head()
sentences_df.shape
# Get all the words in a sentence and remove the special characters using 're' library

words = nltk.word_tokenize(text_rand)
words
len(words)
# Before we look into stemming lets remove the stop words which are not using for NLP analysis

# Lets have a look at english stopwords



stopwords.words('english')
# Lets remove the stops words and apply stemming on remaining words

# creating PorterStemmer object to 

# take main stem of each word 



sentences = nltk.sent_tokenize(text_rand)

stemmer = PorterStemmer()



# Stemming

for i in range(len(sentences)):

    sentences[i] = re.sub('[^a-zA-Z]',' ',sentences[i])

    stem_word = nltk.word_tokenize(sentences[i].lower())

    stem_word = [stemmer.stem(word) for word in stem_word if word not in set(stopwords.words('english'))]

    sentences[i] = ' '.join(stem_word)  
sentences
sentences = nltk.sent_tokenize(text_rand)

lemmatizer = WordNetLemmatizer() 

# Stemming

for i in range(len(sentences)):

    sentences[i] = re.sub('[^a-zA-Z]',' ',sentences[i])

    lem_word = nltk.word_tokenize(sentences[i].lower())

    lem_word = [lemmatizer.lemmatize(word) for word in lem_word if word not in set(stopwords.words('english'))]

    sentences[i] = ' '.join(lem_word)  
sentences
ps = PorterStemmer()

wordnet=WordNetLemmatizer()

sentences = nltk.sent_tokenize(text_rand)

corpus = []

bow = []



# here you can try both stemming and lemmatization and see the difference

for i in range(len(sentences)):

    sent = re.sub('[^a-zA-Z]', ' ', sentences[i])

    sent = sent.lower()

    sent = sent.split()

#     sent = [ps.stem(word) for word in sent if not word in set(stopwords.words('english'))]

    sent = [wordnet.lemmatize(word) for word in sent if not word in set(stopwords.words('english'))]

    bow.append(sent)

    sent = ' '.join(sent)

    corpus.append(sent)

    

# Creating the Bag of Words model, you can play around with the max_features

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1000)

BoW_count = cv.fit_transform(corpus).toarray()
corpus
flat_list = [word for sub_word in bow for word in sub_word]
flat_list
# Number of features created for BagOfWords

len(flat_list)
X_df = pd.DataFrame(BoW_count)

X_df
ps = PorterStemmer()

wordnet=WordNetLemmatizer()

sentences = nltk.sent_tokenize(text_rand)

corpus_tfidf = []

for i in range(len(sentences)):

    sent = re.sub('[^a-zA-Z]', ' ', sentences[i])

    sent = sent.lower()

    sent = sent.split()

    sent = [wordnet.lemmatize(word) for word in sent if not word in set(stopwords.words('english'))]

    sent = ' '.join(sent)

    corpus_tfidf.append(sent)

    

# Creating the TF-IDF model

from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer()

X_tfidf = cv.fit_transform(corpus_tfidf).toarray()
X_tfidf_df = pd.DataFrame(X_tfidf)

X_tfidf_df
# gensim library for word embedding

from gensim.models import Word2Vec



# define training data

sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],

            ['this', 'is', 'the', 'second', 'sentence'],

            ['yet', 'another', 'sentence'],

            ['one', 'more', 'sentence'],

            ['and', 'the', 'final', 'sentence']]



# train model

model = Word2Vec(sentences, min_count=1)



# summarize the loaded model

print(model)



# summarize vocabulary

words = list(model.wv.vocab)

print(words)



# access vector for one word

print(model['sentence'])



# save model

model.save('model.bin')



# load model

new_model = Word2Vec.load('model.bin')

print(new_model)
# lets see the word which has the highest prob of coming after words 'this' and 'is'

# we can play around with topn variable

result = model.most_similar(positive=['this', 'is'], topn=1)

print(result)