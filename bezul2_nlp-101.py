import numpy as np

import pandas as pd

import nltk

import os

print(os.listdir("../input"))
dialog = pd.read_csv('../input/clean_dialog.csv')

dialog.head()
text = dialog['dialog'].loc[0]

print(text)
sentences = nltk.sent_tokenize(text)

print(sentences)
len(sentences)

# Narrator introduction was split into 7 sentences
words_list = []

for s in sentences:

    words = nltk.word_tokenize(s)

    words_list.append(words)
print(words_list)
from nltk.stem import PorterStemmer

from nltk.corpus import wordnet



def stemm(word):

    stemmer = PorterStemmer()

    print('Word: ', word, ' ... Stemmer: ', stemmer.stem(word))

    

stemm('dogs')

stemm('reading')
for w in words_list[2]:

    stemm(w)
from nltk.stem import WordNetLemmatizer



def lemmat(word, pos):

    lemmatizer = WordNetLemmatizer()

    print("Word is: :", word, "   ...   Lemmatizer:", lemmatizer.lemmatize(word, pos))



lemmat('caring', wordnet.VERB)
for w in words_list[2]:

    lemmat(w, wordnet.VERB)
from nltk.corpus import stopwords

print(stopwords.words("english"))
stop_words = set(stopwords.words("english"))



words_list_clean = []

for words in words_list:

    words_clean = [w for w in words if not w in stop_words]

    words_list_clean.append(words_clean)

print('Sentence before removing stop words:')

print(words_list[0])

print()

print('Sentence after removing stop words:')

print(words_list_clean[0])
from sklearn.feature_extraction.text import CountVectorizer



# Here we build our model, only words from whole list of lists

words_for_bag = []

for sen in words_list_clean:

    for wrd in sen:

        words_for_bag.append(wrd)

        

print(len(words_for_bag))        



# Preparing vocabulary dictionary based on sklearn 

count_vectorizer = CountVectorizer()



# Creatin bag-of-words model

bag_of_words = count_vectorizer.fit_transform(words_for_bag)



# bag-of-words model as a pandas df

feature_names = count_vectorizer.get_feature_names()



pd.DataFrame(bag_of_words.toarray(), columns = feature_names)
# We will calculate TF-IDF based on Narrator saying from first episode

print(sentences)
from sklearn.feature_extraction.text import TfidfVectorizer



# Let's create vectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')

X = vectorizer.fit_transform(sentences)



# First vector from first document

first_vector_tfidfvectorizer=X[0]

 

# Preparinf pandas dataframe

df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])

df.sort_values(by=["tfidf"],ascending=False)