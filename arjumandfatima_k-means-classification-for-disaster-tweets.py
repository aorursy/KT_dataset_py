df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

print( df.head())

print(df['text'])
from nltk import sent_tokenize



df['tokenized_sents'] = df.apply(lambda column: sent_tokenize(column['text']), axis=1)

print(df.head())
import re

from nltk.corpus import stopwords

from nltk import word_tokenize



stop = stopwords.words('english')

df['tokenized_sents'] = df['text'].str.lower()

df['tokenized_sents'] = df['tokenized_sents'].apply(lambda x: re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', str(x),flags=re.MULTILINE))



df['tokenized_sents'] = df['tokenized_sents'].apply(lambda x: re.sub("[^a-zA-Z]",  " ", str(x)))

df['tokenized_sents'] = df['tokenized_sents'].str.strip()

df['tokenized_sents'] = df.apply(lambda column: word_tokenize(column['tokenized_sents']), axis=1)

print(df["tokenized_sents"])
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

import numpy as np

# stopword removal, stemming and lemmatization and then finding unique words

allwords = []

all_sentences = df['tokenized_sents'].tolist()

for slist in all_sentences:

    for s in slist:

        if s not in stop: 

            allwords.append(wordnet_lemmatizer.lemmatize(porter_stemmer.stem(s)))



set_allwords = set(allwords)

allwords_unique = list(set_allwords)

print("length of unique words ", len(allwords_unique))



# finding tfidf for unique words 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

vectorizer = TfidfVectorizer(stop)

X = vectorizer.fit_transform(allwords_unique)



# Training K-means for k=2 (real or not ) 

true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

model.fit(X)

labels = set(model.labels_)

labels = list(labels)

count =0



#testing the trained model on test dataset



# read test.csv

import pandas as pd

df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



# dataframe for storing test results based on sample submission format

df_test_results = pd.DataFrame( columns=["id", "target"])

# for each tweet text in test dataset, preprocess "text" and then transform it and used the transformed representation for prediction

for x in df_test["text"]:

    text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', x.lower(),flags=re.MULTILINE)

    text = re.sub("[^a-zA-Z]",  " ", text)

    text = text.strip()

    words = word_tokenize(text)

    for w in words:

        w = wordnet_lemmatizer.lemmatize(porter_stemmer.stem(w))

    X = vectorizer.transform(list(set(words)))

    predicted = model.predict(X)

    # the predicted model returns a list of integers containing predicted label for each word. The idea is to assign the most frequently occuring label to the overall tweet.

    if np.count_nonzero(predicted == labels[0]) >  np.count_nonzero(predicted == labels[1]) :

        predicted_label = labels[0]

#         print(predicted_label)

        df_test_results.at[count,'target'] = predicted_label

    else:

        predicted_label = labels[1]

#         print(predicted_label)

        df_test_results.at[count,'target'] = predicted_label



#     count = count + 1

df_test_results["id"] = df_test["id"]



# print(df_test_results)

export_csv = df_test_results.to_csv (r'ArjumandFatima_NLP_DisasterTweet_Submission1.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


