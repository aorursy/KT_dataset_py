import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
import nltk
nltk.download_shell()
messages = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')
messages.head()
messages=messages[['v1','v2']]
messages.head()
messages.columns = ['label','message']
messages.describe()
messages.groupby('label',axis=0).describe()
messages['length'] = messages['message'].apply(len)
messages.head()
messages['length'].plot(bins=50,kind='hist')
messages.describe()
messages[messages['length']==910]['message'].iloc[0]
messages.hist(column='length',by='label',bins=50,figsize=(12,4))
import string
mess = "Sample Message! Notice: it has punctuatuation."
mess
nopunc_mess = [char for char in mess if char not in string.punctuation]
nopunc_mess
nopunc_mess = ''.join(nopunc_mess)
nopunc_mess
from nltk.corpus import stopwords
# Printing out some stopwords
stopwords.words('english')[0:10]
nopunc_mess.split()
# Removing the stopwords
clean_mess = [word for word in nopunc_mess.split() if word.lower() not in stopwords.words('english')]
clean_mess
def text_process(mess):

    # Removing the punctuation from the string

    nopunc_mess = [char for char in mess if char not in string.punctuation]

    

    # Join the characters again to form the string

    nopunc_mess = ''.join(nopunc_mess)

    

    # Removing any stopwords in the list of words

    return [word for word in nopunc_mess.split() if word.lower() not in stopwords.words('english')]
messages.head()
messages['message'].head(5).apply(text_process)
messages.head()
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
message4 = messages['message'][3]
print(message4)
bow4 = bow_transformer.transform([message4])
bow4.shape
print(bow4)
print(bow_transformer.get_feature_names()[4068])

print(bow_transformer.get_feature_names()[9554])
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)

print('Amount of Non-Zero occurences: ', messages_bow.nnz)
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

print('sparsity: {}'.format((sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer
# Creating an instance of tfidf transformer and fitting it to the bag of words
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
tfidf_transformer.idf_[bow_transformer.vocabulary_["university"]]
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])

print('expected:', messages.label[3])
all_pred = spam_detect_model.predict(messages_tfidf)
all_pred
X = messages['message']

y = messages['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train
from sklearn.pipeline import Pipeline
pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))