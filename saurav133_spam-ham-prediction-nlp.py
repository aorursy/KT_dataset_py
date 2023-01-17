# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
messages = pd.read_csv('../input/spam.csv', encoding='latin-1')

messages.head()
messages.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
messages.head()
messages.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
messages.head()
messages.describe()
messages.groupby('label').describe()
messages['length'] = messages['message'].apply(len)

messages.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
messages['length'].plot(bins=100, kind='hist') 
messages.length.describe()
messages[messages['length'] == 910]['message'].iloc[0]
messages.hist(column='length', by='label', bins=50, figsize=(12,4))
import string



mess = 'Sample message! Notice: it has punctuation.'



# Check characters to see if they are in punctuation

nopunc = [char for char in mess if char not in string.punctuation]



# Join the characters again to form the string.

nopunc = ''.join(nopunc)
from nltk.corpus import stopwords
stopwords.words('english')[0:10]  #to see some stopwords
nopunc.split()
# Now just remove any stopwords

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess
def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]



    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    

    # Now just remove any stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
messages.head()
# Check to make sure its working

messages['message'].head(5).apply(text_process)
from sklearn.feature_extraction.text import CountVectorizer
#this takes time to complete depending on the computer speed

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
# Print total number of vocab words

print(len(bow_transformer.vocabulary_))
message4 = messages['message'][3]

print(message4)
bow4 = bow_transformer.transform([message4])

print(bow4)

print(bow4.shape)
print(bow_transformer.get_feature_names()[3996])

print(bow_transformer.get_feature_names()[9445])
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)

print('Amount of Non-Zero occurences: ', messages_bow.nnz)
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

print('sparsity: {}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer = TfidfTransformer().fit(messages_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
messages_tfidf = tfidf_transformer.transform(messages_bow)

print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])

print('expected:', messages.label[3])
all_predictions = spam_detect_model.predict(messages_tfidf)

print(all_predictions)
from sklearn.metrics import classification_report, confusion_matrix

print (classification_report(messages['label'], all_predictions))

print('\n')

m_confusion_test = confusion_matrix(messages['label'], all_predictions)

pd.DataFrame(data = m_confusion_test, columns = ['Predicted HAM', 'Predicted SPAM'],

            index = ['Actual HAM', 'Actual SPAM'])
from sklearn.model_selection import train_test_split



msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print (classification_report(label_test, predictions))

print('\n')

m_confusion_test = confusion_matrix(label_test, predictions)

pd.DataFrame(data = m_confusion_test, columns = ['Predicted HAM', 'Predicted SPAM'],

            index = ['Actual HAM', 'Actual SPAM'])