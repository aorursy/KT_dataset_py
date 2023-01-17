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
messages = [line.rstrip() for line in open('../input/SMSSpamCollection')]
print(len(messages))
# Let's print the first 10 messages and number them
for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')
import pandas as pd
messages = pd.read_csv('../input/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()
messages.describe()
messages.groupby('label').describe()
messages['length'] = messages['message'].apply(len)
messages.head()
# Data Viz
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style(style = 'whitegrid')
messages['length'].plot(bins=50, kind='hist')
messages.length.describe()
# Woah! 910 characters, let's use masking to find this message:
messages[messages['length'] == 910]['message'].iloc[0]
# Looks like we have some sort of Love Letter here!
messages.hist(column='length', by='label', bins=50,figsize=(12,4))
# Text Preprocessing
import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)
# Removing Stopwords
from nltk.corpus import stopwords
stopwords.words('english')[0:10]
nopunc.split()
# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess
# Now let's put both of these together in a function to apply it to our DataFrame later on:
def text_process(mess):
    
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#Takes in a string of text, then performs the following:
#1. Remove all punctuation
#2. Remove all stopwords
#3. Returns a list of the cleaned text
messages.head()
# Now let's "tokenize" these messages
# Check to make sure its working
messages['message'].head(5).apply(text_process)
messages.head()
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer:
message4 = messages['message'][3]
print(message4)
# Now let's see its vector representation:
bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)
# This means that there are seven unique words in message number 4 (after removing common stop words).
#Two of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:
print(bow_transformer.get_feature_names()[4068])
print(bow_transformer.get_feature_names()[9554])
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
# We'll go ahead and check what is the IDF (inverse document frequency) of the word "u" and of word "university"?
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
# To transform the entire bag-of-words corpus into TF-IDF corpus at once:
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
# Let's try classifying our single random message and checking how we do:
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))
# In the above "evaluation",we evaluated accuracy on the same data we used for training.
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
# Creating a Data Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
# Now we can directly pass message text data and the pipeline will do our pre-processing for us! We can treat it as a model/estimator API:
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))
# Now we have a classification report for our model on a true testing set!