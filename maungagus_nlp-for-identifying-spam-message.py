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
import nltk

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = [line.strip() for line in open('../input/SMSSpamCollection')]



print(len(df))
for message_no, df in enumerate(df[:5]):

    print(message_no, df)

    print('\n')
import pandas as pd
messages = pd.read_csv('../input/SMSSpamCollection', sep='\t', names=["label", "message"])



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
messages.hist(column='length', by='label', bins=100, figsize=(12,4))
import string

from nltk.corpus import stopwords
def text_process(mess):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    #Cek apakah karakternya berupa tanda baca (punctuation)

    nopunc = [char for char in mess if char not in string.punctuation]

    

    #Menggabungkan karakter-karakternya lagi untuk membentuk data string

    nopunc = ''.join(nopunc)

    

    #Menghilangkan stopwords

    clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    

    return clean_mess
messages['message'].head(5).apply(text_process)[3]
messages.head()
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
print(len(bow_transformer.vocabulary_))
bow_transformer.vocabulary_
message4 = messages['message'][3]

print(message4)
bow4 = bow_transformer.transform([message4])

print(bow4)
print(bow_transformer.get_feature_names()[4068])

print(bow_transformer.get_feature_names()[9554])
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse matrix: ', messages_bow.shape)

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
messages_tfidf
messages.label.head()
from sklearn.naive_bayes import MultinomialNB



spam_detect_model = MultinomialNB()

spam_detect_model.fit(messages_tfidf, messages['label'])
print('predicted: ', spam_detect_model.predict(tfidf4)[0])

print('expected: ', messages.label[3])
all_predictions = spam_detect_model.predict(messages_tfidf)

print(all_predictions)
from sklearn.metrics import classification_report,confusion_matrix

print (classification_report(messages['label'], all_predictions))
print(confusion_matrix(messages['label'], all_predictions))