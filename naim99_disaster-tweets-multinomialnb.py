



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import re
train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data  =pd.read_csv('../input/nlp-getting-started/test.csv')

train_data.head(10)

train_data.dtypes
train_data['text'][11]
import re

def  clean_text(df, text_field, new_text_field_name):

    df[new_text_field_name] = df[text_field].str.lower()

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

    # remove numbers

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))

    

    return df

data_clean = clean_text(train_data, 'text', 'text_clean')

data_clean_test = clean_text(test_data,'text', 'text_clean')

data_clean.head()
import nltk.corpus

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

data_clean.head()
import nltk 

from nltk.tokenize import BlanklineTokenizer

from nltk.tokenize import TweetTokenizer

nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

data_clean['text_tokens'] = data_clean['text_clean'].apply(lambda x: word_tokenize(x))

data_clean.head()
from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize

def word_stemmer(text):

    stem_text = [PorterStemmer().stem(i) for i in text]

    return stem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_stemmer(x))

data_clean.head()
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]

    return lem_text

data_clean['text_clean_tokens'] = data_clean['text_tokens'].apply(lambda x: word_lemmatizer(x))

data_clean.head()
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)







data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_URL(x))
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_html(x))
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_emoji(x))
import string

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: remove_punct(x))
"""!pip3 install pyspellchecker==20.2.2

from spellchecker import SpellChecker



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)



data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: correct_spellings(x))""" 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from multiprocessing import Pool





freq = pd.Series(' '.join(data_clean['text_clean']).split()).value_counts()[:10]



freq = list(freq.index)

data_clean['text_clean'] = data_clean['text_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
X_train, X_test, Y_train, Y_test = train_test_split(data_clean['text_clean'], 

                   

                                                    data_clean['target'], 

                                                    test_size = 0.2,

                                                    random_state = 10)


tfidf = TfidfVectorizer(encoding='utf-8',

                       ngram_range=(1,1),

                       max_df=1.0,

                       min_df=10,

                       max_features=500,

                       norm='l2',

                       sublinear_tf=True)
train_features = tfidf.fit_transform(X_train).toarray()

print(train_features.shape)
test_features = tfidf.transform(X_test).toarray()

print(test_features.shape)
train_labels = Y_train

test_labels = Y_test


import pandas as pd

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
mnb_classifier = MultinomialNB()
mnb_classifier.fit(train_features, train_labels)
mnb_prediction = mnb_classifier.predict(test_features)
training_accuracy = accuracy_score(train_labels, mnb_classifier.predict(train_features))

print(training_accuracy)
testing_accuracy = accuracy_score(test_labels, mnb_prediction)

print(testing_accuracy)
print(classification_report(test_labels, mnb_prediction))
conf_matrix = confusion_matrix(test_labels, mnb_prediction)

print(conf_matrix)
test_vectorizer =tfidf.transform( data_clean_test['text_clean']).toarray()
test_vectorizer.shape
final_predictions = mnb_classifier.predict(test_vectorizer)
final_predictions
submission_df = pd.DataFrame()
submission_df['id'] = data_clean_test['id']

submission_df['target'] = final_predictions
submission_df
submission_df['target'].value_counts()
submission = submission_df.to_csv('Result.csv',index = False)