# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np #linear algebra

import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

from wordcloud import WordCloud

import nltk

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import operator



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score as acc

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/training_nlu_smaller.csv")
df_no = df[df['previous_intent'] == 'no']

df_no.drop(['previous_intent','template'], axis = 1, inplace = True)

df_no.reset_index(drop=True, inplace=True)

df_no
labels = set(df_no['current_intent'])
book_len = df_no[df_no['current_intent'] == 'book'].shape[0]

neg_len = df_no[df_no['current_intent'] == 'negation'].shape[0]

check_len = df_no[df_no['current_intent'] == 'check-in'].shape[0]

status_len = df_no[df_no['current_intent'] == 'status'].shape[0]

cancel_len = df_no[df_no['current_intent'] == 'cancel'].shape[0]
plt.bar(10,book_len,3, label="book")

plt.bar(15,neg_len,3, label="negation")

plt.bar(20,check_len,3, label="check-in")

plt.bar(25,status_len,3, label="status")

plt.bar(30,cancel_len,3, label="cancel")



plt.legend()

plt.ylabel('Number of examples')

plt.title('Propoertion of examples')

plt.show()
def remove_punctuation(text):

    import string

    translator = str.maketrans('', '', string.punctuation)

    return text.translate(translator)
df_no['query'] = df_no['query'].apply(remove_punctuation)

df_no.head(10)
def cloud(text,c, l):

    wordcloud = WordCloud(background_color='white', colormap='Spectral',random_state = 42).generate(" ".join([i for i in text.str.upper()]))

    #fig = plt.figure(figsize=(15,1))

    plt.figure(figsize=(20,10))

    plt.subplot(2,3,c)

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis("off")



countn = 0

for label in labels:

    countn = countn + 1

    df_label = df_no[df['current_intent']==label]

    cloud(df_label['query'], countn, label)
words = nltk.word_tokenize(" ".join(df_no['query'].values.tolist()))

counter = Counter(words)

print(counter.most_common(50))
def ngrams(input, n):

    input = nltk.word_tokenize(" ".join(input.values.tolist()))

    #input = input.split(' ')

    output = {}

    for i in range(len(input)-n+1):

        g = ' '.join(input[i:i+n])

        output.setdefault(g, 0)

        output[g] += 1

    return output
top_dict = {}

for label in labels:

    df_label = df_no[df['current_intent']==label]

    my_unigrams = ngrams(df_label['query'], 1)

    my_unigrams = list(reversed(sorted(my_unigrams.items(), key=operator.itemgetter(1))))

    top_dict[label] = my_unigrams[:10]

top_dict
countn = 0

for label in labels:

    countn = countn + 1

    vocab = []

    count = []

    for key, value in top_dict[label]:

        vocab.append(key)

        count.append(value)

    my_unigrams = pd.Series(count, index=vocab)

    my_unigrams = my_unigrams.sort_values(ascending=False)

    plt.subplot(2,3,countn)

    plt.xlabel(label)  

    top_vacab = my_unigrams.head(10)

    top_vacab.plot(kind = 'barh', figsize=(20,10))
top_dict = {}

for label in labels:

    df_label = df_no[df['current_intent']==label]

    my_unigrams = ngrams(df_label['query'], 2)

    my_unigrams = list(reversed(sorted(my_unigrams.items(), key=operator.itemgetter(1))))

    top_dict[label] = my_unigrams[:10]

top_dict
countn = 0

for label in labels:

    countn = countn + 1

    vocab = []

    count = []

    for key, value in top_dict[label]:

        vocab.append(key)

        count.append(value)

    my_unigrams = pd.Series(count, index=vocab)

    my_unigrams = my_unigrams.sort_values(ascending=False)

    plt.subplot(2,3,countn)

    plt.xlabel(label)  

    top_vacab = my_unigrams.head(10)

    top_vacab.plot(kind = 'barh', figsize=(20,10))
top_dict = {}

for label in labels:

    df_label = df_no[df['current_intent']==label]

    my_unigrams = ngrams(df_label['query'], 3)

    my_unigrams = list(reversed(sorted(my_unigrams.items(), key=operator.itemgetter(1))))

    top_dict[label] = my_unigrams[:10]

top_dict
countn = 0

for label in labels:

    countn = countn + 1

    vocab = []

    count = []

    for key, value in top_dict[label]:

        vocab.append(key)

        count.append(value)

    my_unigrams = pd.Series(count, index=vocab)

    my_unigrams = my_unigrams.sort_values(ascending=False)

    plt.subplot(3,2,countn)

    plt.xlabel(label)  

    top_vacab = my_unigrams.head(10)

    top_vacab.plot(kind = 'barh', figsize=(20,20))
top_dict = {}

for label in labels:

    df_label = df_no[df['current_intent']==label]

    my_unigrams = ngrams(df_label['query'], 4)

    my_unigrams = list(reversed(sorted(my_unigrams.items(), key=operator.itemgetter(1))))

    top_dict[label] = my_unigrams[:10]

top_dict
countn = 0

for label in labels:

    countn = countn + 1

    vocab = []

    count = []

    for key, value in top_dict[label]:

        vocab.append(key)

        count.append(value)

    my_unigrams = pd.Series(count, index=vocab)

    my_unigrams = my_unigrams.sort_values(ascending=False)

    plt.subplot(3,2,countn)

    plt.xlabel(label)  

    top_vacab = my_unigrams.head(10)

    top_vacab.plot(kind = 'barh', figsize=(20,20))
def length(text):    

    text = text.split(' ')

    return len(text)
df_no['length'] = df_no['query'].apply(length)

df_no.head(10)
book_len = df_no[df_no['current_intent'] == 'book']

neg_len = df_no[df_no['current_intent'] == 'negation']

check_len = df_no[df_no['current_intent'] == 'check-in']

status_len = df_no[df_no['current_intent'] == 'status']

cancel_len = df_no[df_no['current_intent'] == 'cancel']
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

bins = 1

plt.hist(book_len['length'], alpha = 0.6, bins=bins, label='Book')

plt.hist(neg_len['length'], alpha = 0.8, bins=bins, label='Neg')

plt.hist(check_len['length'], alpha = 0.4, bins=bins, label='Check')

plt.hist(status_len['length'], alpha = 0.2, bins=bins, label='Status')

plt.hist(cancel_len['length'], alpha = 0.3, bins=bins, label='Cancel')



plt.xlabel('length')

plt.ylabel('numbers')

plt.legend(loc='upper right')

plt.xlim(0,30)

plt.grid()

plt.show()
tfid_vectorizer = TfidfVectorizer("english")

tfid_vectorizer.fit(df_no['query'])

tfid_matrix = tfid_vectorizer.transform(df_no['query'])

array = tfid_matrix.todense()
df_tf = pd.DataFrame(array)

df_tf.head(10)
encoder = LabelEncoder()

encoder.fit(df_no['current_intent'])

feature_labels = encoder.transform(df_no['current_intent'])

# feature_labels = np_utils.to_categorical(feature_labels)

feature_labels = pd.DataFrame(feature_labels)

# feature_labels.head(10)
from sklearn.feature_selection import SelectKBest, chi2



X_new = SelectKBest(chi2, k=300).fit_transform(df_tf, feature_labels)

X_new.shape
df_feature = pd.DataFrame(X_new)

df_feature.head(10)
X_train, X_test, y_train, y_test = train_test_split(

    df_feature,

    feature_labels,

    test_size=0.25,

    random_state=42)



print('Training dataset shape:', X_train.shape, y_train.shape)

print('Testing dataset shape:', X_test.shape, y_test.shape)
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



results = confusion_matrix(y_test, y_pred) 

  

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_test, y_pred) )

print('Report : ')

print(classification_report(y_test, y_pred) )