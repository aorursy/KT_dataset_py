import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

import matplotlib.lines as mlines

import seaborn as sns



import warnings



import pickle

import time



import re

from bs4 import BeautifulSoup

import nltk

from nltk.tokenize import ToktokTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

from string import punctuation



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import hamming_loss





import logging

from scipy.sparse import hstack



warnings.filterwarnings("ignore")

plt.style.use('bmh')

%matplotlib inline
# Setting a random seed in order to keep the same random results each time I run the notebook

np.random.seed(seed=11)
import os 

print(os.listdir("../input"))
# Importing the database 



df = pd.read_csv("../input/Questions.csv", encoding="ISO-8859-1")
df.head(5)
tags = pd.read_csv("../input/Tags.csv", encoding="ISO-8859-1", dtype={'Tag': str})
tags.head(5)
df.info()
tags.info()
tags['Tag'] = tags['Tag'].astype(str)
grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
grouped_tags.head(5)
grouped_tags.reset_index()
grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags.values})
grouped_tags_final.head(5)
df.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
df = df.merge(grouped_tags_final, on='Id')
df.head(5)
df = df[df['Score']>5].copy()
print('Dupplicate entries: {}'.format(df.duplicated().sum()))

df.drop_duplicates(inplace = True)
df.drop(columns=['Id', 'Score'], inplace=True)
from collections import Counter
df.head(5)
df['Tags'] = df['Tags'].apply(lambda x: x.split())
all_tags = [item for sublist in df['Tags'].values for item in sublist]
len(all_tags)
my_set = set(all_tags)

unique_tags = list(my_set)

len(unique_tags)
counts = Counter(all_tags)
print(counts.most_common(20))
frequencies_words = counts.most_common(20)

tags_features = [word[0] for word in frequencies_words]
print(tags_features)
def most_common(tags):

    tags_filtered = []

    for i in range(0, len(tags)):

        if tags[i] in tags_features:

            tags_filtered.append(tags[i])

    return tags_filtered
df['Tags'] = df['Tags'].apply(lambda x: most_common(x))

df['Tags'] = df['Tags'].apply(lambda x: x if len(x)>0 else None)
df.shape
df.dropna(subset=['Tags'], inplace=True)
df.shape
!pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git
import preprocess_kgptalkie as ps
def get_clean(x):

    x = str(x).lower().replace('\\', '').replace('_', ' ')

    x = ps.cont_exp(x)

    x = ps.remove_emails(x)

    x = ps.remove_urls(x)

    x = ps.remove_html_tags(x)

    x = ps.remove_accented_chars(x)

    x = ps.remove_special_chars(x)

    x = re.sub("(.)\\1{2,}", "\\1", x)

    return x
df['Body'] = df['Body'].apply(lambda x: get_clean(x))
df['Title'] = df['Title'].apply(lambda x: get_clean(x))
df['Text'] = df['Title'] + " " +  df['Body']
df.head()
y = df['Tags']
multilabel = MultiLabelBinarizer()

y = multilabel.fit_transform(y)
tfidf = TfidfVectorizer(analyzer = 'word', max_features=1000)

X = tfidf.fit_transform(df['Text'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) # Do 70/30 split
def avg_jacard(y_true,y_pred):

    '''

    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics

    https://www.oreilly.com/library/view/mastering-machine-learning/9781788299879/87b63eb8-f52c-496a-b73b-42f8aef549fb.xhtml

    '''

    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)

    

    return jacard.mean()*100



def print_score(y_pred, clf):

    print("Clf: ", clf.__class__.__name__)

    print("Jacard score: {}".format(avg_jacard(y_test, y_pred)))

    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)*100))

    print("---")    
sgd = SGDClassifier()

lr = LogisticRegression()

svc = LinearSVC()



for classifier in [sgd, lr, svc]:

    clf = OneVsRestClassifier(classifier)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print_score(y_pred, classifier)
for i in range(y_train.shape[1]):

    print(multilabel.classes_[i])

    print(confusion_matrix(y_test[:,i], y_pred[:,i]))

    print("")
df[['Text', 'Tags']].to_csv('stackoverflow.csv')
import pandas as pd

import numpy as np



from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix

from sklearn.metrics import hamming_loss





# warnings.filterwarnings("ignore")

# plt.style.use('bmh')

# %matplotlib inline
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Flatten,Embedding,Activation, Dropout

from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D 
import tensorflow as tf

print(tf.__version__)
from sklearn.model_selection import train_test_split

import ast
df = pd.read_csv('./stackoverflow.csv', index_col = 0)
df.head()
df['Tags'] = df['Tags'].apply(lambda x: ast.literal_eval(x))
df['Tags']
df['Tags'].iloc[0]
multilabel = MultiLabelBinarizer()

y = multilabel.fit_transform(df['Tags'])
multilabel.classes_
text = df['Text'].tolist()
text[:2]
token = Tokenizer()

token.fit_on_texts(text)
y
y.shape
len(token.word_counts)
vocab_size = len(token.word_index) + 1 #https://keras.io/api/layers/core_layers/embedding/

vocab_size
x = ['i love i rt the']

# x = [1, 2, 3, 4, 6]
token.texts_to_sequences(x)
encoded_text = token.texts_to_sequences(text)
max_length = 100

X = pad_sequences(encoded_text, maxlen=max_length, padding='post')
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)
from tensorflow.keras.optimizers import Adam

vec_size = 50

def get_model():

  model = Sequential()

  model.add(Embedding(vocab_size, vec_size, input_length=max_length))



  model.add(Conv1D(32, 2, activation = 'relu'))

  model.add(MaxPooling1D(2))

  model.add(Dropout(0.2))



  model.add(Conv1D(64, 3, activation = 'relu'))

  model.add(MaxPooling1D(2))

  model.add(Dropout(0.3))



#   model.add(Dense(128, activation='relu'))

#   model.add(Dropout(0.2))



  model.add(Dense(128, activation='relu'))



  model.add(GlobalMaxPooling1D())



  model.add(Dense(y.shape[1], activation='softmax'))



  return model



from keras import backend as K

def avg_jacard(y_true,y_pred):

    '''

    see https://en.wikipedia.org/wiki/Multi-label_classification#Statistics_and_evaluation_metrics

    '''

    jacard = K.sum(K.minimum(y_true,y_pred)) / K.sum(K.maximum(y_true,y_pred))

    

    return K.mean(jacard)
model = get_model()

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = [avg_jacard])

model.fit(X_train, y_train, epochs = 1, validation_data = (X_test, y_test), batch_size = 128)
x = "I have some HTML tables which I extract from a third party program which I'd like to show without using a javascript the user gets to see 4 categories and each category has multiple options. From each category only 1 item can be selected"
def get_clean(x):

    x = str(x).lower().replace('\\', '').replace('_', ' ')

    x = ps.cont_exp(x)

    x = ps.remove_emails(x)

    x = ps.remove_urls(x)

    x = ps.remove_html_tags(x)

    x = ps.remove_accented_chars(x)

    x = ps.remove_special_chars(x)

    x = re.sub("(.)\\1{2,}", "\\1", x)

    return x
def get_encoded(x):

#   x = get_clean(x)

  x = token.texts_to_sequences([x])

  x = pad_sequences(x, maxlen=max_length, padding = 'post')

  return x
coded = get_encoded(x)
coded
model.predict_classes(coded)
multilabel.inverse_transform(model.predict_classes(coded))
multilabel.classes_[11]
multilabel.classes_