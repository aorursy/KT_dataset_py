!wget "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
import zipfile

import os

with zipfile.ZipFile("./glove.twitter.27B.zip","r") as zip_ref:

    zip_ref.extract("glove.twitter.27B.200d.txt")

    print(zip_ref.filelist)

ii = ['glove.twitter.27B.zip']

for i in ii:

    os.remove(i)

print(os.listdir("./"))

del zip_ref
import pandas as pd

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
X_train_comments = train_df['comment_text'].values
y_train_ori = train_df['target'].values
y_train_ori2 = []

for i in y_train_ori:

    if (i>=0.5):

        y_train_ori2.append(1)

    else:

        y_train_ori2.append(0)
from collections import Counter

from sklearn.datasets import make_classification

from imblearn.under_sampling import RandomUnderSampler 

print('Original dataset shape %s' % Counter(y_train_ori2))

rus = RandomUnderSampler(random_state=42)

X_res, y_res = rus.fit_resample(X_train_comments.reshape(-1,1), y_train_ori2)

print('Resampled dataset shape %s' % Counter(y_res))
x_test_comments= test_df["comment_text"].values
import re

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from collections import OrderedDict

import string



from bs4 import BeautifulSoup

from nltk import WordPunctTokenizer

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import SnowballStemmer

from nltk.stem import WordNetLemmatizer

def clean(text):

    tok = WordPunctTokenizer()

    pat1 = '@[\w\-]+'  # for @

    pat2 = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'

            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # for url

    pat3 = '#[\w\-]+'  # for hashtag

    pat4 = 'ï»¿'

    pat5 = '[' + string.punctuation + ']'  # for punctuation

    pat6 = '[^\x00-\x7f]'

    soup = BeautifulSoup(text, 'html.parser')  # html decoding ("@amp")

    souped = soup.get_text()

    souped = re.sub(pat1, '', souped)  # remove @

    souped = re.sub(pat2, '', souped)  # remove url

    souped = re.sub(pat4, '', souped)  # remove strange symbols

    souped = re.sub(pat5, '', souped)  # remove punctuation

    souped = re.sub(pat3, '', souped)  # remove "#" symbol and keeps the words

    clean = re.sub(pat6, '', souped)  # remove non-ascii characters

    lower_case = clean.lower()  # convert to lowercase

    words = tok.tokenize(lower_case)

    return (" ".join(words)).strip()

def my_clean(text,stops = False,stemming = False,minLength = 2):

    text = str(text)

    text = text.lower().split()

    text = [w for w in text if len(w) >= minLength]



    text = " ".join(text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"don't", "do not ", text)

    text = re.sub(r"aren't", "are not ", text)

    text = re.sub(r"isn't", "is not ", text)

    text = re.sub(r"%", " percent ", text)

    text = re.sub(r"that's", "that is ", text)

    text = re.sub(r"doesn't", "dos not ", text)

    text = re.sub(r"he's", "he is ", text)

    text = re.sub(r"she's", "she is ", text)

    text = re.sub(r"it's", "it is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = text.lower().split()

    text = [w for w in text if len(w) >= minLength]

    if stemming and stops:

        text = [word for word in text if word not in stopwords.words('english')]

        wordnet_lemmatizer = WordNetLemmatizer()

        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)

        text = [englishStemmer.stem(word) for word in text]

        text = [wordnet_lemmatizer.lemmatize(word) for word in text]

        # text = [lancaster.stem(word) for word in text]

        text = [word for word in text if word not in stopwords.words('english')]

    elif stops:

        text = [word for word in text if word not in stopwords.words('english')]

    elif stemming:

        wordnet_lemmatizer = WordNetLemmatizer()

        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)

        text = [englishStemmer.stem(word) for word in text]

        text = [wordnet_lemmatizer.lemmatize(word) for word in text]

    text = " ".join(text)

    return text
X_train_comments_pre = []

count = 0

max_length = -5

import time

start = time.time()

for t in X_res:

    te = my_clean(t,False,True,2)

    X_train_comments_pre.append(te)#You can add one more clean()

    length = len(te.split(' '))

    if length > max_length:

        max_length = length

    

    if count % 10000 == 0:

        print(count)

        final = time.time()

        total = final - start

        print(total)

    

    count = count + 1
print(X_res[1])

print(X_train_comments_pre[1])

print(max_length)
from sklearn.model_selection import train_test_split

X_train, X_test ,y_train ,y_test = train_test_split(X_train_comments_pre,y_res, random_state=826, test_size=0.33)
from keras.preprocessing.text import Tokenizer

print("Opening Glove")

embeddings_index = dict()

f = open('./glove.twitter.27B.200d.txt', encoding='utf-8')

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

vocabulary_size = 50000

tokenizer = Tokenizer(num_words=vocabulary_size)

tokenizer.fit_on_texts(X_train)

embedding_matrix = np.zeros((50000, 200))

for word, index in tokenizer.word_index.items():

    if index > 50000 - 1:

        break

    else:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector
from sklearn.base import BaseEstimator, TransformerMixin

from keras.preprocessing.sequence import pad_sequences



class MyPadder(BaseEstimator,TransformerMixin):

    def __init__(self,maxlen=5000):

        self.maxlen = maxlen

        self.max_index = None



    def fit(self,X,y=None):

        self.max_index = pad_sequences(X,maxlen=self.maxlen).max()

        return self



    def transform(self,X,y=None):

        X = pad_sequences(X,maxlen=self.maxlen)

        X[X>self.max_index] = 0

        return X

from keras.preprocessing.text import Tokenizer

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np

class MyTextsToSequences(Tokenizer, BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)



    def fit(self,texts,y=None):

        self.fit_on_texts(texts)

        return self



    def transform(self,texts,y=None):

        return np.array(self.texts_to_sequences(texts))

max_length=250

sequencer = MyTextsToSequences(num_words=50000)

padder = MyPadder(max_length)
sequencer.fit(X_train)

X_train_copy = sequencer.transform(X_train)

X_test_copy = sequencer.transform(X_test)

padder.fit(X_train_copy)

X_train_copy = padder.transform(X_train_copy)

X_test_copy = padder.transform(X_test_copy)
from keras import Input, Model

from keras.optimizers import Adam

from keras.utils import plot_model

from keras.wrappers.scikit_learn import KerasClassifier

from lime.lime_text import LimeTextExplainer

from keras.models import Sequential

from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, concatenate

from keras.layers.embeddings import Embedding

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.pipeline import make_pipeline

import numpy as np

from collections import OrderedDict

from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=500)

vec.fit(X_train)

X_train_copy2 = vec.transform(X_train)

X_test_copy2 = vec.transform(X_test)
main_input = Input(shape=(max_length,), dtype='int32', name='main_input')

glove_Embed = (Embedding(50000, 200, input_length=max_length, weights=[embedding_matrix], trainable=False))(main_input)



x = Conv1D(64, 5, activation='relu')(glove_Embed)

x = Conv1D(32, 5, activation='relu')(x)

x = Dropout(rate=0.05)(x)

x = MaxPooling1D(pool_size=4)(x)

x = Dropout(rate=0.35)(x)

x = LSTM(50)(x)



y = Dense(300,activation='relu')(glove_Embed)

y = Dropout(rate=0.05)(y)

y = LSTM(300)(y)

y = Dropout(rate=0.35)(y)

y = Dense(100,activation='relu')(y)

y = Dense(50,activation='relu')(y)



main_input2 = Input(shape=(len(vec.get_feature_names()),), dtype='float32', name='main_input2')

e = Dense(300,activation='relu')(main_input2)

e = Dense(1000,activation='relu')(e)

e = Dropout(rate=0.35)(e)

e = Dense(200,activation='relu')(e)

e = Dropout(rate=0.05)(e)

e = Dense(50,activation='relu')(e)



z = concatenate([x, y, e])



z = Dense(128,activation='relu')(z)

z = Dropout(0.05)(z)

z = Dense(64,activation='relu')(z)

z = Dropout(0.1)(z)

z = Dense(32,activation='relu')(z)

output_lay = Dense(1, activation='sigmoid')(z)

model = Model(inputs=[main_input,main_input2], outputs=[output_lay])

model.compile(optimizer=Adam(),loss='binary_crossentropy',metrics=['accuracy'])

print(model.summary())

model.fit([X_train_copy,X_train_copy2], [y_train],validation_data=([X_test_copy,X_test_copy2],y_test),epochs=4, batch_size=128)  # starts training

y_predicted = model.predict([X_test_copy,X_test_copy2])



y_pred = []

for i in y_predicted:

    if (i>=0.5):

        y_pred.append(1)

    else:

        y_pred.append(0)

model_name = "dn"

# We want both weighted and macro, because the dataset is imbalanced!

print(model_name, 'f1 weighted', metrics.f1_score(y_pred, y_test, average="weighted"))

print(model_name, 'f1 macro', metrics.f1_score(y_pred, y_test, average="macro"))

print(model_name, 'precision weighted', metrics.precision_score(y_pred, y_test, average="weighted"))

print(model_name, 'precision macro', metrics.precision_score(y_pred, y_test, average="macro"))

print(model_name, 'recall weighted', metrics.recall_score(y_pred, y_test, average="weighted"))

print(model_name, 'recall macro', metrics.recall_score(y_pred, y_test, average="macro"))

print(model_name, 'acc', metrics.accuracy_score(y_pred, y_test))

print()
del X_train_copy2,X_test_copy2, train_df, X_train_copy, X_test_copy, embeddings_index,embedding_matrix, f, X_train_comments_pre, X_train, X_test ,y_train ,y_test, X_res,y_res, y_train_ori2, X_train_comments, y_train_ori
print(x_test_comments[0]) ##Do preproccessing! 

print(len(x_test_comments))

#test_df.head
x_test_comments_pre = []

count = 0

max_length = -5

import time

start = time.time()

for t in x_test_comments:

    te = my_clean(t,False,True,2)

    x_test_comments_pre.append(te)#You can add one more clean()

    length = len(te.split(' '))

    if length > max_length:

        max_length = length

    

    if count % 10000 == 0:

        print(count)

        final = time.time()

        total = final - start

        print(total)

    

    count = count + 1
x_test_comments_pre_copy = sequencer.transform(x_test_comments_pre)

x_test_comments_pre_copy = padder.transform(x_test_comments_pre_copy)
x_test_comments_pre_copy2 = vec.transform(x_test_comments_pre)
new_y_preds = model.predict([x_test_comments_pre_copy,x_test_comments_pre_copy2])
submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': np.mean(new_y_preds, axis=1)

})

submission.to_csv('submission.csv', index=False)
submission