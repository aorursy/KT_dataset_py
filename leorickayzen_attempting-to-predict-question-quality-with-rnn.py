import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt

import re
df = pd.read_csv('../input/Questions.csv', encoding='ISO-8859-1')
df['all_text'] = df['Title'] + ' ' + df['Body']
def bad_question(score):

    if score < 0:

        return 1

    else:

        return 0



df['bad_question'] = df['Score'].apply(lambda x: bad_question(x))
df_questions = df[['all_text', 'bad_question']]

train, test = train_test_split(df_questions, test_size=0.05, random_state=42)
train_bad = train.loc[train['bad_question'] == 1]

train_good = train.loc[train['bad_question'] == 0]
len(train_bad)
len(train_good)
good_sample = train_good.sample(frac=0.05)

train = good_sample.append(train_bad)
df['all_text'].iloc[0]
df['all_text'] = df['all_text'].apply(lambda x: re.sub('(\<code\>.*?<\/code\>)', '', x))
df['all_text'] = df['all_text'].apply(lambda x: re.sub('<[^>]+>', '', x))
df['all_text'].iloc[0]
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(train['all_text'])

X_train_counts.shape
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape
X_test_counts = count_vect.transform(test['all_text'])

X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
clf = RandomForestClassifier(n_estimators = 5, n_jobs = -1, verbose=1, class_weight="balanced")

#clf = MLPClassifier(hidden_layer_sizes = (10, 10), verbose=True, early_stopping=True)

clf.fit(X_train_tfidf, train['bad_question'])
y_pred = clf.predict(X_test_tfidf)

y_pred_train = clf.predict(X_train_tfidf)
precision_recall_fscore_support(train['bad_question'], y_pred_train, average='macro')
precision_recall_fscore_support(test['bad_question'], y_pred, average='macro')
from sklearn.preprocessing import LabelEncoder

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.optimizers import RMSprop

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping
X_train = train['all_text']

y_train = train['bad_question']

X_test = test['all_text']

y_test = test['bad_question']
max_words = 2000

max_len = 150

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():

    inputs = Input(name='inputs',shape=[max_len])

    layer = Embedding(max_words,50,input_length=max_len)(inputs)

    layer = LSTM(64)(layer)

    layer = Dense(256,name='FC1')(layer)

    layer = Activation('relu')(layer)

    layer = Dropout(0.9)(layer)

    layer = Dense(1,name='out_layer')(layer)

    layer = Activation('sigmoid')(layer)

    model = Model(inputs=inputs,outputs=layer)

    return model
model = RNN()

model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(sequences_matrix,y_train,batch_size=128,epochs=20,

          validation_split=0.2)
test_sequences = tok.texts_to_sequences(X_test)

test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
y_hat = model.predict(test_sequences_matrix, verbose = 1)
y_hat[y_hat > 0.5] = 1

y_hat[y_hat <= 0.5] = 0

precision_recall_fscore_support(y_test, y_hat, average='macro')
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords

from sklearn.svm import LinearSVC

import string

import re

import spacy

spacy.load('en')

from spacy.lang.en import English

parser = English()
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]



class CleanTextTransformer(TransformerMixin):

    def transform(self, X, **transform_params):

        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):

        return self



def get_params(self, deep=True):

        return {}

    

def cleanText(text):

    text = text.strip().replace("\n", " ").replace("\r", " ")

    text = text.lower()

    return text



def tokenizeText(sample):

    tokens = parser(sample)

    lemmas = []

    for tok in tokens:

        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

    tokens = lemmas

    tokens = [tok for tok in tokens if tok not in STOPLIST]

    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    return tokens

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))



clf = LinearSVC()



pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='macro')