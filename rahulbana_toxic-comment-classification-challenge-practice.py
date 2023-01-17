import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
df_train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
df_test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")

df_submission = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

df_train.head()
#df_test.head(2)

train_data = df_train['comment_text']
test_data = df_test['comment_text']
all_data = pd.concat([train_data, test_data])
#all_data
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
def run_classifier(data, features,  diff_columns, clf):
    kdict = {}
    kdict['class'] = []
    kdict['score_mean'] = []
    kdict['score_std'] = []
    kdict['scores'] = []

    for i in diff_columns:
        labels = data[i]
        res = cross_val_score(clf, features, labels, cv=10, scoring='roc_auc')

        kdict['class'].append(i)
        kdict['score_mean'].append(np.mean(res))
        kdict['score_std'].append(np.std(res))
        kdict['scores'].append(res)

    return pd.DataFrame.from_dict(kdict)
word_vect = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vect.fit(all_data)
train_features = word_vect.transform(train_data)
test_features = word_vect.transform(test_data)

classifier = LogisticRegression(C=0.1, solver='sag')
run_classifier(df_train, train_features,  classes, classifier)
#classifier = SVC()
#run_classifier(df_train, train_features,  classes, classifier)

word_vect = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,3}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vect.fit(all_data)

run_classifier(df_train, train_features, classes, classifier)
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation,Bidirectional, GlobalMaxPool1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.model_selection import train_test_split
EMBEDDING_FILE='../input/glove6b50dtxt/glove.6B.50d.txt'

embed_size = 50 # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a comment to use
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
x_train, x_test, y_train, y_test = train_test_split(X_t, y, test_size=.2, shuffle=True)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(GRU(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=32, epochs=4, validation_data=(x_test, y_test), shuffle=True)
model.evaluate(x_test, y_test)
ypred = model.predict(X_te)
ypred[0]
df_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')
df_submission.head(1)
df_submission['toxic'] = ypred[:,0]
df_submission['severe_toxic'] = ypred[:,1]
df_submission['obscene'] = ypred[:,2]
df_submission['threat'] = ypred[:,3]
df_submission['insult'] = ypred[:,4]
df_submission['identity_hate'] = ypred[:,5]
df_submission.head(1)