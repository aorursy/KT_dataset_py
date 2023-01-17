# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import string

from nltk.util import ngrams

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_colwidth', 100)





from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()
import cufflinks as cf

import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



init_notebook_mode(connected=True)

cf.go_offline()
train_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")

sample_df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/sample_submission.csv")
df = train_df
train_df.shape, test_df.shape
train_df.head()
target_dist = train_df.target.value_counts()

target_dist = target_dist.reset_index().rename(columns={'index':"Target Label", 'target':'Count'})
fig = px.pie(target_dist, names='Target Label', values='Count', title="Target Label Distribution")

fig.show()
# seperating out label 0 and 1 data

label1 = train_df[train_df.target==1]

label0 = train_df[train_df.target==0]
label1.head()
label0.head()
def count_stopwords(text):

    count = 0

    text = text.split(" ")

    text = [word for word in text if word in stopwords.words('english')]

    # print(text)

    return len(text)



def count_punct(text):

    count = 0

    text = [c for c in text if c in list(string.punctuation)]

    # print(text)

    return len(text)



def get_ngrams(text, n=2):

    ngram = list(ngrams(text.split(), n=n))

    df = pd.DataFrame.from_dict(data = dict(Counter(ngram)), orient='index')

    return df
train_df = train_df.reindex(np.random.permutation(train_df.index))
train_df_subset = train_df.iloc[:100000, :]
train_df_subset['length_of_sent'] = train_df_subset.question_text.apply(len)

train_df_subset["word_count"] = train_df_subset.question_text.apply(lambda x: len(x.split(" ")))

train_df_subset['count_stopwords'] = train_df_subset.question_text.apply(lambda x : count_stopwords(x))

train_df_subset['count_punctuation'] = train_df_subset.question_text.apply(lambda x : count_punct(x))
train_df_subset.head()
train_df_subset = train_df_subset.reset_index(drop=True)
full_text = ""

for i in range(train_df_subset.shape[0]):

    full_text = full_text + " " + train_df_subset.question_text[i]
bigram_df = get_ngrams(full_text, n=2)

trigram_df = get_ngrams(full_text, n=3)

quadgram_df = get_ngrams(full_text, n=4)



del full_text
bigram_df = bigram_df.sort_values(by=0, ascending=False)

trigram_df = trigram_df.sort_values(by=0, ascending=False)

quadgram_df = quadgram_df.sort_values(by=0, ascending=False)
label1 = train_df_subset[train_df_subset.target==1]

label0 = train_df_subset[train_df_subset.target==0]
fig1 = px.histogram(train_df_subset, x="length_of_sent", color = "target",

                   title = "Length of Sentence Distribution")

fig1.show()



fig2 = px.histogram(train_df_subset, x="word_count", color = "target",

                   title = "Count of Word Distribution")

fig2.show()



fig3 = px.histogram(train_df_subset, x="count_punctuation", color = "target",

                   title = "Count of Punctuation Distribution")

fig3.show()



fig4 = px.histogram(train_df_subset, x="count_stopwords", color = "target",

                   title = "Count of Stopwords Distribution")

fig4.show()
label1_sort = label1.sort_values(by="length_of_sent")

label0_sort = label0.sort_values(by="length_of_sent")



fig1 = px.bar(label1_sort.head(20).reset_index(drop=True), y='length_of_sent', title="Short Lengths - 0")

fig2 = px.bar(label0_sort.head(20).reset_index(drop=True), y='length_of_sent', title="Short Lengths - 1")

fig1.show()

fig2.show()



fig3 = px.bar(label1_sort.tail(20).reset_index(drop=True), y='length_of_sent', title="Long Lengths - 1")

fig4 = px.bar(label0_sort.tail(20).reset_index(drop=True), y='length_of_sent', title="Long Lengths - 0")

fig3.show()

fig4.show()
# bigram_df
bigram_df = bigram_df.reset_index().rename(columns={'index':'Bigram', 0:'count'})

trigram_df = trigram_df.reset_index().rename(columns={'index':'Trigram', 0:'count'})

quadgram_df = quadgram_df.reset_index().rename(columns={'index':'Quadgram', 0:'count'})
bigram_top50= bigram_df.head(50)

bigram_top50.Bigram = bigram_top50.Bigram.apply(lambda x: " ".join(x))



trigram_top50= trigram_df.head(50)

trigram_top50.Trigram = trigram_top50.Trigram.apply(lambda x: " ".join(x))



quadgram_top50= quadgram_df.head(50)

quadgram_top50.Quadgram = quadgram_top50.Quadgram.apply(lambda x: " ".join(x))
fig1 = bigram_top50.iplot(kind='barh', x='Bigram', y='count', title='Most Frequent top 20 Bigrams')

fig2 = trigram_top50.iplot(kind='barh', x='Trigram', y='count', title='Most Frequent top 20 Trigrams')

fig3 = quadgram_top50.iplot(kind='barh', x='Quadgram', y='count', title='Most Frequent top 20 Quadgram')
X_original = df.question_text

Y_Original = df.target
df.target.value_counts()
# Undersampling



df = df.sample(frac=1)



df_label0 = df[df.target==0].iloc[:80810, :]

df_label1 = df[df.target==1]



undersampled_df = pd.concat([df_label0, df_label1])

new_df = undersampled_df.sample(frac=1, random_state=21)



new_df.head()
undersampled_target_dist = new_df.target.value_counts().reset_index().rename(columns={'index':'Target Label', 'target':'Count'})

fig = px.pie(undersampled_target_dist,  names='Target Label', values='Count', title="Target Label Distribution")

fig.show()
del bigram_df

del trigram_df

del quadgram_df

del bigram_top50

del trigram_top50

del quadgram_top50

del label0

del label0_sort

del label1

del label1_sort
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",

                       "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",

                       "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",

                       "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", 

                       "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",

                       "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am",

                       "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",

                       "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",

                       "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",

                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",

                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",

                       "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",

                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is",

                       "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",

                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",

                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",

                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",

                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is",

                       "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                       "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                       "you'd": "you would", "you'd've": "you would have",

                       "you'll": "you will", "you'll've": "you will have",

                       "you're": "you are", "you've": "you have" }
# ! pip install autocorrect

# ! pip install contractions

# from autocorrect import Speller

# import contractions

# spell = Speller('en')



def clean_data(text):

    text = " ".join([contraction_mapping[word].lower() if word in contraction_mapping.keys() else word.lower() for word in text.split(' ')])

    text = text.replace('http.*.com', '')

    text = re.sub('<[^<]+?>','', text)

    text = ''.join([c for c in text if c not in list(string.punctuation)])

    text = ''.join(c for c in text if not c.isdigit())

    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split(" ")])

    # text = ' '.join([spell(w) for w in (nltk.word_tokenize(text))])

    # text = contractions.fix(text)

    # text = ' '.join([word for word in text.split(' ') if word not in stopwords.words('english')])

    text = text.replace("’", ' ')

    text = text.replace('“', ' ')

    text = re.sub(' +', ' ', text)

    return text
# new_df.question_text = new_df.question_text.apply(lambda text: text.replace('http.*.com', ''))# 
# new_df.question_text = new_df.question_text.apply(lambda text: re.sub('<[^<]+?>','', text))
# new_df.question_text = new_df.question_text.apply(lambda text: ''.join([c for c in text if c not in list(string.punctuation)]))#
# new_df.question_text = new_df.question_text.apply(lambda text: ''.join(c for c in text if not c.isdigit()))
# new_df.question_text = new_df.question_text.apply(lambda text: ' '.join([lemmatizer.lemmatize(word) for word in text.split(" ")]))
# new_df.question_text = new_df.question_text.apply(lambda text: contractions.fix(text))# 
# new_df.question_text = new_df.question_text.apply(lambda text: ' '.join([word for word in text.split(' ') if word not in stopwords.words('english')]))
# new_df.question_text = new_df.question_text.apply(lambda text: text.replace("’", ' '))


# new_df.question_text = new_df.question_text.apply(lambda text: text.replace('“', ' '))
# new_df.question_text = new_df.question_text.apply(lambda text: re.sub(' +', ' ', text))
new_df.question_text = new_df.question_text.apply(lambda x : clean_data(x))

test_df.question_text = test_df.question_text.apply(lambda x : clean_data(x))
# new_df.to_pickle("training_undersampled_df.pkl")

# test_df.to_pickle("test_df.pkl")



# new_df = pd.read_pickle('training_undersampled_df.pkl')

# test_df = pd.read_pickle('test_df.pkl')
new_df.question_text = new_df.question_text.apply(lambda x: x.lower())

test_df.question_text = test_df.question_text.apply(lambda x: x.lower())
import zipfile

archive = zipfile.ZipFile('/kaggle/input/quora-insincere-questions-classification/embeddings.zip', 'r')

news_path=archive.open('GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', 'r')



# from gensim.models import KeyedVectors

# embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)



# del embeddings_index
def build_vocab(text):

    text = text.split(" ")

    for word in text:

        if word not in vocab:

            vocab[word] = 1

        else:

            vocab[word] += 1
import operator

def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    

    return unknown_words
vocab = {}

temp = new_df.question_text.apply(lambda x: build_vocab(x))

del temp
# oov = check_coverage(vocab, embeddings_index)

# del oov
# del oov
import io

from tqdm import tqdm

embeddings_glove={}



with zipfile.ZipFile("/kaggle/input/quora-insincere-questions-classification/embeddings.zip") as zf:

    with io.TextIOWrapper(zf.open("glove.840B.300d/glove.840B.300d.txt"), encoding="utf-8") as f:

        for line in tqdm(f):

            values=line.split(' ') # ".split(' ')" only for glove-840b-300d; for all other files, ".split()" works

            word=values[0]

            vectors=np.asarray(values[1:],'float32')

            embeddings_glove[word]=vectors
oov_glove = check_coverage(vocab, embeddings_glove)
oov_glove
import gc

gc.collect()
# count = 0

# for key, val in oov_glove:

#     try:

#         embeddings_glove[key] = embeddings_index[key]

#     except:

#         count += 1
# oov_glove = check_coverage(vocab, embeddings_glove)
# len(oov_glove), len(vocab)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

length_of_vocab = 80000

max_len = 150

def tokenize(train, test):

    tokenizer = Tokenizer(num_words = length_of_vocab)

    tokenizer.fit_on_texts(train)

    

    X = tokenizer.texts_to_sequences(train)

    X = pad_sequences(X, maxlen = max_len, padding ="post")

    

    X_test = tokenizer.texts_to_sequences(test)

    X_test = pad_sequences(X_test, maxlen = max_len, padding ="post")

    

    return X, X_test, tokenizer.word_index, tokenizer
X, X_test, word_index, tokenizer = tokenize(new_df.question_text, test_df.question_text)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, new_df.target, test_size=0.2)
def create_embeded_matrix(embeded_glove, word_index, length_of_vocab):

    embed = np.stack(embeded_glove.values())

    embed_mean, embed_std = embed.mean(), embed.std()

    embed_size = embed.shape[1]

    embeded_matrix = np.random.normal(embed_mean, embed_std, (length_of_vocab, embed_size))

    

    for word, index in word_index.items():

        if index >= length_of_vocab:

            continue

        embeded_vector = embeded_glove.get(word)

        if embeded_vector is not None:

            embeded_matrix[index] = embeded_vector

    return embeded_matrix
embed_matrix = create_embeded_matrix(embeddings_glove, word_index, length_of_vocab)
from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

from keras.models import Model

from keras.layers import Dense,LSTM, Dropout,Conv1D, Embedding, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from tensorflow.keras import regularizers

from keras.regularizers import l2, l1, l1_l2

# slim = tf.contrib.slim



inp    = Input(shape=(max_len,))

x      = Embedding(length_of_vocab, 300, weights=[embed_matrix], trainable=False)(inp)

sdrop = SpatialDropout1D(rate=0.4)(x)

b1 = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(0.01), activity_regularizer=l1_l2(0.01)))(sdrop)

conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(b1)

gmax1_p = GlobalAveragePooling1D()(conv1)

d1 = Dense(128, activation="relu")(gmax1_p)

drop = Dropout(0.5)(d1)

d2 = Dense(1, activation="sigmoid")(drop)

# b1 = keras.layers.BatchNormalization()

model  = Model(inputs=inp, outputs=d2)



model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.01), metrics=['accuracy', f1])
model.summary()
history = model.fit(X_train, Y_train, batch_size=256, epochs=5, verbose=1, validation_split=0.2)
model.evaluate(x = X_val, y = Y_val)
predict = model.predict(X_test)
predict = np.where(predict < 0.5, 0, 1)
test_df.head()
result = pd.DataFrame()

result['qid'] = test_df.qid

result['prediction'] = predict
result.head()
result.to_csv('submission.csv', index=False)