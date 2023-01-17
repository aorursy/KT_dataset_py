# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import operator 

from sklearn import metrics



import re

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from sklearn.preprocessing import LabelEncoder

from gensim.models import KeyedVectors

print(os.listdir("../input"))

print(os.listdir("../input/embeddings"))

print(os.listdir("../input/embeddings/glove.840B.300d"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
print(train_df.shape)

print(test_df.shape)
train_df.groupby('target').size()
train1_df = train_df[train_df["target"]==1]

train0_df = train_df[train_df["target"]==0]
from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=300, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='brown',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 0,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  
plot_wordcloud(train1_df["question_text"], title="Word Cloud of Insincere Questions")
plot_wordcloud(train0_df["question_text"], title="Word Cloud of Sincere Questions")
def word_freq(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
sentences = train_df["question_text"].apply(lambda x: x.split())

vocab = word_freq(sentences)
embedd_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embeddings_index = KeyedVectors.load_word2vec_format(embedd_path, binary=True)
def get_embeddings_coverage(vocab,embeddings_index):

    vocab_embed = {}

    out_of_vocab= {}

    k = 0

    i = 0

    for word in vocab:

        try:

            vocab_embed[word] = embeddings_index[word]

            k += vocab[word]

        except:

            out_of_vocab[word] = vocab[word]

            i += vocab[word]

            pass

    print('found word embeddings for {:.2%} of vocab'.format(len(vocab_embed) / len(vocab)))

    print('found word embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_oov = sorted(out_of_vocab.items(), key=operator.itemgetter(1))[::-1]



    return vocab_embed, sorted_oov
vocab_embed, oov = get_embeddings_coverage(vocab,embeddings_index)
def remove_punctuation(sentence):

    """

    Utility function to remove punctuations from sentence text using simple regex statements..

    """

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        sentence = sentence.replace(punct, '')

        

    return sentence
train_df["question_text"] = train_df["question_text"].apply(lambda x: remove_punctuation(x))

sentences = train_df["question_text"].apply(lambda x: x.split())

vocab = word_freq(sentences)
vocab_embed, oov = get_embeddings_coverage(vocab,embeddings_index)
def clean_numbers(x):

    """

    Utility function to format the numbers in the sentences

    """

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x
train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))

sentences = train_df["question_text"].apply(lambda x: x.split())

vocab = word_freq(sentences)
vocab_embed, oov = get_embeddings_coverage(vocab,embeddings_index)
#thanks to https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings

def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re





mispell_dict = {'colour':'color',

                'centre':'center',

                'didnt':'did not',

                'doesnt':'does not',

                'isnt':'is not',

                'shouldnt':'should not',

                'favourite':'favorite',

                'travelling':'traveling',

                'counselling':'counseling',

                'theatre':'theater',

                'cancelled':'canceled',

                'labour':'labor',

                'organisation':'organization',

                'wwii':'world war 2',

                'citicise':'criticize',

                'instagram': 'social medium',

                'whatsapp': 'social medium',

                'snapchat': 'social medium'



                }

mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)
train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))

sentences = train_df["question_text"].apply(lambda x: x.split())

to_remove = ['a','to','of','and']

sentences = [[word for word in sentence if not word in to_remove] for sentence in sentences]

vocab = word_freq(sentences)
vocab_embed, oov = get_embeddings_coverage(vocab,embeddings_index)
# some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use
## split to train and val

from sklearn.model_selection import train_test_split



train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=32)

train_x = train_df['question_text']

val_x = val_df['question_text']

test_x = test_df['question_text']
## Tokenize the sentences

from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)

val_x = tokenizer.texts_to_sequences(val_x)

test_x = tokenizer.texts_to_sequences(test_x)
## Pad the sentences 

from keras.preprocessing.sequence import pad_sequences



train_x = pad_sequences(train_x, maxlen=maxlen)

val_x = pad_sequences(val_x, maxlen=maxlen)

test_x = pad_sequences(test_x, maxlen=maxlen)
## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0

for word, i in word_index.items():

    if i >= max_features: continue

    if word in embeddings_index:

        embedding_vector = embeddings_index.get_vector(word)

        embedding_matrix[i] = embedding_vector
embedding_matrix.shape
from keras.models import Sequential

from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from keras.models import Model

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

from keras.models import Model
model = Sequential()

model.add(Embedding(max_features, embed_size, input_length=maxlen)) # weights=[embedding_matrix]))

model.add(Conv1D(64, 5, activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(50))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 

                    optimizer='adam', 

                    metrics=['accuracy'])

model.summary()
model.fit(train_x, train_y, batch_size=512, epochs=2, validation_data=(val_x, val_y))
model.evaluate(val_x, val_y, batch_size=128)
val_y.shape
pred_y = model.predict([val_x], batch_size=512, verbose=1)
pred_y = model.predict([val_x], batch_size=512, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_y>thresh).astype(int))))
model = Sequential()

model.add(Embedding(max_features, embed_size, input_length=maxlen, weights=[embedding_matrix]))

model.add(Conv1D(64, 5, activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPooling1D(pool_size=4))

model.add(LSTM(50))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 

                    optimizer='adam', 

                    metrics=['accuracy'])

model.summary()
del model
model.fit(train_x, train_y, batch_size=512, epochs=1, validation_data=(val_x, val_y))
pred_y = model.predict([val_x], batch_size=512, verbose=1)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_y>thresh).astype(int))))