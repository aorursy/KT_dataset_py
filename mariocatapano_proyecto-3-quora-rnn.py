from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

#import tensorflow as tf

import os

import numpy as np 

import pandas as pd

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from keras.models import Model,Sequential

from keras.layers import  Dense, Bidirectional,LSTM,Dropout

from keras.optimizers import Adam

tqdm.pandas()

from gensim.models import KeyedVectors

import operator

import gc

import matplotlib.pyplot as plt


for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train=pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test=pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

trainx=train.drop(['qid','target'],axis=1)

trainy=train['target']

testx=train.drop(['qid'],axis=1)
def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
sentences = train["question_text"].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

print({k: vocab[k] for k in list(vocab)[:8]})
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    if file == '../input/quora-insincere-questions-classification/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
embed_glove = load_embed('../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')
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
print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)
oov_glove[:10]
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
def unknown_punct(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown
print("Glove :")

print(unknown_punct(embed_glove, punct))
def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x
train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))

sentences = train["question_text"].apply(lambda x: x.split())

vocab = build_vocab(sentences)
oov_glove = check_coverage(vocab,embed_glove)
oov_glove[:50]
train['lowered_question'] = train['question_text'].apply(lambda x: x.lower())
def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")
print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)

add_lower(embed_glove, vocab)

oov_glove = check_coverage(vocab, embed_glove)
oov_glove = check_coverage(vocab,embed_glove)
del embed_glove

del sentences

del vocab

gc.collect()
embeddings_index = {}

f = open('../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')

for line in tqdm(f):

    values = line.split(" ")

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
train_df,validation_df= train_test_split(train, test_size=0.1)
train_df['treated_question'] = train_df['question_text'].apply(lambda x: x.lower())

validation_df['treated_question'] = validation_df['question_text'].apply(lambda x: x.lower())
train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_text(x))

validation_df['treated_question'] = validation_df['treated_question'].apply(lambda x: clean_text(x))
y = train['target'].values
def text_to_array(text):

    empyt_emb = np.zeros(300)

    text = text[:-1].split()[:30]

    embeds = [embeddings_index.get(x, empyt_emb) for x in text]

    embeds+= [empyt_emb] * (30 - len(embeds))

    return np.array(embeds)



val_vects = np.array([text_to_array(X_text) for X_text in tqdm(validation_df["treated_question"][:3000])])

val_y = np.array(validation_df["target"][:3000])
batch_size = 128



def batch_gen(train_df):

    n_batches = math.ceil(len(train_df) / batch_size)

    while True: 

        train_df = train_df.sample(frac=1.)  # Shuffle the data.

        for i in range(n_batches):

            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]

            text_arr = np.array([text_to_array(text) for text in texts])

            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])
model = Sequential()

model.add(Bidirectional(LSTM(64, return_sequences=True),

                        input_shape=(30, 300)))

model.add(Bidirectional(LSTM(64)))

model.add(Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
mg = batch_gen(train_df)

history=model.fit_generator(mg, epochs=10,

                    steps_per_epoch=1000,

                    validation_data=(val_vects, val_y),

                    verbose=True)
plt.figure(figsize=(12,8))

plt.plot(history.history['accuracy'], label='Train Accuracy')

plt.plot(history.history['val_accuracy'], label='Test Accuracy')

plt.show()
batch_size = 256

def batch_gen(test_df):

    n_batches = math.ceil(len(test_df) / batch_size)

    for i in range(n_batches):

        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]

        text_arr = np.array([text_to_array(text) for text in texts])

        yield text_arr



test_df = test



all_preds = []

for x in tqdm(batch_gen(test_df)):

    all_preds.extend(model.predict(x).flatten())
y_te = (np.array(all_preds) > 0.5).astype(np.int)



submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})

submit_df.to_csv("submission.csv", index=False)