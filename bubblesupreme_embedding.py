import os

print(os.listdir("../input"))
import pandas as pd

import numpy as np

import tensorflow as tf

import re, csv, string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional, GlobalMaxPool1D, Bidirectional

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

import gensim

import gensim.models.keyedvectors as word2vec

import nltk.data

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer

from nltk.stem.lancaster import LancasterStemmer

import heapq

from sklearn.model_selection import KFold, cross_val_score, train_test_split

import gc
train = pd.read_csv('../input/bad-comments/train.csv')

train.head(5)
train.isnull().any()
train = train[:10000]
train.drop(['id'], axis=1).describe()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)
def loadStemmer(stemmer):

    choice = {'porter': PorterStemmer, 'lancaster': LancasterStemmer, 'snowball': SnowballStemmer}

    return choice[stemmer]()

contractions = {"isn't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def replace_contraction(match):

    return contractions[match.group(0)]
STOP_WORDS = set(stopwords.words('english'))
def clean_data():

    for i in range(train.shape[0]):

        text = str(train.loc[i,'comment_text'])

        text=text.lower()

        text = contractions_re.sub(replace_contraction, text)

        text = re.sub('[^A-Za-z]', ' ', text)

        tokens = word_tokenize(text)

        tokens = [ word for word in tokens if word not in STOP_WORDS ]

        train.loc[i,'comment_text'] = ' '.join(tokens)
clean_data()
model_google_news = gensim.models.KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', 

                                                        binary=True)
words = model_google_news.index2word
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
vocab_train = build_vocab(train.comment_text)



WORDS = {}

for i,word in enumerate(words):

    WORDS[word] = i
def words(text): return re.findall(r'\w+', text)



def frequency(word):

    return - WORDS.get(word, 0)



def correction(word):

    # только для слов, длина которых больше 1, для других нет смысла вызывать алгоритм

    if len(word)>1:

        return max(candidates(word), key=frequency)

    return word



def candidates(word):

    return (known([word]) or known(edits(word)) or [word])



def known(words):

    return set(w for w in words if w in WORDS)



def edits(word):

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + replaces + inserts)
corrected_words_train = {}

for word in vocab_train:

    corrected_word = correction(word)

    if corrected_word != word:

        corrected_words_train[word] = corrected_word
print('humour',corrected_words_train['humour'])
print('centres',corrected_words_train['centres'])
print('lner',corrected_words_train['lner'])
print('wulf',corrected_words_train['wulf'])
def replace_misspell(text, mispell_dict ):

    text = str(text)

    for mispell in mispell_dict.keys():

        text=text.replace(mispell, mispell_dict[mispell])

    return text
stemmer = loadStemmer('lancaster')

lemmer = WordNetLemmatizer()
def remove_misspellings_and_normalize():

    for i in range(train.shape[0]):

        text = str(train.loc[i,'comment_text'])

        text = replace_misspell(text, corrected_words_train)

        tokens = word_tokenize(text)

        tokens = [stemmer.stem(w) for w in tokens]    

        tokens = [lemmer.lemmatize(w) for w in tokens]

        train.loc[i,'comment_text'] = ' '.join(tokens)
remove_misspellings_and_normalize()
y_train = train[list_classes].values

x_train = train[COMMENT]

max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(x_train))

x_train_seq = tokenizer.texts_to_sequences(x_train)
len(vocab_train.keys())
maxlen = 300

x_train_seq = pad_sequences(x_train_seq, maxlen=maxlen)
emb_size = 300
def creat_embedding_matrix(emb_type, emb_size, dict_model=None):

        emb_size_real = emb_size

        if emb_type=="glove":

            EMBEDDING_FILE='../input/glovetwitter27b100dtxt/glove.twitter.27B.200d.txt'

        elif emb_type=="word2vec":

            word2vecDict = model_google_news

        elif emb_type=="fasttext":

            EMBEDDING_FILE='../input/fasttext/wiki.simple.vec'



        emb_index = {}

        if emb_type=="glove" or emb_type=="fasttext":

            emb_file = open(EMBEDDING_FILE, encoding='utf-8')

            for line in emb_file:

                values = line.split()

                word=values[0]

                counter=1

                # может встретиться словосочетание, поэтому проверяем до первого флота

                while True:

                    try:

                        test= float(values[counter])

                        break

                    except ValueError:

                        word+=f' {values[counter]}'

                        counter+=1

                        pass

                word = values[0]

                # берем только те слова, которые есть в наших текстах

                if word in vocab_train.keys():

                    size_coefs = len(values[counter:emb_size+counter])

                    # проверяем совпадает ли запрошенная размерность с реальной

                    # если запрошенная больше, то печатаем ошибку и делаем матрицу максимально возможного размера

                    if size_coefs < emb_size:

                        emb_size_real = size_coefs

                        print(f'Warning!\nDesired size {emb_size} is not possible, returned size {emb_size_real}')

                        emb_size = emb_size_real

                    coefs = np.asarray(values[counter:emb_size+counter], dtype='float32')

                    emb_index[word] = coefs

            emb_file.close()

            print(f'Loaded {len(emb_index)} word vectors.')

        elif emb_type=="word2vec":

            for word in word2vecDict.wv.vocab:

                # берем только те слова, которые есть в наших текстах

                if word in vocab_train.keys():

                    size_coefs = word2vecDict.word_vec(word).shape[0]

                    if size_coefs < emb_size:

                        emb_size_real = size_coefs

                        print(f'Warning!\nDesired size {emb_size} is not possible, returned size {emb_size_real}')

                        emb_size = emb_size_real

                    coefs = word2vecDict.word_vec(word)[:emb_size+1]

                    emb_index[word] = coefs

            print(f'Loaded {len(emb_index)} word vectors.')

        elif emb_type=="own":

            if dict_model is None:

                print(f"Dict was not received!")

                return None, None

            for word in dict_model.keys():

                    size_coefs = dict_model[word].shape[0]

                    # проверяем совпадает ли запрошенная размерность с реальной

                    # если запрошенная больше, то печатаем ошибку и делаем матрицу максимально возможного размера

                    if size_coefs < emb_size:

                        emb_size_real = size_coefs

                        print(f'Warning!\nDesired size {emb_size} is not possible, returned size {emb_size_real}')

                        emb_size = emb_size_real

                    coefs = dict_model[word][:emb_size+1]

                    emb_index[word] = coefs

            print(f'Loaded {len(emb_index)} word vectors.')

        else:

            print(f"Wrong type {emb_type}!")

            return None, None

        

        all_embeddings = np.stack(list(emb_index.values()))

        emb_mean, emb_std = all_embeddings.mean(), all_embeddings.std()

        

        #предварительно создаем матрицу и забиваем ее средними значениями

        emb_matrix = np.random.normal(emb_mean, emb_std, (len(vocab_train.keys()), emb_size_real))

        gc.collect()



        # из словаря делаем матрицу

        counter = 0

        for word in vocab_train.keys():

            try:

                # если слово есть в нашем словаре, то берем оттуда вектор и записываем его в нашу матрицу

                embedding_vector=emb_index[word]

                emb_matrix[counter] = embedding_vector

            except KeyError:

                # если слова не оказалось- оставляем среднее значение

                pass

            counter+=1

        print(f'total embedded: {len(emb_matrix)} common words')

        

        del(emb_index)

        gc.collect()

        

        return emb_matrix, emb_size_real
embedding_matrix_w2v, emb_size = creat_embedding_matrix('word2vec',emb_size)

def get_emb_model(emb_matrix):

    inp = Input(shape=(maxlen, ))

    x = Embedding(emb_matrix.shape[0], emb_matrix.shape[1],weights=[emb_matrix],trainable=False)(inp)

    x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.1)(x)

    x = Dense(50, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])

    return model
model_w2v = get_emb_model(embedding_matrix_w2v)
model_w2v.summary()
batch_size = 32

epochs = 4
fold = KFold(n_splits=2, shuffle=True)

for train_index, test_index in fold.split(train):

    train_data = x_train_seq[train_index]

    test_data = x_train_seq[test_index]

    train_label = y_train[train_index]

    test_label = y_train[test_index]

    hist = model_w2v.fit(train_data,train_label, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    test_score = model_w2v.evaluate(test_data, test_label)

    print('Test loss {:.4f}, accuracy {:.2f}%'.format(test_score[0], test_score[1] * 100))
emb_size=200

embedding_matrix_glove, emb_size = creat_embedding_matrix('glove',emb_size)
model = get_emb_model(embedding_matrix_glove)
fold = KFold(n_splits=2, shuffle=True)

for train_index, test_index in fold.split(train):

    train_data = x_train_seq[train_index]

    test_data = x_train_seq[test_index]

    train_label = y_train[train_index]

    test_label = y_train[test_index]

    hist = model.fit(train_data,train_label, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    test_score = model.evaluate(test_data, test_label)

    print('Test loss {:.4f}, accuracy {:.2f}%'.format(test_score[0], test_score[1] * 100))
emb_size = 300

embedding_matrix_wiki, emb_size = creat_embedding_matrix('fasttext',emb_size)
model_wiki=get_emb_model(embedding_matrix_wiki)
fold = KFold(n_splits=2, shuffle=True)

for train_index, test_index in fold.split(train):

    train_data = x_train_seq[train_index]

    test_data = x_train_seq[test_index]

    train_label = y_train[train_index]

    test_label = y_train[test_index]

    hist = model_wiki.fit(train_data,train_label, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    test_score = model_wiki.evaluate(test_data, test_label)

    print('Test loss {:.4f}, accuracy {:.2f}%'.format(test_score[0], test_score[1] * 100))
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

train_words=[]

for i in range(train.shape[0]):

    train_words.append(train.loc[i][COMMENT].split())
train_words[0]
emb_size = 300

model_w2v = gensim.models.Word2Vec(size=emb_size, window=5, workers=4,negative=100)
model_w2v.build_vocab(train_words)
len(model_w2v.wv.vocab)
model_w2v.train(train_words, total_examples=model_w2v.corpus_count, epochs=model_w2v.iter)
w2v = dict(zip(model_w2v.wv.index2word, model_w2v.wv.vectors))
embedding_matrix_own, emb_size = creat_embedding_matrix('own', emb_size, w2v)
model_own_emb = get_emb_model(embedding_matrix_own)
fold = KFold(n_splits=2, shuffle=True)

for train_index, test_index in fold.split(train):

    train_data = x_train_seq[train_index]

    test_data = x_train_seq[test_index]

    train_label = y_train[train_index]

    test_label = y_train[test_index]

    hist = model_own_emb.fit(train_data,train_label, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    test_score = model_own_emb.evaluate(test_data, test_label)

    print('Test loss {:.4f}, accuracy {:.2f}%'.format(test_score[0], test_score[1] * 100))