import os

import time

import numpy as np

import pandas as pd 

from tqdm import tqdm

from keras.engine.topology import Layer

import math

import operator 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, TimeDistributed, CuDNNLSTM,Conv2D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, Flatten, Reshape, AveragePooling2D, Average

from keras.models import Model

from keras.layers import Wrapper

from keras.models import Model

from keras.layers import Dense, Embedding, Bidirectional, CuDNNGRU, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dropout

from keras.optimizers import Adam

import keras.backend as K

import matplotlib as plt

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import initializers, regularizers, constraints, optimizers, layers

tqdm.pandas()

import pandas as pd

import numpy as np

import operator 

import re

import gc

import keras

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
train_df[train_df.target==1].head()
train_df[train_df.target==0].head()
train_df['target'].value_counts().plot(kind = 'pie', labels = ['no tóxica 0', 'tóxica 1 '],

     startangle = 90, autopct = '%1.0f%%')




def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    

    return embeddings_index



glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

#paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

#wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
#Glove

embed_glove = load_embed(glove)

#Paragram

#embed_paragram = load_embed(paragram)

#FastText

#embed_fasttext = load_embed(wiki_news)
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



    print('Se encontraron embeddings para {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    #print('Se encontraron embeddings para  {:.3%} of all text (palabras conocidas y desconocidas)'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
df = pd.concat([train_df ,test_df])

vocab = build_vocab(df['question_text'])
print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)
df['question_text'] = df['question_text'].apply(lambda x: x.lower())
def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Anadidas {count} palabras al embedding")
print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)

add_lower(embed_glove, vocab)

oov_glove = check_coverage(vocab, embed_glove)
oov_glove[:10]
contraction_mapping = {

    "ain't": "is not",

    "aren't": "are not",

    "can't": "cannot",

    "'cause": "because",

    "could've": "could have",

    "couldn't": "could not",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "he would",

    "he'll": "he will",

    "he's": "he is",

    "how'd": "how did",

    "how'd'y": "how do you",

    "how'll": "how will",

    "how's": "how is",

    "I'd": "I would",

    "I'd've": "I would have",

    "I'll": "I will",

    "I'll've": "I will have",

    "I'm": "I am",

    "I've": "I have",

    "i'd": "i would",

    "i'd've": "i would have",

    "i'll": "i will",

    "i'll've": "i will have",

    "i'm": "i am",

    "i've": "i have",

    "isn't": "is not",

    "it'd": "it would",

    "it'd've": "it would have",

    "it'll": "it will",

    "it'll've": "it will have",

    "it's": "it is",

    "let's": "let us",

    "ma'am": "madam",

    "mayn't": "may not",

    "might've": "might have",

    "mightn't": "might not",

    "mightn't've": "might not have",

    "must've": "must have",

    "mustn't": "must not",

    "mustn't've": "must not have",

    "needn't": "need not",

    "needn't've": "need not have",

    "o'clock": "of the clock",

    "oughtn't": "ought not",

    "oughtn't've": "ought not have",

    "shan't": "shall not",

    "sha'n't": "shall not",

    "shan't've": "shall not have",

    "she'd": "she would",

    "she'd've": "she would have",

    "she'll": "she will",

    "she'll've": "she will have",

    "she's": "she is",

    "should've": "should have",

    "shouldn't": "should not",

    "shouldn't've": "should not have",

    "so've": "so have",

    "so's": "so as",

    "this's": "this is",

    "that'd": "that would",

    "that'd've": "that would have",

    "that's": "that is",

    "there'd": "there would",

    "there'd've": "there would have",

    "there's": "there is",

    "here's": "here is",

    "they'd": "they would",

    "they'd've": "they would have",

    "they'll": "they will",

    "they'll've": "they will have",

    "they're": "they are",

    "they've": "they have",

    "to've": "to have",

    "wasn't": "was not",

    "we'd": "we would",

    "we'd've": "we would have",

    "we'll": "we will",

    "we'll've": "we will have",

    "we're": "we are",

    "we've": "we have",

    "weren't": "were not",

    "what'll": "what will",

    "what'll've": "what will have",

    "what're": "what are",

    "what's": "what is",

    "what've": "what have",

    "when's": "when is",

    "when've": "when have",

    "where'd": "where did",

    "where's": "where is",

    "where've": "where have",

    "who'll": "who will",

    "who'll've": "who will have",

    "who's": "who is",

    "who've": "who have",

    "why's": "why is",

    "why've": "why have",

    "will've": "will have",

    "won't": "will not",

    "won't've": "will not have",

    "would've": "would have",

    "wouldn't": "would not",

    "wouldn't've": "would not have",

    "y'all": "you all",

    "y'all'd": "you all would",

    "y'all'd've": "you all would have",

    "y'all're": "you all are",

    "y'all've": "you all have",

    "you'd": "you would",

    "you'd've": "you would have",

    "you'll": "you will",

    "you'll've": "you will have",

    "you're": "you are",

    "you've": "you have",

    "who'd": "who would",

    "who're": "who are",

    "'re": " are",

    "tryin'": "trying",

    "doesn'": "does not",

    'howdo': 'how do',

    'whatare': 'what are',

    'howcan': 'how can',

    'howmuch': 'how much',

    'howmany': 'how many',

    'whydo': 'why do',

    'doI': 'do I',

    'theBest': 'the best',

    'howdoes': 'how does',

}
def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known
print("- Contracciones Conocidas  -")

print("   Glove :")

print(known_contractions(embed_glove))
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
df['question_text'] = df['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
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
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
df['question_text'] = df['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
vocab = build_vocab(df['question_text'])

print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)
oov_glove[:100]
mispell_dict = {'advanatges': 'advantages', 

                'irrationaol': 'irrational' ,

                'defferences': 'differences',

                'lamboghini':'lamborghini',

                'hypothical':'hypothetical',

                'colour': 'color', 

                'centre': 'center', 

                'favourite': 'favorite',

                'travelling': 'traveling',

                'counselling': 'counseling', 

                'theatre': 'theater',

                'cancelled': 'canceled', 

                'labour': 'labor',

                'organisation': 'organization',

                'wwii': 'world war 2',

                'citicise': 'criticize', 

                'youtu ': 'youtube ', 

                'Qoura': 'Quora', 

                'sallary': 'salary',

                'Whta': 'What', 

                'narcisist': 'narcissist',

                'howdo': 'how do',

                'whatare': 'what are', 

                'howcan': 'how can',

                'howmuch': 'how much',

                'howmany': 'how many',

                'whydo': 'why do',

                'doI': 'do I',

                'theBest': 'the best',

                'howdoes': 'how does',

                'mastrubation': 'masturbation',

                'mastrubate': 'masturbate',

                "mastrubating": 'masturbating',

                'pennis': 'penis',

                'Etherium': 'Ethereum',

                'narcissit': 'narcissist',

                'bigdata': 'big data',

                '2k17': '2017',

                '2k18': '2018',

                'qouta': 'quota',

                'exboyfriend': 'ex boyfriend',

                'airhostess': 'air hostess',

                "whst": 'what',

                'watsapp': 'whatsapp', 

                'demonitisation': 'demonetization',

                'demonitization': 'demonetization',

                'demonetisation': 'demonetization',

                'pokémon': 'pokemon'}
def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
df['question_text'] = df['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))

vocab = build_vocab(df['question_text'])

print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)
def clean_numbers(x):



    x = re.sub('[0-9]{5,}', ' number ', x)

    x = re.sub('[0-9]{4}', ' number ', x)

    x = re.sub('[0-9]{3}', ' number ', x)

    x = re.sub('[0-9]{2}', ' number ', x)

    return x
df['question_text'] = df['question_text'].apply(lambda x: clean_numbers(x))
vocab = build_vocab(df['question_text'])

print("Glove : ")

oov_glove = check_coverage(vocab, embed_glove)
# Pasar a minusculas

train_df['treated_question'] = train_df['question_text'].apply(lambda x: x.lower())

# Eliminar Contracciones

train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

# Eliminar Caracteres Especiales

train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# Errores de Spelling

train_df['treated_question'] = train_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

#Eliminar Numeros

train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_numbers(x))
# Pasar a minusculas

test_df['treated_question'] = test_df['question_text'].apply(lambda x: x.lower())

# Eliminar Contracciones

test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_contractions(x, contraction_mapping))

# Eliminar Caracteres Especiales

test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# Errores de Spelling

test_df['treated_question'] = test_df['treated_question'].apply(lambda x: correct_spelling(x, mispell_dict))

#Eliminar Numeros

test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_numbers(x))
'''def add_features(df):

    

    df['question_text'] = df['question_text'].progress_apply(lambda x:str(x))

    df['total_length'] = df['question_text'].progress_apply(len)

    df['capitals'] = df['question_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))

    df['caps_vs_length'] = df.progress_apply(lambda row: float(row['capitals'])/float(row['total_length']),

                                axis=1)

    df['num_words'] = df.question_text.str.count('\S+')

    df['num_unique_words'] = df['question_text'].progress_apply(lambda comment: len(set(w for w in comment.split())))

    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']  



    return df'''
#train_df = add_features(train_df)

#test_df = add_features(test_df)
train, val = train_test_split(train_df, test_size=0.2, random_state=2)

xtrain = train['question_text'].fillna('_na_').values

xval = val['question_text'].fillna('_na_').values

xtest = test_df['question_text'].fillna('_na_').values
# Tamaño del vector embedding

EMBED_SIZE = 300

# Palabras únicas

MAX_FEATURES = 100000 

# Longitud máxima de la frase

MAXLEN = 60 



#Metodo para decirle de cuantas palabras va a truncar el diccionario al Tokenizar

tokenizer = Tokenizer(num_words=MAX_FEATURES)



#Metodo para contar la frecuencia de palabras y asignarle el indice segun la frecuencia 

tokenizer.fit_on_texts(list(xtrain))



xtrain = tokenizer.texts_to_sequences(xtrain)

xval = tokenizer.texts_to_sequences(xval)

xtest = tokenizer.texts_to_sequences(xtest)
xtrain = pad_sequences(xtrain, maxlen=MAXLEN)

xval = pad_sequences(xval, maxlen=MAXLEN)

xtest = pad_sequences(xtest, maxlen=MAXLEN)
ytrain = train['target'].values

yval = val['target'].values
def load_glove_matrix(word_index, embeddings_index):



    all_embs = np.stack(embeddings_index.values())

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    EMBED_SIZE = all_embs.shape[1]

    

    nb_words = min(MAX_FEATURES, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))



    for word, i in word_index.items():

        if i >= MAX_FEATURES:

            continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector



    return embedding_matrix

np.random.seed(2)



trn_idx = np.random.permutation(len(xtrain))

val_idx = np.random.permutation(len(xval))



xtrain = xtrain[trn_idx]

ytrain = ytrain[trn_idx]

xval = xval[val_idx]

yval = yval[val_idx]



embedding_matrix_glove = load_glove_matrix(tokenizer.word_index, embed_glove)
class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)

        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0], self.features_dim

def f1(y_true, y_pred):



    def recall(y_true, y_pred):

        

        true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives/(possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        

        true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives/(predicted_positives + K.epsilon())

        return precision



    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)



    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def f1(y_true, y_pred):



    def recall(y_true, y_pred):

        

        true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives/(possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        

        true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives/(predicted_positives + K.epsilon())

        return precision



    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)



    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def model_lstm_att(embedding_matrix):

    

    inp = Input(shape=(MAXLEN,))

    x = Embedding(MAX_FEATURES, EMBED_SIZE, weights=[embedding_matrix], trainable=False)(inp)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)

    

    att = Attention(MAXLEN)(x)

    

    y = Dense(32, activation='relu')(att)

    #y = Dense(32, activation='relu')(y)

    

    y = Dropout(0.1)(y)

    outp = Dense(1, activation='sigmoid')(y)    



    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy', 'accuracy'])

    

    return model
paragram = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

embedding_matrix_para = load_glove_matrix(tokenizer.word_index, load_embed(paragram))
embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_para], axis=0)
model_lstm = model_lstm_att(embedding_matrix)

model_lstm.summary()
'''checkpoints = ModelCheckpoint('weights.hdf5', monitor="val_f1", mode="max", verbose=True, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)'''
'''epochs = 8

batch_size = 512'''
'''history = model_lstm.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, 

                    validation_data=[xval, yval], callbacks=[checkpoints, reduce_lr])'''
def train_pred(model, epochs=2):

    

    for e in range(epochs):

        model.fit(xtrain, ytrain, batch_size=512, epochs=3, validation_data=(xval, yval))

        pred_val_y = model.predict([xval], batch_size=1024, verbose=1)



        best_thresh = 0.5

        best_score = 0.0

        for thresh in np.arange(0.1, 0.501, 0.01):

            thresh = np.round(thresh, 2)

            score = metrics.f1_score(yval, (pred_val_y > thresh).astype(int))

            if score > best_score:

                best_thresh = thresh

                best_score = score



        print("Val puntuacion F1: {:.4f}".format(best_score))



    pred_test_y = model.predict([xtest], batch_size=1024, verbose=1)



    return pred_val_y, pred_test_y, best_score
outputs = []

pred_val_y, pred_test_y, best_score = train_pred(model_lstm, epochs=2)

outputs.append([pred_val_y, pred_test_y, best_score, 'model_lstm_att only Glove'])
'''plt.figure(figsize=(12,8))

plt.plot(history.history['acc'], label='Entrenamiento Accuracy')

plt.plot(history.history['val_acc'], label='Validacion Accuracy')

plt.show()'''
outputs.sort(key=lambda x: x[2]) 

weights = [i for i in range(1, len(outputs) + 1)]

weights = [float(i) / sum(weights) for i in weights] 



pred_val_y = np.mean([outputs[i][0] for i in range(len(outputs))], axis = 0)



thresholds = []

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    res = metrics.f1_score(yval, (pred_val_y > thresh).astype(int))

    thresholds.append([thresh, res])

    print("La puntuación F1 en los limites {0} y {1}".format(thresh, res))

    

thresholds.sort(key=lambda x: x[1], reverse=True)

best_thresh = thresholds[0][0]
print("Mejor limite:", best_thresh, "y puntuacion F1 ", thresholds[0][1])
pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis = 0)

pred_test_y = (pred_test_y > best_thresh).astype(int)
sub = pd.read_csv('../input/sample_submission.csv')

out_df = pd.DataFrame({"qid":sub["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)