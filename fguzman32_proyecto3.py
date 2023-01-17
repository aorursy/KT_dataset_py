import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from keras.engine.topology import Layer

import math

import operator 

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, TimeDistributed, CuDNNLSTM,Conv2D, SpatialDropout1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, concatenate, Flatten, Reshape, AveragePooling2D, Average, BatchNormalization

from keras.models import Model

#from keras.layers import Wrapper

import keras.backend as K

from keras.optimizers import Adam

from keras import initializers, regularizers, constraints, optimizers, layers

import re

import gc

from sklearn.preprocessing import StandardScaler

tqdm.pandas()

 

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

import tensorflow as tf

import re

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test_df = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')



print("Train shape: ",train_df.shape)

print("Test shape: ",test_df.shape)
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",

                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",

                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                       "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                       "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have",

                       "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",

                       "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",

                       "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",

                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",

                       "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",

                       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would",

                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}



contraction_patterns = [ (r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),

                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'),

                        (r'dont', 'do not'), (r'wont', 'will not') ]



punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



mispell_dict = {'advanatges': 'advantages', 'irrationaol': 'irrational' , 'defferences': 'differences','lamboghini':'lamborghini','hypothical':'hypothetical', 'colour': 'color',

                'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'qoura' : 'quora', 'sallary': 'salary', 'Whta': 'What',

                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do',

                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating',

                'pennis': 'penis', 'Etherium': 'Ethereum', 'etherium': 'ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota',

                'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',

                'demonitization': 'demonetization', 'demonetisation': 'demonetization', 'pokémon': 'pokemon'}
#metodo para sustituir las diferentes tildes por '

def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

        #si t esta en mapping agregar la nueva sustitucion

        #si no dejarla igual

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text

#replace the constractions in text

def replaceContraction(text):

    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]

    for (pattern, repl) in patterns:

        (text, count) = re.subn(pattern, repl, text)

    return text



#clean text

#reemplazando caracteres 

def clean_text(x):

    x = str(x)

    #se reemplazan estos caracteres por un espacio

    for punct in "/-'":

        x = x.replace(punct, ' ')

    #si un & esta pegado a una palabra se agrega un espacio

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    #cualquiera de estos simbolos que se encuentre en la data

    #sera reemplazado por ''

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x



#numeros mayores a 2 digitos se les coloca e nombre de 'number'

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', ' number ', x)

    x = re.sub('[0-9]{4}', ' number ', x)

    x = re.sub('[0-9]{3}', ' number ', x)

    x = re.sub('[0-9]{2}', ' number ', x)

    return x







#replace special characters

def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    for p in punct:

        text = text.replace(p, f' {p} ')

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters

    for s in specials:

        text = text.replace(s, specials[s])

    return text





#por cada palabra mal escrita se reemplaza por la palabra bien escrita

#ese dictionario esta en misspell_dict

def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
glove = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'

embed_glove = load_embed(glove)
len(embed_glove)
# poner en minusculas

train_df['treated_question'] = train_df['question_text'].progress_apply(lambda x: x.lower())

# aplicar contracciones

train_df['treated_question'] = train_df['treated_question'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

# special characters

train_df['treated_question'] = train_df['treated_question'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# aplicar funcion correct_spelling

train_df['treated_question'] = train_df['treated_question'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

#aplicar funcion clean_numbers

train_df['treated_question'] = train_df['treated_question'].apply(lambda x: clean_numbers(x))





# poner en minuscula

test_df['treated_question'] = test_df['question_text'].progress_apply(lambda x: x.lower())

# Contracciones

test_df['treated_question'] = test_df['treated_question'].progress_apply(lambda x: clean_contractions(x, contraction_mapping))

# special characters

test_df['treated_question'] = test_df['treated_question'].progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

# aplicar funcion correct_spelling

test_df['treated_question'] = test_df['treated_question'].progress_apply(lambda x: correct_spelling(x, mispell_dict))

test_df['treated_question'] = test_df['treated_question'].apply(lambda x: clean_numbers(x))
train, val = train_test_split(train_df, test_size=0.2, random_state=3)
xtrain = train['question_text'].fillna('_na_').values

xval = val['question_text'].fillna('_na_').values

xtest = test_df['question_text'].fillna('_na_').values
print(xtrain)
#numero de dimensiones del embedding

embed_size = 300

#numero de palabras en diccionario

max_features = 10000

#maximo numero de palabras que se analizaran

maxlen = 80





#e diccionario va a tener 10.000 palabras

#se tendran 10.000 guardadas basado en la frecuencia en que ocurren

#solamente se guardaran las 10.000 palabras mas frecuentes

tokenizer = Tokenizer(num_words=max_features)



tokenizer.fit_on_texts(list(xtrain))



xtrain = tokenizer.texts_to_sequences(xtrain)

xval = tokenizer.texts_to_sequences(xval)

xtest = tokenizer.texts_to_sequences(xtest)


xtrain = pad_sequences(xtrain, maxlen=maxlen)

xval = pad_sequences(xval,maxlen=maxlen)

xtest = pad_sequences(xtest,maxlen=maxlen)
ytrain = train['target'].values

yval = val['target'].values
def load_glove_matrix(word_index, embeddings_index):



    all_embs = np.stack(embeddings_index.values())

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    EMBED_SIZE = all_embs.shape[1]

    

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



    for word, i in word_index.items():

        if i >= max_features:

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
def model():

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix_glove], trainable=True)(inp)

    x = SpatialDropout1D(0.15)(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)

    x = Conv1D(filters=64, kernel_size=1)(x)

    x = GlobalMaxPool1D()(x)

    x_f = Dense(128, activation="relu")(x)

    x_f = Dropout(0.15)(x_f)

    x_f = BatchNormalization()(x_f)

    x_f = Dense(1, activation="sigmoid")(x_f)

    

    model = Model(inputs=inp, outputs = x_f)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0010), metrics=['binary_accuracy'])

    return model
model1 = model()

print(model1.summary())
model1.fit(xtrain,ytrain, batch_size = 512, epochs= 2, validation_data=(xval,yval))
prediction = model1.predict(xtest, batch_size=1024)
out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = prediction

out_df.to_csv('submission.csv', index=False)