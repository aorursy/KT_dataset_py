# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import operator

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from keras import backend as K, initializers, regularizers, constraints, optimizers, layers

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, concatenate

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.optimizers import Adam

from keras.models import Model

from keras.engine.topology import Layer     



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Carga de los datasets de la competencia a nuestro kernel 

train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
# Este es el tamaño del vector embedding 

word_vector = 300 

# Cantidad de palabras a tomar para matriz de embeddings  

max_features = 100000 

 # Tamaño maximo de cada pregunta 

size_question = 40
# Recorre el parametro "texto" ingresado a la función para identificar cada palabra y su respectivo numero de ocurrencias a lo largo de texto

# En otras palabras, se nos dara un diccionario de las palabras



def crear_vocabulario(texto):  

    oraciones = texto.apply(lambda x: x.split()).values 

    vocabulario = {}

    

    for oracion in oraciones:

        for palabra in oracion: 

            try:

                vocabulario[palabra] += 1

            except KeyError:

                vocabulario[palabra] = 1

    return vocabulario





df = pd.concat([train ,test], sort=False)



vocabulario = crear_vocabulario(df['question_text'])
# Esta funcion toma el parametro "archivo" para la creacion de un embedding_index

# Embedding index: diccionario donde las claves son los embeddings y  los valores son vectores que contienen  sus respectiva representacion como embedding



def cargar_embeddings(archivo):

    

    def obten_coeficientes(palabra,*arr): 

        return palabra, np.asarray(arr, dtype='float32')

    

    embeddings_index = dict(obten_coeficientes(*o.split(" ")) for o in open(archivo, encoding='latin'))

    

    return embeddings_index



# Cargamos los embeddings de una de las fuentes dadas por la competencia. Se ha seleccionado trabajar bajo un embedding GloVe

src_glove = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'



embedding_glove = cargar_embeddings(src_glove)



len(embedding_glove)
# Creamos una matriz llamada "embedding_matrix" que contenga dentro de ella todos los embeddings



def cargar_matriz_glove(word_index, embeddings_index):



    embeddings = np.stack(embeddings_index.values())

    emb_mean, emb_std = embeddings.mean(), embeddings.std()

    word_vector = embeddings.shape[1]

    

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, word_vector))



    for palabra, i in word_index.items():

        if i >= word_vector:

            continue

        embedding_vector = embeddings_index.get(palabra)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector



    return embedding_matrix
# El porcentaje que se imprimirá hace referencia a la extensión que nuestros embeddings cubren el vocabulario 

# Asimismo, se imprimiran las palabras del vocabulario con un mayor numero de ocurrencias a lo largo  del  texto



def check_coverage(vocabulario, embeddings_index):

    palabra_conocida = {}

    palabra_desconocida = {}

    nb_palabra_conocida = 0

    nb_palabra_desconocida = 0

    for palabra in vocabulario.keys():

        try:

            palabra_conocida[palabra] = embeddings_index[palabra]

            nb_palabra_conocida += vocabulario[palabra]

        except:

            palabra_desconocida[palabra] = vocabulario[palabra]

            nb_palabra_desconocida += vocabulario[palabra]

            pass

    

    

    print('Se encontraron embeddings para el {:.3%} del vocabulario'.format(len(palabra_conocida)/len(vocabulario)))

    print('Se encontraron embeddings para el {:.3%} de todo el cuerpo de texto'.format(nb_palabra_conocida/(nb_palabra_conocida + nb_palabra_desconocida)))

    palabra_desconocida = sorted(palabra_desconocida.items(), key=operator.itemgetter(1))[::-1]



    return palabra_desconocida
palabra_desconocida = check_coverage(vocabulario, embedding_glove)



#Para mejorar el modelo se observaran que palabras no estan en el diccionario

palabra_desconocida[:20]
# En el print realizado anteriormente, se ve que se toman en cuenta muchos factores que diferencian las  palabras entre si. Como la presencia de mayusculas en la palabra

# Función para cambiar las palabras mayuscular a minusculas en el texto



def cambiar_mayusculas(embedding, vocabulario):

    count = 0

    for palabra in vocabulario:

        if palabra in embedding and palabra.lower() not in embedding:  

            embedding[palabra.lower()] = embedding[palabra]

            count += 1

    print(f"Se agregaron {count} palabras al embedding")

    

#se cambia todo a minusculas

train['question_text'] = train['question_text'].apply(lambda x: x.lower())

test['question_text'] = test['question_text'].apply(lambda x: x.lower())
# Antes de quitar mayúsculas

palabra_desconocida = check_coverage(vocabulario, embedding_glove)

# Después de quitar mayúsculas

cambiar_mayusculas(embedding_glove, vocabulario) 

palabra_desconocida = check_coverage(vocabulario, embedding_glove)
# Definimos las contracciones que se pueden encontrar en  el texto y sucesivamente el correspondiente reemplazo 

lista_contracciones = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",

                       "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",

                       "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",

                       "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",

                       "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", 

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", 

                       "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", 

                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", 

                       "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 

                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 

                       "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",

                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", 

                       "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", 

                       "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will",

                       "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 

                       "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                       "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", 

                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 

                       "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", 

                       "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", 

                       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 

                       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are",

                       "y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                       "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 

                       'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 

                       'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 

                       'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',

                       'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 

                       'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',

                       'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 

                       'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',

                       'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
# Agregamos mas palabras al vocabulario si deshacemos las contracciones, simplificando el contenido del texto 

def quitar_contracciones(texto, lista):

    caracteres_especiales = ["’", "‘", "´", "`"]

    for s in caracteres_especiales:

        texto = texto.replace(s, "'")

    texto = ' '.join([lista[t] if t in lista else t for t in texto.split(" ")])

    return texto



#Eliminando las contracciones

train['question_text'] = train['question_text'].apply(lambda x: quitar_contracciones(x, lista_contracciones))

test['question_text'] = test['question_text'].apply(lambda x: quitar_contracciones(x, lista_contracciones))
# Definimos más caracteres especiales que no se incluyeron anterioremente ya que no estan incluidos o vinculados a las contracciones

puntuaciones = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

puntuaciones += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'



#Funcion para obtener todos los caracteres desconocidos entre el embedding y la lista de caracteres

def caracteres_desconocidos(embedding, puntuacion):

    desconocido = ''

    for p in puntuacion:

        if p not in embedding:

            desconocido += p

            desconocido += ' '

    return desconocido
#Definiendo el diccionario para mapear los caracteres especiales

lista_puntuaciones = {"‘": "'", "´": "'", "°": "", "€": "e", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '…': ' '}



#Funcion para eliminar caracteres desconocidos y reemplazarlos por el correspondiente

def eliminar_caracteres(texto, puntuacion, lista):

    for p in lista:

        texto = texto.replace(p, lista[p])

    

    for p in puntuacion:

        texto = texto.replace(p, f' {p} ')

    

    return texto

#Eliminando caracteres especiales

train['question_text'] = train['question_text'].apply(lambda x: eliminar_caracteres(x, puntuaciones, lista_puntuaciones))

test['question_text'] = test['question_text'].apply(lambda x: eliminar_caracteres(x, puntuaciones, lista_puntuaciones))
# Reconstruyendo el diccionario

df = pd.concat([train ,test], sort=False)

vocabulario = crear_vocabulario(df['question_text'])



# Imprimiendo las primeras 10 palabras desconocidas del glove y los porcentajes de embeddings/texto

palabara_desconocida = check_coverage(vocabulario, embedding_glove)

palabra_desconocida[:10]
#Reservando un 10% para el conjunto de validacion usando la libreria sklearn

train, val = train_test_split(train, test_size=0.2, random_state=42)



#Rellenamos los espacios vacíos en nuestros datasets

x_train = train['question_text'].fillna('_na_').values

x_val = val['question_text'].fillna('_na_').values

x_test = test['question_text'].fillna('_na_').values
# Haciendo usao de la librería Keras, tokenizaremos oraciones de acuerdo a max_features en los datasets de entrenamiento, validacion y evaluacion 

tokenizer = Tokenizer(num_words = max_features)

tokenizer.fit_on_texts(list(x_train))



x_train = tokenizer.texts_to_sequences(x_train)

x_val = tokenizer.texts_to_sequences(x_val)

x_test = tokenizer.texts_to_sequences(x_test)

print(x_train[0])



#Nos aseguraremos de que cada oracion tenga un tamaño de pregunta

x_train = pad_sequences(x_train, maxlen = size_question)

x_val = pad_sequences(x_val, maxlen = size_question)

x_test = pad_sequences(x_test, maxlen = size_question)
# Definimos los target values del modelo

y_train = train['target'].values

y_val = val['target'].values



# Mezclando el conjunto de datos

np.random.seed(42)



trn_idx = np.random.permutation(len(x_train))

val_idx = np.random.permutation(len(x_val))



x_train = x_train[trn_idx]

y_train = y_train[trn_idx]

x_val = x_val[val_idx]

y_val = y_val[val_idx]
#Cargando la matriz glove de embeddings

embedding_matrix_glove = cargar_matriz_glove(tokenizer.word_index, embedding_glove)
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

        shapeW = (input_shape[-1],)

        shapeB = (input_shape[1],)

        self.W = self.add_weight(shape= shapeW,

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight(shape=shapeB,

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
#se crea una funcion para aplicar la metrica de f1

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

#se crea el modelo de lstm con attention

def modelo_lstm_att(embedding_matrix):

    

    x_input = Input(shape=(size_question,))

    x = Embedding(max_features, word_vector, weights=[embedding_matrix], trainable=False)(x_input)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x = Bidirectional(LSTM(32, return_sequences=True))(x)

    

    att = Attention(size_question)(x)

    

    y = Dense(32, activation='relu')(att)

    y = Dropout(0.1)(y)

    y_output = Dense(1, activation='sigmoid')(y)    



    model = Model(inputs=x_input, outputs=y_output)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1, 

                                                                        "acc"])

    

    return model
#Entrenamiento del modelo!

def train_pred(model, epochs=2):

    

    for e in range(epochs):

        model.fit(x_train, y_train, batch_size=512, epochs=3, validation_data=(x_val, y_val))

        pred_val_y = model.predict([x_val], batch_size=1024, verbose=0)

        best_thresh = 0.5

        best_score = 0.0

        for thresh in np.arange(0.1, 0.501, 0.01):

            thresh = np.round(thresh, 2)

            score = metrics.f1_score(y_val, (pred_val_y > thresh).astype(int))

            if score > best_score:

                best_thresh = thresh

                best_score = score



        print("Val F1 Score: {:.4f}".format(best_score))



    pred_test_y = model.predict([x_test], batch_size=1024, verbose=0)



    return pred_val_y, pred_test_y, best_score
paragram = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

embedding_matrix_paragram = cargar_matriz_glove(tokenizer.word_index, cargar_embeddings(paragram))
embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_paragram], axis=0)
#creacion y entrenamiento del modelo

model_lstm = modelo_lstm_att(embedding_matrix)

model_lstm.summary()
outputs = []

pred_val_y, pred_test_y, best_score = train_pred(model_lstm, epochs=3)

outputs.append([pred_val_y, pred_test_y, best_score, 'model_lstm_att only Glove'])
#find best threshold

outputs.sort(key=lambda x: x[2]) 

weights = [i for i in range(1, len(outputs) + 1)]

weights = [float(i) / sum(weights) for i in weights] 



pred_val_y = np.mean([outputs[i][0] for i in range(len(outputs))], axis = 0)



thresholds = []

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    res = metrics.f1_score(y_val, (pred_val_y > thresh).astype(int))

    thresholds.append([thresh, res])

    print("F1 score at threshold {0} is {1}".format(thresh, res))

    

thresholds.sort(key=lambda x: x[1], reverse=True)

best_thresh = thresholds[0][0]
print("Best threshold:", best_thresh, "and F1 score", thresholds[0][1])
#prediciones y archivo para el submit

pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis = 0)

pred_test_y = (pred_test_y > best_thresh).astype(int)
sub = pd.read_csv('../input/quora-insincere-questions-classification/sample_submission.csv')

out_df = pd.DataFrame({"qid":sub["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)
import pandas as pd

sample_submission = pd.read_csv("../input/quora-insincere-questions-classification/sample_submission.csv")

test = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")

train = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")