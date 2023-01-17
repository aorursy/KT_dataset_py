# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importando las librerias 



import operator



from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



from keras import backend as K, initializers, regularizers, constraints, optimizers, layers

from keras.layers import Dense, Input,LSTM, Embedding, Dropout, Activation,GRU, Conv1D, concatenate

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate, Lambda

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.optimizers import Adam

from keras.models import Model

from keras.engine.topology import Layer



import os

print(os.listdir("../input"))
#Ahora se hace el procesamiento de los datos

entrenamiento = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

prueba = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")



#Definición de parámetros

tamano_embedding=300

MAX_FEATURES=100000 #esta es la cantidad máxima de palabras a tomar en cuenta

MAXLEN =40 #la longitud maxima de la pregunta será 40
#Construccion del diccionario 



def construccion_diccionario(texto): 

    #ahora se procede a separar cada oración en una lista de arreglos, para cada oración hay un arreglo y las celdas de estos las ocupan las palabras

    oraciones=texto.apply(lambda x: x.split()).values

    diccionario={}

    

    #Se procede a contar cada palabra en cada una de las oraciones

    for oracion in oraciones:

        for palabra in oracion:

            

            try: 

                diccionario[palabra]+=1 #si la palabra existe en el diccionario se le suma uno a las veces que esta se repite

            except KeyError: 

                diccionario[palabra]=1 #pero, si la palabra no existe en el diccionario se agrega al diccionario con 1

    return diccionario



df = pd.concat([entrenamiento ,prueba], sort=False)



diccionario = construccion_diccionario(df['question_text'])

print("Tamaño inicial del diccionario:")

print(len(diccionario)) #imprimiendo el tamano inicial del diccionario

#Imprimiendo los primeros 10 elementos del diccionario

for x in list(diccionario)[0:10]:

    print (x, diccionario[x])

    print()
#A continuacion se define la funcion cargar_embedding para cargar la matriz de embeddings y sus indices

def cargar_embedding(file):

    def obtener_coeficientes(palabra,*arr): 

        return palabra, np.asarray(arr, dtype='float32')

    indice_embeddings = dict(obtener_coeficientes(*o.split(" ")) for o in open(file, encoding='latin'))

    return indice_embeddings



glove = '../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt'



embed_glove = cargar_embedding(glove)



print('Glove embeddings cargados!')

len(embed_glove)
# Se define una funcion para cargar la matriz de glove



def cargar_m_glove(indice_palabra, indice_embedding):



    all_embs = np.stack(indice_embedding.values())

    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    tamano_embedding = all_embs.shape[1]

    

    nb_palabras = min(MAX_FEATURES, len(indice_palabra))

    matriz_embedding = np.random.normal(emb_mean, emb_std, (nb_palabras, tamano_embedding))



    for palabra, i in indice_palabra.items():

        if i >= MAX_FEATURES:

            continue

        vector_embedding = indice_embedding.get(palabra)

        if vector_embedding is not None:

           matriz_embedding[i] = vector_embedding



    return matriz_embedding
#se define una funcion para determinar el coverage entre el diccionario y un conjunto de embedding



def check_coverage(diccionario,indice_embedding):

    palabras_conocidas = {}

    palabras_desconocidas = {}

    nb_palabras_conocidas = 0

    nb_palabras_desconocidas = 0

    for palabra in diccionario.keys():

        try:

            palabras_conocidas[palabra] = indice_embedding[palabra]

            nb_palabras_conocidas += diccionario[palabra]

        except:

             palabras_desconocidas[palabra] = diccionario[palabra]

             nb_palabras_desconocidas+= diccionario[palabra]

             pass

    

    print("Glove")

    print('Se encontraron embeddings para el {:.3%} del diccionario'.format(len(palabras_conocidas)/len(diccionario)))

    print('Se encontraron embeddings para el {:.3%} de todo el cuerpo de texto'.format(nb_palabras_conocidas/(nb_palabras_conocidas + nb_palabras_desconocidas)))

    palabras_desconocidas = sorted(palabras_desconocidas.items(), key=operator.itemgetter(1))[::-1]



    return palabras_desconocidas
palabras_desconocidas = check_coverage(diccionario, embed_glove)



#Es util observar que palabras no estan en el diccionario, para mejorar el modelo

palabras_desconocidas[:20]
#definimos una funcion para agregar minusculas al embedding

def agregar_minusculas(embedding, diccionario):

    count = 0

    for palabra in diccionario:

        if palabra in embedding and palabra.lower() not in embedding:  

            embedding[palabra.lower()] = embedding[palabra]

            count += 1

    print(f"Added {count} words to embedding")

    

#se lleva todo a minusculas

entrenamiento['question_text'] = entrenamiento['question_text'].apply(lambda x: x.lower())

prueba['question_text'] = prueba['question_text'].apply(lambda x: x.lower())
print("Imprimiendo el Glove!")

#Previo

palabras_desconocidas = check_coverage(diccionario, embed_glove)



#Actualizamos

agregar_minusculas(embed_glove, diccionario) 

palabras_desconocidas = check_coverage(diccionario, embed_glove)



#Imprimimos 10 palabras desconocidas del glove

palabras_desconocidas[:10]
#se define el diccionario para mapear las contracciones de este enlace:  https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/77758

mapeo_contracciones = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



#definimos una funcion para mapear las contracciones en ingles

def quitar_contracciones(texto, mapeo):

    especiales = ["’", "‘", "´", "`"]

    for s in especiales:

        texto = texto.replace(s, "'")

    texto = ' '.join([mapeo[t] if t in mapeo else t for t in texto.split(" ")])

    return texto



#Eliminando las contracciones

entrenamiento['question_text'] = entrenamiento['question_text'].apply(lambda x: quitar_contracciones(x, mapeo_contracciones))

prueba['question_text'] = prueba['question_text'].apply(lambda x: quitar_contracciones(x, mapeo_contracciones))
#Se reconstruye el diccionario de palabras para guardar los cambios

df = pd.concat([entrenamiento ,prueba], sort=False)

diccionario = construccion_diccionario(df['question_text'])



#se imprimen las primeras 10 palabras desconocidas del glove

print("Glove: ")

palabras_desconocidas = check_coverage(diccionario, embed_glove)

palabras_desconocidas[:10]
#Eliminando caracteres especiales

#se definen los caracteres especiales

mapeo_puntuacion = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

mapeo_puntuacion += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'



#Funcion para obtener todos los caracteres desconocidos entre el embedding y la lista de caracteres

def caracteres_desconocidos(embed, puntuacion):

    desconocido = ''

    for p in puntuacion:

        if p not in embed:

            desconocido += p

            desconocido += ' '

    return desconocido





print("Glove:")#Imprimiendo los caracteres desconocidos

print(caracteres_desconocidos(embed_glove, mapeo_puntuacion))
#se define  el diccionario para mapear los caracteres especiales

puntuacion = {"‘": "'", "´": "'", "°": "", "€": "e", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '…': ' '}



#Se define la función eliminar_caracteres para eliminar caracteres desconocidos y reemplazarlos por el correspondiente

def eliminar_caracteres(texto, puntuacion, mapeo):

    for p in mapeo:

        texto = texto.replace(p, mapeo[p])

    

    for p in puntuacion:

        texto = texto.replace(p, f' {p} ')

    

    return texto



#Eliminando caracteres especiales

entrenamiento['question_text'] = entrenamiento['question_text'].apply(lambda x: eliminar_caracteres(x, mapeo_puntuacion, puntuacion))

prueba['question_text'] = prueba['question_text'].apply(lambda x: eliminar_caracteres(x, mapeo_puntuacion, puntuacion))
#Ahora se reconstruye el diccionario de palabras luego de los cambios

df = pd.concat([entrenamiento ,prueba], sort=False)

diccionario = construccion_diccionario(df['question_text'])



#Imprimiendo las primeras 10 palabras desconocidas del glove

print("Glove: ")

palabras_desconocidas = check_coverage(diccionario, embed_glove)

palabras_desconocidas[:10]


entrenamiento, val = train_test_split(entrenamiento, test_size=0.2, random_state=42) #Se reserva  un 10% para el conjunto de validacion



#Filtrando los datos para evitar errores

xentrenamiento = entrenamiento['question_text'].fillna('_na_').values

xval = val['question_text'].fillna('_na_').values

xprueba = prueba['question_text'].fillna('_na_').values

#Tokenizaremos oraciones segun el parametro MAX_FEATURES

tokenizer = Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(list(xentrenamiento))



#Tokenizamos el conjunto de entrenamineto, validacion y pruebas

xentrenamiento = tokenizer.texts_to_sequences(xentrenamiento)

xval = tokenizer.texts_to_sequences(xval)

xprueba = tokenizer.texts_to_sequences(xprueba)

print(xentrenamiento[0])

#Nos aseguraremos de que cada oracion tenga un tamaño MAXLEN

xentrenamiento = pad_sequences(xentrenamiento, maxlen=MAXLEN)

xval = pad_sequences(xval, maxlen=MAXLEN)

xprueba = pad_sequences(xprueba, maxlen=MAXLEN)
#Definiendo las salidas esperadas y mezclando el modelo para una mayor generalizacion

yentrenamiento = entrenamiento['target'].values

yval = val['target'].values



#Mezclando el conjunto de datos

np.random.seed(42)



trn_idx = np.random.permutation(len(xentrenamiento))

val_idx = np.random.permutation(len(xval))



xentrenamiento = xentrenamiento[trn_idx]

yentrenamiento = yentrenamiento[trn_idx]

xval = xval[val_idx]

yval = yval[val_idx]
#Cargando la matriz glove de embeddings

matriz_embedding_glove = cargar_m_glove(tokenizer.word_index, embed_glove)

print("Matriz de embeddings cargada!")
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

        shapeW=(input_shape[-1],)

        shapeB=(input_shape[1],)

        self.W = self.add_weight(shape=shapeW,

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





def model_lstm_att(matriz_embedding):

    

    inp = Input(shape=(MAXLEN,))

    x = Embedding(MAX_FEATURES, tamano_embedding, weights=[matriz_embedding], trainable=False)(inp)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x = Bidirectional(LSTM(32, return_sequences=True))(x)

    

    att = Attention(MAXLEN)(x)

    

    y = Dense(32, activation='relu')(att)

    y = Dropout(0.1)(y)

    outp = Dense(1, activation='sigmoid')(y)    



    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1, 

                                                                        "acc"])

    

    return model
def train_pred(model, epochs=2):

    

    for e in range(epochs):

        model.fit(xentrenamiento, yentrenamiento, batch_size=512, epochs=3, validation_data=(xval, yval))

        pred_val_y = model.predict([xval], batch_size=1024, verbose=0)

        best_thresh = 0.5

        best_score = 0.0

        for thresh in np.arange(0.1, 0.501, 0.01):

            thresh = np.round(thresh, 2)

            score = metrics.f1_score(yval, (pred_val_y > thresh).astype(int))

            if score > best_score:

                best_thresh = thresh

                best_score = score



        print("Val F1 Score: {:.4f}".format(best_score))



    pred_test_y = model.predict([xtest], batch_size=1024, verbose=0)



    return pred_val_y, pred_test_y, best_score
def model_lstm_att(embedding_matrix):

    

    inp = Input(shape=(MAXLEN,))

    x = Embedding(MAX_FEATURES, tamano_embedding, weights=[matriz_embedding], trainable=False)(inp)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x = Bidirectional(LSTM(32, return_sequences=True))(x)

    

    att = Attention(MAXLEN)(x)

    

    y = Dense(32, activation='relu')(att)

    y = Dropout(0.1)(y)

    outp = Dense(1, activation='sigmoid')(y)    



    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1, 

                                                                        "acc"])

    

    return model
def train_pred(model, epochs=2):

    

    for e in range(epochs):

        model.fit(xentrenamiento, yentrenamiento, batch_size=512, epochs=3, validation_data=(xval, yval))

        pred_val_y = model.predict([xval], batch_size=1024, verbose=0)

        best_thresh = 0.5

        best_score = 0.0

        for thresh in np.arange(0.1, 0.501, 0.01):

            thresh = np.round(thresh, 2)

            score = metrics.f1_score(yval, (pred_val_y > thresh).astype(int))

            if score > best_score:

                best_thresh = thresh

                best_score = score



        print("Val F1 Score: {:.4f}".format(best_score))



    pred_test_y = model.predict([xprueba], batch_size=1024, verbose=0)



    return pred_val_y, pred_test_y, best_score
paragram = '../input/quora-insincere-questions-classification/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

embedding_matrix_para = cargar_m_glove(tokenizer.word_index, cargar_embedding(paragram))

matriz_embedding = np.mean([matriz_embedding_glove, embedding_matrix_para], axis=0)
#creacion y entrenamiento del modelo

model_lstm = model_lstm_att(matriz_embedding)

model_lstm.summary()
outputs = []

pred_val_y, pred_test_y, best_score = train_pred(model_lstm, epochs=3)

outputs.append([pred_val_y, pred_test_y, best_score, 'model_lstm_att only Glove'])
#encontrar el mejor threshold

outputs.sort(key=lambda x: x[2]) 

weights = [i for i in range(1, len(outputs) + 1)]

weights = [float(i) / sum(weights) for i in weights] 



pred_val_y = np.mean([outputs[i][0] for i in range(len(outputs))], axis = 0)



thresholds = []

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    res = metrics.f1_score(yval, (pred_val_y > thresh).astype(int))

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