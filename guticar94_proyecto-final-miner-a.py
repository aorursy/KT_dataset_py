###Lectura de librerias

import numpy as np

import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from keras.models import Model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler
###Definición de parámetros para el modelo

NUM_MODELS = 2

MAX_FEATURES = 100000 

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

MAX_LEN = 220
###Las siguientes funciones son empleadas para extraer la información de los word vectors en una matriz



###Función que retorna una matriz con las palabras y el arreglo de dimensiones de cada palabra

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



###Función que abre el path de word vectors y envía la salida de get_coefs a un diccionario

def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))



###Función que retorna una matriz con los vectores (word vectors) de todas las palabras únicas contenidas en los datasets 

###del caso (train y test)

def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix
###Función de construcción del modelo

def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    

    ###Capas de LSTM bidireccional

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
### Lectura de datasets

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
train.describe()
###Seleccion de datasets

x_train = train['comment_text'].fillna('').values

x_test = test['comment_text'].fillna('').values



###Definición de y_train

y_train = np.where(train['target'] >= 0.5, 1, 0)



###Extracción de columnas que implican toxicidad para la y auxiliar

y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]



###Extraccion de columnas de identidad para definir los pesos de cada texto en el modelo

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



###Caracteres a borrar

# CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

CHARS_TO_REMOVE =""
sample_weights = np.ones(len(x_train), dtype=np.float32)

sample_weights += train[identity_columns].sum(axis=1)

sample_weights += train['target'] * (train[identity_columns]).sum(axis=1)

sample_weights += (train['target']) * train[identity_columns].sum(axis=1) * 5

sample_weights /= sample_weights.mean()
###Crear tokenización de todas las palabras en los DF

tokenizer = text.Tokenizer(num_words=MAX_FEATURES,filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(list(x_train) + list(x_test))



###Aplicar tokenizado a los dataframe de variables predictoras

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
#crawl-300d-2M es un word vector que contiene un modelo pre entrenado con 2 millones de palabras y 300 dimensiones

#glove.840B.300d es un word vector que contiene un modelo pre entrenado con 840 billones de palabras repetidas y 300 dimensiones

EMBEDDING_FILES = ['../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec','../input/glove840b300dtxt/glove.840B.300d.txt']
embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)



checkpoint_predictions = []

weights = []
###Correr modelo LSTM

for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=2,

            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

            callbacks=[

                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        weights.append(2 ** global_epoch)
###Hacer predicciones y enviar modelo a CSV

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)



submission = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': predictions

})



submission.to_csv('submission.csv', index=False)