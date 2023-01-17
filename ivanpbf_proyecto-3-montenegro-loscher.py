import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os



from keras.models import Model, Sequential

from keras.layers import Dense, Embedding, Input, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM, concatenate

from keras.preprocessing import text, sequence



from tqdm import tqdm

from sklearn.model_selection import train_test_split



print(os.listdir("../input"))



        

train_df = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')

test_df = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')

y = train_df["target"]
#procesamiento de palabras, reemplazamos aquellas que vemos necesarias asi como acomodar ciertos signos, esto se aplica a los sets

reemplazar = {r"i'm": 'i am',

                r"'re": ' are',

                r"ain't": 'is not',

                r"let's": 'let us',

                r"didn't": 'did not',

                r"'s":  ' is',

                r"'ve": ' have',

                r"can't": 'can not',

                r"cannot": 'can not',

                r"shan’t": 'shall not',

                r"n't": ' not',

                r"'d": ' would',

                r"'ll": ' will',

                r"'scuse": 'excuse',

                ',': ' ,',

                '.': ' .',

                '!': ' !',

                '?': ' ?',

                '\s+': ' '}

def limpiar(text):

    text = text.lower()

    for s in reemplazar:

        text = text.replace(s, reemplazar[s])

    text = ' '.join(text.split())

    return text



X_train= train_df['question_text'].apply(lambda p: limpiar(p))

X_train = X_train.fillna("dieter").values

X_test= test_df['question_text'].apply(lambda p: limpiar(p))

X_test = X_test.fillna("dieter").values
maxlen = 50 #palabras maximas en un documento

max_carac = 50000 #maximas caracteristicas

embed_tama = 300 #tamano del embedding

batch_size = 256 #batch size a utilizar, no queremos que sea tan alto ni tan bajo

epochs = 3 #tardan bastante, pero el modelo llega a ser lo suficientemente preciso para no tener que usar una gran cantidad de epochs



tokenizer = text.Tokenizer(num_words=max_carac) #tokenizer permite vectorizar un cuerpo de texto, convirtiendo cada cuerpo en una sequencia de ints

tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#usando el archivo embedding que nos ofrece, crear la matriz de embedding usando GloVe

def get_coefs(word,*arr): 

    return word, np.asarray(arr, dtype='float32')



def cargar_archivo_embedding(file):  #cargamos e indexamos el embedding a usar, en este caso glove para la representacion de palabras 

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        index_e = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    else:

        index_e = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

    return index_e



glove = cargar_archivo_embedding('../input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt')
#creacion de la matriz a traves de los datos anteriores

def hacer_matriz_embedding(embedding, tokenizer, features):

    todos_embs = np.stack(embedding.values())

    emb_mean,emb_std = todos_embs.mean(), todos_embs.std()

    embed_tama = todos_embs.shape[1]

    index_palabras = tokenizer.word_index

    matriz_embedding = np.random.normal(emb_mean, emb_std, (features, embed_tama))

    

    for word, i in index_palabras.items():

        if i >= features:

            continue

        vector_embedding = embedding.get(word)

        if vector_embedding is not None: 

            matriz_embedding[i] = vector_embedding

    

    return matriz_embedding



embed_mat = hacer_matriz_embedding(glove, tokenizer, max_carac) #recibe glove, tokenizer procesado y las caracteristicas

print(embed_mat)
#creacion del modelo, usamos LSTM y word embedding, se decidio esta LSTM bidireccional que en el recorrido recuerde las palabras importantes para el contexto

def embed_model():

    model = Sequential()

    inp = Input(shape=(maxlen, )) #instanciar keras tensor

    x = Embedding(max_carac, embed_tama, weights=[embed_mat])(inp) #embedding a traves del procesamiento que hicimos

    x = Bidirectional(LSTM(64, return_sequences=True))(x) #LSTM y Bidirectional, dimension de 64 que retorna el output de la secuencia

    avg_pool = GlobalAveragePooling1D()(x) #global average pooling para data temporal 

    max_pool = GlobalMaxPooling1D()(x) #max pooling para data espacial

    conc = concatenate([avg_pool, max_pool]) #concatenando ambos poolings

    outp = Dense(1, activation="sigmoid")(conc) #funcion de activacion sigmoide 

    

    model = Model(inputs = inp,outputs = outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return model



model = embed_model()
#entrenar

X_t, X_val, y_t, y_val = train_test_split(x_train, y, test_size = 0.1, random_state= np.random) #dividimos entre train y Y

historial = model.fit(X_t, y_t, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True) #entrenamos con nuestro modelo 
# corremos el modelo en un test para generar nuestras predicciones respecto a ello

y_pred = model.predict(x_test, batch_size=batch_size)

y_pred.shape
#output

y_te = (y_pred[:,0] > 0.5).astype(np.int) #clasificacion





submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te}) #generar archivo de output

submit_df.to_csv("submission.csv", index=False)