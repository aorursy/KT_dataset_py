import numpy as np



import pandas as pd



import operator



from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



from keras import backend as K, initializers, regularizers, constraints, optimizers, layers

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, concatenate

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
#Revisando el tamaño del dataset

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print("Train shape : ",train.shape)

print("Test shape : ",test.shape)
#Definiendo los parametros para el procesamiento de los datos

EMBED_SIZE = 300 # Este es el tamaño de vector de palabras

MAX_FEATURES = 100000 # Cantidad máxima de palabras a tomar en cuenta. Si el vocabulario supera este número, Keras escoge las palabras con mayor frecuencia. 

MAXLEN = 40 # Este sera el tamaño maximo de la pregunta
#Construyendo el diccionario de palabras: Creamos un diccionario, donde el key es la palabra, y el valor es la frecuencia de esa palabra. 

def build_vocab(texts): #reconstruiremos el diccionario varias veces durante el pre-procesamiento para ver cambios

    sentences = texts.apply(lambda x: x.split()).values #Dividimos las oraciones en una lista de arreglos, donde cada arreglo representa una oración. Las celdas en ese arreglo son palabras.

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            #Contamos cada palabra en por oración. Si existe en el vocabulario, se le suma 1 a las veces que se repite. Si no existe, se agrega al vocabulario con 1. 

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab



df = pd.concat([train ,test], sort=False)



vocab = build_vocab(df['question_text'])

print("Tamaño inicial del vocabulario:")

print(len(vocab))

#Imprimiendo los primeros 10 elementos del diccionario

for x in list(vocab)[0:10]:

    print (x, vocab[x])

    print()
#Funcion para cargar la matriz de embeddings y sus indices 



def cargar_embedding(file):

    def get_coeficientes(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coeficientes(*o.split(" ")) for o in open(file, encoding='latin'))

    return embeddings_index



glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'



embed_glove = cargar_embedding(glove)



print('Glove embeddings cargados!')

len(embed_glove)
# Funcion para cargar la matriz de glove



def cargar_matriz_glove(word_index, embeddings_index):



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
# Funcion para evaluar el coverage entre el diccionacio y un conjunto de embedding



"""

La funcion calcula la intersección entre el diccionario y los embeddings

Si hay una palabra del diccionario que no este en el embedding se devuelve como desconocida. 

"""

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

    

    print("Glove")

    print('Se encontraron embeddings para el {:.3%} del diccionario'.format(len(known_words)/len(vocab)))

    print('Se encontraron embeddings para el {:.3%} de todo el cuerpo de texto'.format(nb_known_words/(nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
unknown_words = check_coverage(vocab, embed_glove)



#Para mejorar el modelo es util observar que palabras no estan en el diccionario

unknown_words[:20]
#Funcion para agregar minusculas al embedding

def agregar_minusculas(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")

    

#Llevando todo a minusculas

train['question_text'] = train['question_text'].apply(lambda x: x.lower())

test['question_text'] = test['question_text'].apply(lambda x: x.lower())
print("Imprimiendo el Glove!")

#Previo

unknown_words = check_coverage(vocab, embed_glove)



#Actualizado

agregar_minusculas(embed_glove, vocab) 

unknown_words = check_coverage(vocab, embed_glove)



#Imprimiendo las primeras 10 palabras desconocidas del glove

unknown_words[:10]
#Definiendo el diccionario para mapear las contracciones segun el enlace anterior

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}



#Tamaño del diccionario de mapeo de contracciones

len(contraction_mapping)
#Funcion para mapear las contracciones en ingles!

def quitar_contracciones(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



#Eliminando las contracciones!

train['question_text'] = train['question_text'].apply(lambda x: quitar_contracciones(x, contraction_mapping))

test['question_text'] = test['question_text'].apply(lambda x: quitar_contracciones(x, contraction_mapping))
#Reconstruyendo el diccionario de palabras luego de los cambios

df = pd.concat([train ,test], sort=False)

vocab = build_vocab(df['question_text'])



#Imprimiendo las primeras 10 palabras desconocidas del glove

print("Glove: ")

unknown_words = check_coverage(vocab, embed_glove)

unknown_words[:10]
#Definiendo los caracteres especiales...

punct_mapping = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'



#Funcion para obtener todos los caracteres desconocidos entre el embedding y la lista de caracteres

def caracteres_desconocidos(embed, punct):

    unknown = ''

    for p in punct:

        if p not in embed:

            unknown += p

            unknown += ' '

    return unknown



#Imprimiendo los caracteres desconocidos!

print("Glove:")

print(caracteres_desconocidos(embed_glove, punct_mapping))
#Definiendo el diccionario para mapear los caracteres especiales

puncts = {"‘": "'", "´": "'", "°": "", "€": "e", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '…': ' '}



#Funcion para eliminar caracteres desconocidos y reemplazarlos por el correspondiente

def eliminar_caracteres(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    return text



#Eliminando caracteres especiales

train['question_text'] = train['question_text'].apply(lambda x: eliminar_caracteres(x, punct_mapping, puncts))

test['question_text'] = test['question_text'].apply(lambda x: eliminar_caracteres(x, punct_mapping, puncts))
#Reconstruyendo el diccionario de palabras luego de los cambios

df = pd.concat([train ,test], sort=False)

vocab = build_vocab(df['question_text'])



#Imprimiendo las primeras 10 palabras desconocidas del glove

print("Glove: ")

unknown_words = check_coverage(vocab, embed_glove)

unknown_words[:10]
#Reservando un 10% para el conjunto de validacion

train, val = train_test_split(train, test_size=0.2, random_state=42)



#Filtrando los datos para evitar errores

xtrain = train['question_text'].fillna('_na_').values

xval = val['question_text'].fillna('_na_').values

xtest = test['question_text'].fillna('_na_').values
#Tokenizaremos oraciones segun el parametro MAX_FEATURES que se definio antes, es decir, 10000

tokenizer = Tokenizer(num_words=MAX_FEATURES)

tokenizer.fit_on_texts(list(xtrain))



#Tokenizamos el conjunto de entrenamineto, validacion y pruebas

xtrain = tokenizer.texts_to_sequences(xtrain)

xval = tokenizer.texts_to_sequences(xval)

xtest = tokenizer.texts_to_sequences(xtest)

print(xtrain[0])

#Nos aseguraremos de que cada oracion tenga un tamaño MAXLEN, definido anteriormente, es decir, 40

xtrain = pad_sequences(xtrain, maxlen=MAXLEN)

xval = pad_sequences(xval, maxlen=MAXLEN)

xtest = pad_sequences(xtest, maxlen=MAXLEN)
#Definiendo las salidas esperadas y mezclando el modelo para una mayor generalizacion

ytrain = train['target'].values

yval = val['target'].values



#Mezclando el conjunto de datos

np.random.seed(42)



trn_idx = np.random.permutation(len(xtrain))

val_idx = np.random.permutation(len(xval))



xtrain = xtrain[trn_idx]

ytrain = ytrain[trn_idx]

xval = xval[val_idx]

yval = yval[val_idx]
#Cargando la matriz glove de embeddings

embedding_matrix_glove = cargar_matriz_glove(tokenizer.word_index, embed_glove)

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
def model_lstm_att(embedding_matrix):

    

    inp = Input(shape=(MAXLEN,))

    x = Embedding(MAX_FEATURES, EMBED_SIZE, weights=[embedding_matrix], trainable=False)(inp)

    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)

    

    att = Attention(MAXLEN)(x)

    

    y = Dense(32, activation='relu')(att)

    y = Dropout(0.1)(y)

    outp = Dense(1, activation='sigmoid')(y)    



    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1, 

                                                                        "acc"])

    

    return model
#Entrenamiento del modelo!

def train_pred(model, epochs=2):

    

    for e in range(epochs):

        model.fit(xtrain, ytrain, batch_size=512, epochs=3, validation_data=(xval, yval))

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
paragram = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

embedding_matrix_para = cargar_matriz_glove(tokenizer.word_index, cargar_embedding(paragram))
embedding_matrix = np.mean([embedding_matrix_glove, embedding_matrix_para], axis=0)
#creacion y entrenamiento del modelo

model_lstm = model_lstm_att(embedding_matrix)

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

    res = metrics.f1_score(yval, (pred_val_y > thresh).astype(int))

    thresholds.append([thresh, res])

    print("F1 score at threshold {0} is {1}".format(thresh, res))

    

thresholds.sort(key=lambda x: x[1], reverse=True)

best_thresh = thresholds[0][0]
print("Best threshold:", best_thresh, "and F1 score", thresholds[0][1])
#prediciones y archivo para el submit

pred_test_y = np.mean([outputs[i][1] for i in range(len(outputs))], axis = 0)

pred_test_y = (pred_test_y > best_thresh).astype(int)
sub = pd.read_csv('../input/sample_submission.csv')

out_df = pd.DataFrame({"qid":sub["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)