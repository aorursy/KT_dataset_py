import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import f1_score



from keras import backend as K

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D

from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate

from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

from keras.optimizers import Adam

from keras.models import Model

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers
embed_size = 300

max_features = 120000

maxlen = 70

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]



def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def split_text(x):

    x = wordninja.split(x)

    return '-'.join(x)





def load_and_prec():

    train_df = pd.read_csv("../input/train.csv")

    test_df = pd.read_csv("../input/test.csv")

    

    train_df["question_text"] = train_df["question_text"].str.lower()

    test_df["question_text"] = test_df["question_text"].str.lower()

    

    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))

    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))

    

    print("Train shape : ",train_df.shape)

    print("Test shape : ",test_df.shape)

    

    ## split to train and val

    train_df, val_df = train_test_split(train_df, test_size=0.001, random_state=2018) # hahaha





    ## fill up the missing values

    train_X = train_df["question_text"].fillna("_na_").values

    val_X = val_df["question_text"].fillna("_na_").values

    test_X = test_df["question_text"].fillna("_na_").values



    ## Tokenize the sentences

    tokenizer = Tokenizer(num_words=max_features)

    tokenizer.fit_on_texts(list(train_X))

    train_X = tokenizer.texts_to_sequences(train_X)

    val_X = tokenizer.texts_to_sequences(val_X)

    test_X = tokenizer.texts_to_sequences(test_X)



    ## Pad the sentences 

    train_X = pad_sequences(train_X, maxlen=maxlen)

    val_X = pad_sequences(val_X, maxlen=maxlen)

    test_X = pad_sequences(test_X, maxlen=maxlen)



    ## Get the target values

    train_y = train_df['target'].values

    val_y = val_df['target'].values  

    



    

    #shuffling the data

    np.random.seed(2018)

    trn_idx = np.random.permutation(len(train_X))

    val_idx = np.random.permutation(len(val_X))



    train_X = train_X[trn_idx]

    val_X = val_X[val_idx]

    train_y = train_y[trn_idx]

    val_y = val_y[val_idx]    

    

    return train_X, val_X, test_X, train_y, val_y, tokenizer.word_index
def load_fasttext(word_index):    

    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,'rt', encoding='UTF8') if len(o)>100)



    all_embs = np.stack(embeddings_index.values())

    emb_mean,emb_std = all_embs.mean(), all_embs.std()

    embed_size = all_embs.shape[1]



    # word_index = tokenizer.word_index

    nb_words = min(max_features, len(word_index))

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():

        if i >= max_features: continue

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None: embedding_matrix[i] = embedding_vector



    return embedding_matrix
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

        return input_shape[0],  self.features_dim

def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def model_gru(embedding_matrix):

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

    x = Bidirectional(CuDNNGRU(100, return_sequences=True))(x)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

    x = Attention(maxlen)(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])

    

    return model
def train_pred(model, epochs=2):

    for e in range(epochs):

        model.fit(train_X, train_y, batch_size=512, epochs=1, validation_data=(val_X, val_y))

        pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)

    pred_test_y = model.predict([test_X], batch_size=1024, verbose=1)

    return pred_val_y, pred_test_y
train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec()

vocab = []



for w,k in word_index.items():

    vocab.append(w)

    if k >= max_features:

        break

        

embedding_matrix = load_fasttext(word_index)
embedding_matrix
pred_val_y, pred_test_y = train_pred(model_gru(embedding_matrix), epochs = 10)
pred_test_y1 = (pred_test_y > 0.34).astype(int)

test_df = pd.read_csv("../input/test.csv", usecols=["qid"])

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y1

out_df.to_csv("submission.csv", index=False)