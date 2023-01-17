import numpy as np, pandas as pd, random as rn, os, gc, re, time
start = time.time()
seed = 32
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '4'
np.random.seed(seed)
rn.seed(seed)
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads = 1,
                              inter_op_parallelism_threads = 1)
tf.set_random_seed(seed)
sess = tf.Session(graph = tf.get_default_graph(), config = session_conf)
from keras import backend as K
K.set_session(sess)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from keras.layers import Input, Dense, CuDNNLSTM, Bidirectional, Activation, Conv1D
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Add, Flatten, BatchNormalization, GlobalAveragePooling1D
from keras.layers import concatenate, SpatialDropout1D, CuDNNGRU, Lambda, GaussianDropout
from keras.layers import PReLU, ReLU, ELU
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.initializers import he_normal, he_uniform,  glorot_normal
from keras.initializers import glorot_uniform, zeros, orthogonal
from keras.models import Model, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

train = pd.read_csv("../input/train.csv").fillna("missing")
test = pd.read_csv("../input/test.csv").fillna("missing")

embedding_file1 = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
embedding_file2 = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"

embed_size = 300
max_features = 100000
max_len = 60
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', 
          '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
          '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  
          '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', 
          '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', 
          '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', 
          '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', 
          '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', 
          '¹', '≤', '‡', '√', 'β', 'α', '∅', 'θ', '÷', '₹']

def clean_punct(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, f' {punct} ')
    return x

train["question_text"] = train["question_text"].apply(lambda x: clean_punct(x))
test["question_text"] = test["question_text"].apply(lambda x: clean_punct(x))
test_shape = test.shape
print(test_shape)
sincere = train[train["target"] == 0]
insincere = train[train["target"] == 1]

print("Sincere questions {}; Insincere questions {}".format(sincere.shape[0], insincere.shape[0]))
temp1 = sincere.sample(53552)
temp2 = insincere.sample(2818)
fake_test = pd.concat([temp1, temp2], sort = True).reset_index()
train = train.drop(fake_test.index).reset_index()
target = train["target"].values

print("Fake test data shape {}".format(fake_test.shape))
print("New train data shape {}".format(train.shape))
train.head()
fake_test.head()
def get_glove(embedding_file):
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    return embeddings_index, emb_mean, emb_std

def get_para(embedding_file):
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, 
                                                                   encoding="utf8", 
                                                                   errors='ignore') if len(o)>100)
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    return embeddings_index, emb_mean, emb_std

glove_index, glove_mean, glove_std = get_glove(embedding_file1)
para_index, para_mean, para_std = get_para(embedding_file2)
def get_embed(tokenizer = None, embeddings_index = None, emb_mean = None, emb_std = None):
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return nb_words, embedding_matrix
tokenizer = Tokenizer(num_words = max_features, lower = True)
tokenizer.fit_on_texts(train["question_text"])

train_token = tokenizer.texts_to_sequences(train["question_text"])
fake_test_token = tokenizer.texts_to_sequences(fake_test["question_text"])
test_token = tokenizer.texts_to_sequences(test["question_text"])

train_seq = pad_sequences(train_token, maxlen = max_len)
fake_test_seq = pad_sequences(fake_test_token, maxlen = max_len)
X_test = pad_sequences(test_token, maxlen = max_len)
del train_token, fake_test_token, test_token; gc.collect()
nb_words, embedding_matrix1 = get_embed(tokenizer = tokenizer, embeddings_index = glove_index, 
                                        emb_mean = glove_mean, 
                                        emb_std = glove_std)
nb_words, embedding_matrix2 = get_embed(tokenizer = tokenizer, embeddings_index = para_index, 
                                        emb_mean = para_mean, 
                                        emb_std = para_std)
embedding_matrix = np.mean([embedding_matrix1, embedding_matrix2], axis = 0)
del embedding_matrix1, embedding_matrix2; gc.collect()
print("Embedding matrix completed!")
from keras.engine import Layer, InputSpec
from keras.layers import K

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
def get_f1(true, val):
    precision, recall, thresholds = precision_recall_curve(true, val)
    thresholds = np.append(thresholds, 1.001) 
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_threshold = thresholds[np.argmax(F)]
    
    return best_threshold, best_score    
def build_model(units = 40, dr = 0.3):
    inp = Input(shape = (max_len, ))
    embed_layer = Embedding(nb_words, embed_size, input_length = max_len,
                            weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(dr, seed = seed)(embed_layer)
    x = Bidirectional(CuDNNLSTM(units, kernel_initializer = glorot_normal(seed = seed), 
                                recurrent_initializer = orthogonal(gain = 1.0, seed = seed), 
                                return_sequences = True))(x)
    x = Bidirectional(CuDNNGRU(units, kernel_initializer = glorot_normal(seed = seed),
                               recurrent_initializer = orthogonal(gain = 1.0, seed = seed),
                               return_sequences = True))(x)    
    att = Attention(max_len)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    main = concatenate([att, avg_pool, max_pool])
    main = Dense(64, kernel_initializer = glorot_normal(seed = seed))(main)
    main = Activation("relu")(main)
    main = Dropout(0.1, seed = seed)(main)
    
    out = Dense(1, activation = "sigmoid", 
                kernel_initializer = glorot_normal(seed = seed))(main)
    model = Model(inputs = inp, outputs = out)
    model.compile(loss = "binary_crossentropy",
                  optimizer = Adam(), 
                  metrics = None)
    
    return model
fold = 5
batch_size = 1024
epochs = 5
oof_pred = np.zeros((train.shape[0], 1))
pred = np.zeros((test_shape[0], 1))
fake_pred = np.zeros((test_shape[0], 1))
thresholds = []

k_fold = StratifiedKFold(n_splits = fold, random_state = seed, shuffle = True)

for i, (train_idx, val_idx) in enumerate(k_fold.split(train_seq, target)):
    print("-"*50)
    print("Trainging fold {}/{}".format(i+1, fold))
    
    X_train, y_train = train_seq[train_idx], target[train_idx]
    X_val, y_val = train_seq[val_idx], target[val_idx]
    
    K.clear_session()
    model = build_model(units = 60)
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs,
              validation_data = (X_val, y_val), verbose = 2)
    val_pred = model.predict(X_val, batch_size = batch_size)
    oof_pred[val_idx] = val_pred
    fake_pred += model.predict(fake_test_seq, batch_size = batch_size)/fold
    pred += model.predict(X_test, batch_size = batch_size)/fold
    
    threshold, score = get_f1(y_val, val_pred)
    print("F1 score at threshold {} is {}".format(threshold, score))
threshold, score = get_f1(target, oof_pred)
print("F1 score after K fold at threshold {} is {}".format(threshold, score))
fake_test["pred"] = (fake_pred > threshold).astype(int)
print("Fake test F1 score is {}".format(f1_score(fake_test["target"], 
                                                 (fake_test["pred"]).astype(int))))
test["prediction"] = (pred > threshold).astype(int)
submission = test[["qid", "prediction"]]
submission.to_csv("submission.csv", index = False)
submission.head()
