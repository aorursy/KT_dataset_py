import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import gc,os,sys

import operator 



from sklearn.metrics import f1_score, roc_auc_score

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, Masking

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras.optimizers import Adam



pd.set_option("display.max_colwidth", 200)



sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format
print(os.listdir("../input"))
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape, test.shape)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
train.head()
test.head()
valcnt = train['target'].value_counts().to_frame()

valcnt.plot.bar()

valcnt.T / len(train)
train.sort_values(['target'], ascending=False).head()
train.sort_values(['target']).head()
# word-count histgram

word_counts = train['question_text'].apply(lambda x: len(x.split()))

word_counts.hist(bins=50, figsize=(10,3))



print('max words: ', max(word_counts))

print('sum words: ', sum(word_counts))

del word_counts
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }



ZEN = "".join(chr(0xff01 + i) for i in range(94))

HAN = "".join(chr(0x21 + i) for i in range(94))

ZEN2HAN = str.maketrans(ZEN, HAN)



def preprocess(data):

    def clean_special_chars(text):

        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&…'

        for p in punct:

            text = text.replace(p, ' ')

        for p in punct_mapping:

            text = text.replace(p, punct_mapping[p])

        #for p in '0123456789':

        #    text = text.replace(p, ' ')

        text = text.translate(ZEN2HAN)

        return text



    data = data.astype(str).apply(lambda x: clean_special_chars(x))

    return data



train['question_text'] = preprocess(train['question_text'])

test['question_text'] = preprocess(test['question_text'])
train['question_text'].values[0]
text_to_word_sequence(train['question_text'].values[0])
X_train = train.drop(['qid','target'], axis=1)

Y_train = train['target']

X_test  = test.drop(['qid'], axis=1)

#train_id  = train['qid']

#test_id  = test['qid']

del train, test



print(X_train.shape, X_test.shape)
TARGET_COLUMN = 'target'

TEXT_COLUMN = 'question_text'

MAX_NUM_WORDS = 300000

TOKENIZER_FILTER = '\r\t\n'



# Create a text tokenizer.

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters=TOKENIZER_FILTER)

tokenizer.fit_on_texts(list(X_train[TEXT_COLUMN]) + list(X_test[TEXT_COLUMN]))
counter = sorted(dict(tokenizer.word_docs).items(), key=lambda x:x[1], reverse=True)

wordcount = pd.Series([x[1] for x in counter], [x[0] for x in counter])

del counter



wordcount[:30].plot.bar(color='navy', width=0.7, figsize=(12,3))
tokenizer_tx = Tokenizer(num_words=MAX_NUM_WORDS, filters=TOKENIZER_FILTER)

tokenizer_tx.fit_on_texts(list(X_train.loc[Y_train == 1, TEXT_COLUMN]))



counter = sorted(dict(tokenizer_tx.word_docs).items(), key=lambda x:x[1], reverse=True)

wordcount_tx = pd.Series([x[1] for x in counter], [x[0] for x in counter])

wordcount_stats = pd.concat([wordcount, wordcount_tx], axis=1, keys=['all', 'toxic'], sort=False)



# word count of contains toxic text is over 80%

wordcount_tx = wordcount_stats[wordcount_stats['all'] * 0.8 <= wordcount_stats['toxic']].copy()

wordcount_tx[wordcount_tx['toxic'] > 0]

wordcount_tx.sort_values(by='toxic', ascending=False, inplace=True)



print(len(wordcount_tx))

wordcount_tx['toxic'][:30].plot.bar(color='red', width=0.7, figsize=(12,3))
# word list - prioritize toxic words

wordcount = pd.concat([wordcount_tx['all'], wordcount]).to_frame().reset_index()

wordcount.drop_duplicates(keep='first', inplace=True)

wordcount = wordcount.set_index('index')[0]

del counter, wordcount_tx, wordcount_stats
wordsum = wordcount.sum()

n_words = len(wordcount)

print('words:', n_words)



cumsum_rate = wordcount.cumsum() / wordsum

cover_rate = {}

for i in range(100, 90, -1):

    p = i / 100

    cover_rate[str(i)+'%'] = n_words - len(cumsum_rate[cumsum_rate > p])

#del cumsum_rate



pd.Series(cover_rate).plot.barh(color='navy', figsize=(12, 3), title='vocab-size by coverage-rate')

pd.Series(cover_rate).to_frame().T
VOCAB_SIZE = 50000



print('covered until', wordcount[VOCAB_SIZE], 'times word')
%%time



EMBEDDINGS_DIMENSION = 300

EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

#EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((VOCAB_SIZE + 1, EMBEDDINGS_DIMENSION))

    unknown_words = []

    for i in range(VOCAB_SIZE):

        try:

            word = wordcount.index[i]

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            unknown_words.append(word)

    return embedding_matrix, unknown_words



#crawl_matrix, unknown_words_crawl = build_matrix(CRAWL_EMBEDDING_PATH)

glove_matrix, unknown_words_glove = build_matrix(EMBEDDING_FILE)



word2index = dict((wordcount.index[i], i) for i in range(VOCAB_SIZE))



#embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

embedding_matrix = glove_matrix

embedding_matrix.shape
words_count = len(unknown_words_glove)

print('n unknown words (glove):', words_count, ', {:.3%} of all words'.format(words_count / n_words))

print('unknown words (glove):', unknown_words_glove)
MAX_SEQUENCE_LENGTH = 128



def word_index(word):

    try:

        return word2index[word]

    except KeyError:

        return VOCAB_SIZE



# All comments must be truncated or padded to be the same length.

def pad_text(texts, tokenizer):

    matrix = [list(map(word_index, text_to_word_sequence(t, filters=TOKENIZER_FILTER))) for t in texts]

    return pad_sequences(matrix, maxlen=MAX_SEQUENCE_LENGTH)



train_text = pad_text(X_train[TEXT_COLUMN], tokenizer)

test_text = pad_text(X_test[TEXT_COLUMN], tokenizer)
del (X_train, X_test)

gc.collect()



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
from sklearn.metrics import f1_score



class F1Callback(Callback):

    def __init__(self):

        self.f1s = []



    def on_epoch_end(self, epoch, logs):

        eps = np.finfo(np.float32).eps

        recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)

        precision = logs["val_true_positives"] / (logs["val_predicted_positives"] + eps)

        f1 = 2*precision*recall / (precision+recall+eps)

        print("f1_val (from log) =", f1)

        self.f1s.append(f1)



def true_positives(y_true, y_pred):

    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))



def possible_positives(y_true, y_pred):

    return K.sum(K.round(K.clip(y_true, 0, 1)))



def predicted_positives(y_true, y_pred):

    return K.sum(K.round(K.clip(y_pred, 0, 1)))
from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers



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
def build_model(lr=0.0, lr_d=0.0, units=64, spatial_dr=0.0, 

                dense_units=0, dr=0.1, conv_size=32, epochs=20):

    

    file_path = "best_model.hdf5"

    check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)



    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_layer = Embedding(*embedding_matrix.shape,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

    

    x = embedding_layer(sequence_input)

    x = SpatialDropout1D(spatial_dr)(x)

    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)   

    x = Conv1D(conv_size, 2, padding="valid", kernel_initializer="he_uniform")(x)

  

    #att = Attention(MAX_SEQUENCE_LENGTH)(x)

    avg_pool1 = GlobalAveragePooling1D()(x)

    max_pool1 = GlobalMaxPooling1D()(x)     

    

    #x = concatenate([att, avg_pool1, max_pool1])

    x = concatenate([avg_pool1, max_pool1])

    x = BatchNormalization()(x)

    x = Dense(int(dense_units / 2), activation='relu')(x)

    x = Dropout(dr)(x)

    

    preds = Dense(1, activation='sigmoid')(x)

    

    model = Model(inputs=sequence_input, outputs=preds)

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), 

                  metrics=['acc', true_positives, possible_positives, predicted_positives])

    model.summary()

    history = model.fit(train_text, Y_train, batch_size=1024, epochs=epochs, validation_split=0.1, 

                        verbose=1, callbacks=[check_point, early_stop, F1Callback()])



    #model = load_model(file_path, custom_objects={'F1Callback':F1Callback, 'Attention':Attention})

    return model
model = build_model(lr=1e-3, lr_d=1e-7, units=64, spatial_dr=0.2, dense_units=64, dr=0.1, conv_size=64, epochs=10)

pred = model.predict(test_text, batch_size=1024)
def threshold_search(y_true, y_proba):

    best_threshold = 0

    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:

        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)

        if score > best_score:

            best_threshold = threshold

            best_score = score

    search_result = {'threshold': best_threshold, 'f1': best_score}

    return search_result



#train_meta = model.predict(train_text, batch_size=1024)

#threshold_search(Y_train, train_meta)
pred = (pred[:,0] > 0.5).astype(np.int)



submission = pd.read_csv('../input/sample_submission.csv', index_col='qid')

submission['prediction'] = pred

submission.reset_index(drop=False, inplace=True)

submission.to_csv('submission.csv', index=False)

submission.head()
pd.Series(pred).value_counts().to_frame().T / len(pred)