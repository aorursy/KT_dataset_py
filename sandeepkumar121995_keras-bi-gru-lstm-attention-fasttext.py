import time

start_time = time.time()

from sklearn.model_selection import train_test_split

import sys, os, re, csv, codecs, numpy as np, pandas as pd

np.random.seed(32)

os.environ["OMP_NUM_THREADS"] = "4"

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU

from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, CuDNNGRU,CuDNNLSTM

from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D

from keras.models import Model, load_model

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

from keras import backend as K

from keras.engine import InputSpec, Layer

from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D



import logging

from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback



import warnings

warnings.filterwarnings('ignore')
class RocAucEvaluation(Callback):

    def __init__(self, validation_data=(), interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_val, self.y_val = validation_data



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict(self.X_val, verbose=0)

            score = roc_auc_score(self.y_val, y_pred)

            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
class AttentionWeightedAverage(Layer):

    """

    Computes a weighted average of the different channels across timesteps.

    Uses 1 parameter pr. channel to compute the attention value for a single timestep.

    """



    def __init__(self, return_attention=False, **kwargs):

        self.init = initializers.get('uniform')

        self.supports_masking = True

        self.return_attention = return_attention

        super(AttentionWeightedAverage, self).__init__(** kwargs)



    def build(self, input_shape):

        self.input_spec = [InputSpec(ndim=3)]

        assert len(input_shape) == 3



        self.W = self.add_weight(shape=(input_shape[2], 1),

                                 name='{}_W'.format(self.name),

                                 initializer=self.init)

        self.trainable_weights = [self.W]

        super(AttentionWeightedAverage, self).build(input_shape)



    def call(self, x, mask=None):

        # computes a probability distribution over the timesteps

        # uses 'max trick' for numerical stability

        # reshape is done to avoid issue with Tensorflow

        # and 1-dimensional weights

        logits = K.dot(x, self.W)

        x_shape = K.shape(x)

        logits = K.reshape(logits, (x_shape[0], x_shape[1]))

        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))



        # masked timesteps have zero weight

        if mask is not None:

            mask = K.cast(mask, K.floatx())

            ai = ai * mask

        att_weights = ai / K.sum(ai, axis=1, keepdims=True)

        weighted_input = x * K.expand_dims(att_weights)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:

            return [result, att_weights]

        return result



    def get_output_shape_for(self, input_shape):

        return self.compute_output_shape(input_shape)



    def compute_output_shape(self, input_shape):

        output_len = input_shape[2]

        if self.return_attention:

            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]

        return (input_shape[0], output_len)



    def compute_mask(self, input, input_mask=None):

        if isinstance(input_mask, list):

            return [None] * len(input_mask)

        else:

            return None
os.listdir('../input/')
train = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv")

test = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
train.head(3)
test.head(3)
## Since we have only 13GB RAM get rid of the other columns in train data set

train = train[['id','target','comment_text']]

train.head(2)
# Since the target column is in probability. we will convert this to first 0 and 1 based on 0.5 threshold

train['target'] = np.where(train['target'] >= 0.5, 1, 0)
embed_size = 300

max_features = 130000

max_len = 220

y = train['target'].values
X_train, X_valid, Y_train, Y_valid = train_test_split(train[['comment_text']], y, test_size = 0.1)
raw_text_train = X_train["comment_text"].str.lower()

raw_text_valid = X_valid["comment_text"].str.lower()

raw_text_test = test["comment_text"].str.lower()
%%time

tk = Tokenizer(num_words = max_features, lower = True)

tk.fit_on_texts(raw_text_train)

X_train["comment_seq"] = tk.texts_to_sequences(raw_text_train)

X_valid["comment_seq"] = tk.texts_to_sequences(raw_text_valid)

test["comment_seq"] = tk.texts_to_sequences(raw_text_test)
%%time

X_train = pad_sequences(X_train.comment_seq, maxlen = max_len)

X_valid = pad_sequences(X_valid.comment_seq, maxlen = max_len)

test = pad_sequences(test.comment_seq, maxlen = max_len)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))



word_index = tk.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
file_path = "best_model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid), interval = 1)

early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):

    inp = Input(shape = (max_len,))

    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)

    x = SpatialDropout1D(dr)(x)

    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)

    x = Bidirectional(CuDNNGRU(units, return_sequences=True))(x)  

    avg_pool = GlobalAveragePooling1D()(x)

    max_pool = GlobalMaxPooling1D()(x) 

    att = AttentionWeightedAverage()(x)

    conc = concatenate([att,avg_pool, max_pool])

    output = Dropout(0.7)(conc)

    output = Dense(units=144)(output)

    output = Activation('relu')(output)

    prediction = Dense(1, activation = "sigmoid")(conc)

    model = Model(inputs = inp, outputs = prediction)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, Y_train, batch_size = 128, epochs = 3, validation_data = (X_valid, Y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
%%time

model = build_model(lr = 1e-3, lr_d = 0, units = 112, dr = 0.3)
pred = model.predict(test, batch_size = 1024, verbose = 1)
submission = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv")

submission['prediction'] = pred

submission.to_csv("submission.csv", index = False)

submission.head(10)