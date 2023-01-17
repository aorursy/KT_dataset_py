from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
import string
train_df = pd.read_csv("../input/train.csv")
gc.collect()
test_df = pd.read_csv("../input/test.csv")
train_df["question_text"].isna().sum(), test_df["question_text"].isna().sum(), 
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state = 1001)
train_df.shape, val_df.shape
max_features = 95000
max_len = 72
embed_size = 300
%%time
# embdedding setup
# Source https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
embeddings_index = {}
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
gc.collect()
print('Found %s word vectors.' % len(embeddings_index))
# Convert values to embeddings
def text_to_array(text, zeros = 300, split_val = max_len):
    empyt_emb = np.zeros(zeros)
    text = text[:-1].split()[:split_val]
    embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (split_val - len(embeds))
    return np.array(embeds)
# train_vects = [text_to_array(X_text) for X_text in tqdm(train_df["question_text"])]
val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"][:5000])]) 
val_y = np.array(val_df["target"][:5000])
# Data providers
batch_size = 256

def batch_gen(train_df):
    n_batches = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])

from keras.models import Sequential,Model
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Input,Dropout, CuDNNGRU
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('normal')

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
class F1Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred = (y_pred > 0.5).astype(int)
            score = f1_score(self.y_val, y_pred)
            print("\n F1 - Epoch: %d - Score: %.6f \n" % (epoch+1, score)) 
esr = EarlyStopping(verbose=2, patience=3)
f1 = F1Evaluation(validation_data=(val_vects, val_y), interval=1)
inp = Input(shape=(max_len,300 ))
x = Bidirectional(CuDNNGRU(192, return_sequences=True))(inp)
x = Bidirectional(CuDNNGRU(64,return_sequences=True))(x)
x = Attention(max_len)(x)
#x = Dropout(0.25)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)

from IPython.display import Image
Image(filename='model.png')
#from keras.utils.vis_utils import model_to_dot

#SVG(model_to_dot(model).create(prog='dot', format='svg'))
model_name = 'gru_model'#%(rate_drop_lstm,rate_drop_dense)
print(model_name)
bst_model_path = model_name + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)
gc.collect() 
np.random.seed(2018)
mg = batch_gen(train_df)
hist = model.fit_generator(mg, epochs=30,
                    steps_per_epoch=512,
                    validation_data=(val_vects, val_y), callbacks=[f1, esr, model_checkpoint],
                    verbose=2)
model.load_weights(bst_model_path)
import matplotlib.pyplot as plt 
gc.collect()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xlim(1,)
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(1,)
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
# prediction part
batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr



all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(model.predict(x).flatten())
%%time
val_preds = []
for x in batch_gen(val_df):
    val_preds.extend(model.predict(x).flatten())
gc.collect()
pd.Series(all_preds).describe()
_thresh = [] 
for thresh in np.arange(0.1, 0.501, 0.01): 
    _thresh.append([thresh, f1_score(val_df["target"], (val_preds>thresh).astype(int))])
    print("F1 score at threshold {0} is {1}".format(thresh, f1_score(val_df["target"], (val_preds>thresh).astype(int))))
_thresh = np.array(_thresh)
best_id = _thresh[:,1].argmax()
best_thresh = _thresh[best_id][0]
best_thresh
y_te = (np.array(all_preds) > best_thresh).astype(np.int)

submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)
submit_df.head()

