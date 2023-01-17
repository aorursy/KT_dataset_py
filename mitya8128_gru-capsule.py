# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

X_train = train_df["question_text"].fillna("dieter").values

test_df = pd.read_csv("../input/test.csv")

X_test = test_df["question_text"].fillna("dieter").values

y = train_df["target"]
train_df.head()
import pandas as pd

import numpy as np

from tqdm import tqdm

tqdm.pandas()

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn import metrics



from keras.preprocessing import text, sequence

from keras.layers import *

from keras.layers import ReLU

from keras.models import *

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.initializers import *

from keras.optimizers import *

import keras.backend as K

from keras.callbacks import *

import tensorflow as tf

import os

import time

import gc

import re

from unidecode import unidecode
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
mispell_dict = {"aren't" : "are not",

"can't" : "cannot",

"couldn't" : "could not",

"didn't" : "did not",

"doesn't" : "does not",

"don't" : "do not",

"hadn't" : "had not",

"hasn't" : "has not",

"haven't" : "have not",

"he'd" : "he would",

"he'll" : "he will",

"he's" : "he is",

"i'd" : "I would",

"i'd" : "I had",

"i'll" : "I will",

"i'm" : "I am",

"isn't" : "is not",

"it's" : "it is",

"it'll":"it will",

"i've" : "I have",

"let's" : "let us",

"mightn't" : "might not",

"mustn't" : "must not",

"shan't" : "shall not",

"she'd" : "she would",

"she'll" : "she will",

"she's" : "she is",

"shouldn't" : "should not",

"that's" : "that is",

"there's" : "there is",

"they'd" : "they would",

"they'll" : "they will",

"they're" : "they are",

"they've" : "they have",

"we'd" : "we would",

"we're" : "we are",

"weren't" : "were not",

"we've" : "we have",

"what'll" : "what will",

"what're" : "what are",

"what's" : "what is",

"what've" : "what have",

"where's" : "where is",

"who'd" : "who would",

"who'll" : "who will",

"who're" : "who are",

"who's" : "who is",

"who've" : "who have",

"won't" : "will not",

"wouldn't" : "would not",

"you'd" : "you would",

"you'll" : "you will",

"you're" : "you are",

"you've" : "you have",

"'re": " are",

"wasn't": "was not",

"we'll":" will",

"didn't": "did not",

"tryin'":"trying"}

def clean_text(x):

    x = str(x)

    for punct in puncts:

        x = x.replace(punct, f' {punct} ')

    return x



def clean_numbers(x):

    x = str(x)

    x = re.sub(r'[0-9]{5,}', '#####', x)

    x = re.sub(r'[0-9]{4}', '####', x)

    x = re.sub(r'[0-9]{3}', '###', x)

    x = re.sub(r'[0-9]{2}', '##', x)

    

    return x



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)



def replace_typical_misspell(text):

    def replace(match):

        return mispellings[match.group(0)]

    

    text = mispellings_re.sub(replace, text)

    return text
#X_train["question_text"] = X_train["question_text"].apply(lambda x: x.lower())

#X_test["question_text"] = X_test["question_text"].apply(lambda x: x.lower())



#X_train["question_text"] = X_train["question_text"].map(clean_text)

#X_test["question_text"] = X_test["question_text"].map(clean_text)



X_train["question_text"] = X_train["question_text"].map(clean_numbers)

X_test["question_text"] = X_test["question_text"].map(clean_numbers)



X_train["question_text"] = X_train["question_text"].map(replace_typical_misspell)

X_test["question_text"] = X_test["question_text"].map(replace_typical_misspell)

SEQ_LEN = 100  # magic number - length to truncate sequences of words

maxlen = 100

max_features = 50000



tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train, maxlen=maxlen)

x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
EMBEDDING_FILE_GLOVE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'



def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')



embeddings_index_glove = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE_GLOVE))



all_embs_glove = np.stack(embeddings_index_glove.values())

emb_mean_glove,emb_std_glove = -0.005838499,0.48782197

embed_size_glove = all_embs_glove.shape[1]





word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix_glove = np.random.normal(emb_mean_glove, emb_std_glove, (nb_words, embed_size_glove))



for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector_glove = embeddings_index_glove.get(word)

    if embedding_vector_glove is not None: embedding_matrix_glove[i] = embedding_vector_glove
# Capsule layer



def squash(x, axis=-1):

    # s_squared_norm is really small

    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)

    # return scale * x

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)

    scale = K.sqrt(s_squared_norm + K.epsilon())

    return x / scale



class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,

                 activation='default', **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.kernel_size = kernel_size

        self.share_weights = share_weights

        if activation == 'default':

            self.activation = squash

        else:

            self.activation = Activation(activation)



    def build(self, input_shape):

        super(Capsule, self).build(input_shape)

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(1, input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     # shape=self.kernel_size,

                                     initializer='glorot_uniform',

                                     trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.W = self.add_weight(name='capsule_kernel',

                                     shape=(input_num_capsule,

                                            input_dim_capsule,

                                            self.num_capsule * self.dim_capsule),

                                     initializer='glorot_uniform',

                                     trainable=True)



    def call(self, u_vecs):

        if self.share_weights:

            u_hat_vecs = K.conv1d(u_vecs, self.W)

        else:

            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])



        batch_size = K.shape(u_vecs)[0]

        input_num_capsule = K.shape(u_vecs)[1]

        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,

                                            self.num_capsule, self.dim_capsule))

        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]



        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]

        for i in range(self.routings):

            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]

            c = K.softmax(b)

            c = K.permute_dimensions(c, (0, 2, 1))

            b = K.permute_dimensions(b, (0, 2, 1))

            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))

            if i < self.routings - 1:

                b = tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])



        return outputs



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)
def capsule():

    K.clear_session()       

    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size_glove, weights=[embedding_matrix_glove], trainable=False)(inp)

    x = SpatialDropout1D(rate=0.2)(x)

    x = Bidirectional(CuDNNLSTM(160, return_sequences=True, 

                                kernel_initializer=glorot_normal(seed=12300), recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

    x = Capsule(num_capsule=15, dim_capsule=12, routings=8, share_weights=True)(x)

    x = Flatten()(x)

    x = Dense(100, activation=None, kernel_initializer=glorot_normal(seed=12300))(x)

    x = Dropout(0.2)(x)

    x = BatchNormalization()(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer=Adam(),)

    return model



model_glove = capsule()
batch_size = 128

epochs = 3
from sklearn.model_selection import train_test_split

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y, test_size = 0.1, random_state=42)
hist_glove = model_glove.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),

                  verbose=True)

pred_glove_y = model_glove.predict(x_test, batch_size=1024)

pred_glove_y = (pred_glove_y[:,0] > 0.33).astype(np.int)
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": pred_glove_y})

submit_df.to_csv("submission.csv", index=False)