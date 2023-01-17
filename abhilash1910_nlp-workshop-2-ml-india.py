# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split

#Tokenize the data and separate them in chunks of 256 units

maxlen=512
chunk_size=256
def fast_encode(texts, tokenizer, chunk_size=chunk_size, maxlen=maxlen):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    #sliding window methodology
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
# Create the model

def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    #Replaced from the Embedding+LSTM/CoNN layers
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
#Detect and deploy

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
#allow experimental tf
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration of hyperparameters
EPOCHS = 3
#batch size denotes the partitioning amongst the cluster replicas.
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer
train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
train_set['question_text'].shape
train_df['question_text']
train_x = fast_encode(train_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
val_x = fast_encode(test_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
train_y=train_set['target'].values
val_y=test_set['target'].values
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((val_x, val_y))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)


print(train_dataset)
print(valid_dataset)
#Build the transformer model
with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
n_steps = train_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
n_steps = valid_x.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS*2
)
#Step 5.
# First load the real tokenizer
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
#Step 5.
#Tokenize the samples
train_x = fast_encode(train_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
val_x = fast_encode(test_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
train_y=train_set['target'].values
val_y=test_set['target'].values
#Step 6.
#Load into Tensorflow compatible datasets
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((val_x, val_y))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

print(train_dataset)
print(valid_dataset)

#Step 7.
#Build the transformer model
with strategy.scope():
    transformer_layer = (
        transformers.TFRobertaModel
        .from_pretrained('roberta-base')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
#Step 8.
#Train the Transformer

n_steps = train_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
#Step 9.
#Validate the model

n_steps = val_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
#Step 5.
# First load the real tokenizer- Bert
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
#Step 5.
#Tokenize the datasets
train_x = fast_encode(train_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
val_x = fast_encode(test_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
train_y=train_set['target'].values
val_y=test_set['target'].values
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

#Step 6.
#Create Tensorflow Datasets
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((val_x, val_y))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

print(train_dataset)
print(valid_dataset)
#Step 7.
#Build the transformer model
with strategy.scope():
    transformer_layer = (
        transformers.TFBertModel
        .from_pretrained('bert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
#Step 8.
#Train the model

n_steps = train_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
#Step 9.
#Validate the model

n_steps = val_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

#Step 5.
# First load the real tokenizer- Bert
tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v1')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
#Step 5.
#Tokenize the datasets
train_x = fast_encode(train_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
val_x = fast_encode(test_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
train_y=train_set['target'].values
val_y=test_set['target'].values
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

#Step 6.
#Create Tensorflow Datasets
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((val_x, val_y))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

print(train_dataset)
print(valid_dataset)
#Step 7.
#Build the transformer model
with strategy.scope():
    transformer_layer = (
        transformers.TFAlbertModel
        .from_pretrained('albert-base-v1')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
#Step 8.
#Train the model

n_steps = train_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
#Step 9.
#Validate the model

n_steps = val_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

#Step 5.
# First load the real tokenizer- GPT-2
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
#Step 5.
#Tokenize the datasets
train_x = fast_encode(train_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
val_x = fast_encode(test_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
train_y=train_set['target'].values
val_y=test_set['target'].values
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

#Step 6.
#Create Tensorflow Datasets
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((val_x, val_y))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

print(train_dataset)
print(valid_dataset)
#Step 7.
#Build the transformer model
with strategy.scope():
    transformer_layer = (
        transformers.TFGPT2Model
        .from_pretrained('gpt2-medium')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
#Step 8.
#Train the model

n_steps = train_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
#Step 9.
#Validate the model

n_steps = val_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

#Step 5.
# First load the real tokenizer- BART
tokenizer = transformers.BartTokenizer.from_pretrained('facebook/bart-base')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
#Step 5.
#Tokenize the datasets
train_x = fast_encode(train_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
val_x = fast_encode(test_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
train_y=train_set['target'].values
val_y=test_set['target'].values
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

#Step 6.
#Create Tensorflow Datasets
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((val_x, val_y))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

print(train_dataset)
print(valid_dataset)
#Step 7.
#Build the transformer model- Bart tokenizer + ALbert model
with strategy.scope():
    transformer_layer = (
        transformers.TFAlbertModel
        .from_pretrained('albert-base-v1')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
#Step 8.
#Train the model

n_steps = train_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
#Step 9.
#Validate the model

n_steps = val_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

#Step 5.
# First load the real tokenizer- BART
tokenizer = transformers.TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
#Step 5.
#Tokenize the datasets
train_x = fast_encode(train_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
val_x = fast_encode(test_set['question_text'].astype(str), fast_tokenizer, maxlen=MAX_LEN)
train_y=train_set['target'].values
val_y=test_set['target'].values
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

#Step 6.
#Create Tensorflow Datasets
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_x, train_y))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((val_x, val_y))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

print(train_dataset)
print(valid_dataset)
#Step 7.
#Build the transformer model- Bart tokenizer + ALbert model
with strategy.scope():
    transformer_layer = (
        transformers.TFTransfoXLModel
        .from_pretrained('transfo-xl-wt103')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
#Step 8.
#Train the model

n_steps = train_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
#Step 9.
#Validate the model

n_steps = val_x.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

#Creating the inputs features
train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
#Codes from day-1
maxlen=1000
max_features=5000 
embed_size=300

#clean some null words or use the previously cleaned & lemmatized corpus

train_x=train_set['question_text'].fillna('_na_').values
val_x=test_set['question_text'].fillna('_na_').values

#Tokenizing steps- must be remembered
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x))
train_x=tokenizer.texts_to_sequences(train_x)
val_x=tokenizer.texts_to_sequences(val_x)

#Pad the sequence- To allow same length for all vectorized words
train_x=pad_sequences(train_x,maxlen=maxlen)
val_x=pad_sequences(val_x,maxlen=maxlen)


#get the target values - either using values or using Label Encoder
train_y=train_set['target'].values
val_y=test_set['target'].values

#Codes from Day-1
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras.engine.topology import Layer

try:
    from dataloader import TokenList, pad_to_longest
    # for transformer
except: pass

#Layer normalization class
class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        #Adding custom weights
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

#Division by 8 (q.k/d^0.5)
class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        #Joining scaled dot product
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        outputs = Add()([outputs, q])
        return self.layer_norm(outputs), attn
#Feedforward layer using COnv1D and Layer normalization.
class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def __call__(self, x):
        output = self.w_1(x) 
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)
#Encoder layer containing self/multi head attention with positionwisefeedforward
class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn
#Decoder layer with same architecture as the encoder.
class DecoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_att_layer  = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def __call__(self, dec_input, enc_output, self_mask=None, enc_mask=None):
        output, slf_attn = self.self_att_layer(dec_input, dec_input, dec_input, mask=self_mask)
        output, enc_attn = self.enc_att_layer(output, enc_output, enc_output, mask=enc_mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn, enc_attn
#This is from the paper "Attention is all you need" which hypothesizes sin and cosine for positional encoding
def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
        if pos != 0 else np.zeros(d_emb) 
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

#normal padding class for masking
def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class Encoder():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, \
                layers=6, dropout=0.1, word_emb=None, pos_emb=None):
        self.emb_layer = word_emb
        self.pos_layer = pos_emb
        self.emb_dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout) for _ in range(layers)]
        
    def __call__(self, src_seq, src_pos, return_att=False, active_layers=999):
        x = self.emb_layer(src_seq)
        if src_pos is not None:
            pos = self.pos_layer(src_pos)
            x = Add()([x, pos])
        x = self.emb_dropout(x)
        if return_att: atts = []
        mask = Lambda(lambda x:GetPadMask(x, x))(src_seq)
        for enc_layer in self.layers[:active_layers]:
            x, att = enc_layer(x, mask)
            if return_att: atts.append(att)
        return (x, atts) if return_att else x


class Transformer():
    def __init__(self, len_limit, d_model=embed_size, \
              d_inner_hid=512, n_head=10, d_k=64, d_v=64, layers=2, dropout=0.1, \
              share_word_emb=False, **kwargs):
        self.name = 'Transformer'
        self.len_limit = len_limit
        self.src_loc_info = True
        self.d_model = d_model
        self.decode_model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                            weights=[GetPosEncodingMatrix(len_limit, d_emb)])

        i_word_emb = Embedding(max_features, d_emb, weights=[embedding_matrix]) # Add Kaggle provided embedding here

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                               word_emb=i_word_emb, pos_emb=pos_emb)

        
    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile(self, active_layers=999):
        src_seq_input = Input(shape=(None,))
        src_seq = src_seq_input
        src_pos = Lambda(self.get_pos_seq)(src_seq)
        if not self.src_loc_info: src_pos = None

        x = self.encoder(src_seq, src_pos, active_layers=active_layers)
        # x = GlobalMaxPool1D()(x) # Not sure about this layer. Just wanted to reduce dimension
        x = GlobalAveragePooling1D()(x)
        outp = Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs=src_seq_input, outputs=outp)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

seq2seq = Transformer(maxlen, layers=1)
seq2seq.compile()
model = seq2seq.model
# Evaluate and Train the Model

model.fit(train_x, train_y, batch_size=1024, epochs=1, validation_data=(val_x, val_y))
