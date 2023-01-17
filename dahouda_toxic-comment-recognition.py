! pip install googletrans
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from tqdm import tqdm
import nltk
import string
import seaborn as sns
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve
from keras import layers
from keras import models
from keras import layers
from keras import *

import transformers
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers

from tokenizers import BertWordPieceTokenizer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns
import os
import re

import transformers
import tensorflow as tf
from tqdm.notebook import tqdm
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tokenizers import BertWordPieceTokenizer
from kaggle_datasets import KaggleDatasets
from googletrans import Translator
from colorama import Fore, Back, Style, init
import plotly.graph_objects as go
translator = Translator()

from tensorflow.keras.layers import (Dense, Input, LSTM, Bidirectional, Activation, Conv1D, 
                                     GRU,Embedding, Flatten, Dropout, Add, concatenate, MaxPooling1D,
                                     GlobalAveragePooling1D,  GlobalMaxPooling1D, 
                                     GlobalMaxPool1D,SpatialDropout1D)

from tensorflow.keras import (initializers, regularizers, constraints, 
                              optimizers, layers, callbacks)

sns.set(style="darkgrid")





import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Detect hardware, return appropriate distribution strategy
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
train = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
test = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/test.csv")
subm = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
train.head()
train.shape
test.head()
test.shape
train['comment_text']

#nltk.download()
# What methods and attributes are in the nltk
dir(nltk)
# Did it work ?
from nltk.corpus import stopwords

stopwords.words('english')[0:10]
# Read in view the raw data

train = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train.head()
# Drop unused columns and label
train = train.drop(labels = ['id','severe_toxic','obscene','threat','insult','identity_hate'], axis=1)
train.columns = ['comment','toxic']
train.head()
# How big is the dataset
train.shape
# What portion of our comment are actually toxic ?
print(train['toxic'].value_counts())

# A toxic comment would receive a 1.0.
# A non-toxic comment would receive a 0.0.
sns.countplot(train.toxic)
# Are we missing any data ?
print("Number of nulls toxic : {}".format(train['toxic'].isnull().sum()))
print("Number of nulls comment : {}".format(train['comment'].isnull().sum()))
# Read in raw data and clean up the columns
pd.set_option('display.max_colwidth', 100)
# What punctuation is included in the dafault list ?
string.punctuation
train.head()
# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(train['comment'], train['toxic'], test_size = 0.2)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
#Prepare Data For Modeling
# Import the tools we will need from Keras

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Initalize and fit the tokenizer
tokenizer =  Tokenizer()
tokenizer.fit_on_texts(X_train)
# Use that tokenizer to transforme the comment in the training and test sets

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
# What do these sequences look like ?
X_train_seq[0]
# pad the sequences so each sequence is the same length
X_train_seq_padded = pad_sequences(X_train_seq, 50)
X_test_seq_padded = pad_sequences(X_test_seq, 50)
# What do these padded sequences look like ?
X_train_seq_padded[0]
# Buil The Model
# Import the tools needed from keras and functions to calculate recall and precision

import keras.backend as K
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
from sklearn import metrics

def recall_m(y_true, y_pred):
    true_positives  = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possibles_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    recall = true_positives / (possibles_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives  = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def roc_auc(predictions, target):
    tg, pr, thsld = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(tg, pr)
    return roc_auc
    
# Construct a simple RNN Model

model =  Sequential()
model.add(Embedding(len(tokenizer.index_word) + 1, 32))
model.add(LSTM(32, dropout = 0, recurrent_dropout = 0))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()
# Compile The Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision_m, recall_m])
# Fit The RNN Model
history = model.fit(X_train_seq_padded, y_train, batch_size=32, epochs=10, validation_data=(X_test_seq_padded, y_test))
# Plot the evaluation metrics by epoch for the model to see if we are over or underfitting


for i in ['accuracy', 'precision_m', 'recall_m']:
    acc = history.history[i]
    val_acc = history.history['val_{}'.format(i)]
    epochs = range(1, len(acc) + 1)
    
    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy')
    plt,plot(epochs, val_acc, label='Test Accuracy')
    plt.title('Results for {}'.format(i))
    plt.legend()
    plt.show()
scores = model.predict(X_test_seq_padded)
print("Auc: %.2f%%" % (roc_auc(scores, y_test)))
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE


# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer
x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train1.toxic.values
y_valid = valid.toxic.values
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)
def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
%%time
with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)
n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS*2
)
sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)
subResult = pd.read_csv('./submission.csv')
subResult.head()
train_set1 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train_set2 = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train_set2.toxic = train_set2.toxic.round().astype(int)
valid = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
# Combine train1 with a subset of train2
train = pd.concat([
    train_set1[['comment_text', 'toxic']],
    train_set2[['comment_text', 'toxic']].query('toxic==1'),
    train_set2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)
])
print(train.shape)
train.head()
print(valid.shape)
valid.head()
print(test.shape)
test.head()
print(train.toxic.value_counts())
sns.countplot(train.toxic)
print(valid.toxic.value_counts())
sns.countplot(valid.toxic)
print(valid.lang.value_counts())
sns.countplot(valid.lang)
print(test.lang.value_counts())
sns.countplot(test.lang)
def get_ax(rows=1, cols=2, size=7):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig, ax
fig, ax = get_ax()

sns.distplot(train[train['toxic']==0]['comment_text'].str.len(), axlabel="Non Toxic", ax=ax[0])
sns.distplot(train[train['toxic']==0]['comment_text'].str.split().str.len(), axlabel="Non Toxic", ax=ax[1])

fig.show()
fig, ax = get_ax()

sns.distplot(train[train['toxic']==1]['comment_text'].str.len(), axlabel="Toxic", ax=ax[0])
sns.distplot(train[train['toxic']==1]['comment_text'].str.split().str.len(), axlabel="Toxic", ax=ax[1])

fig.show()
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=50,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(train['comment_text'].sample(20000), 
               title = '[Comment_Text] Prevalent Words')
show_wordcloud(valid['comment_text'].sample(1000), 
               title = '[Comment_Text] Prevalent Words')
show_wordcloud(test['content'].sample(1000), 
               title = '[Content] Prevalent Words')
for i in range(5):
    print(f'[CONTENT {i}]\n', train['comment_text'][i])
    print()
# fast encoder
def fast_encode(texts, tokenizer, chunk_size=240, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
# general encoder
def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids'])
AUTO = tf.data.experimental.AUTOTUNE

# Create strategy from tpu
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

# Configuration
EPOCHS = 4
BATCH_SIZE = 16* strategy.num_replicas_in_sync
MODEL = 'jplu/tf-xlm-roberta-large'
MAX_LEN = 224
# https://huggingface.co/models
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

#First load the real tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)
%%time

x_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)
x_valid = regular_encode(valid.comment_text.values, tokenizer, maxlen=MAX_LEN)
x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)

y_train = train.toxic.values
y_valid = valid.toxic.values
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)
from tensorflow.keras import backend as K

def label_smoothing(y_true,y_pred):
     return tf.keras.losses.binary_crossentropy(y_true,y_pred,label_smoothing=0.15)

def focal_loss(gamma=2., alpha=.2):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    
    cls_token = sequence_output[:, 0, :]
    x = Dropout(0.3)(cls_token)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss=label_smoothing,
                  metrics=[tf.keras.metrics.AUC()]) # competition metrics
    
    return model
import tensorflow.keras.backend as K

with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained(MODEL)
    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
from IPython.display import SVG
SVG(tf.keras.utils.model_to_dot(model, dpi=70).create(prog='dot', format='svg'))
def callback():
    cb = []
    """
    Model-Checkpoint
    """
    checkpoint = callbacks.ModelCheckpoint('model.h5',
                                       save_best_only=True, 
                                       mode='min',
                                       monitor='val_loss', #  
                                       save_weights_only=True, verbose=0)

    cb.append(checkpoint)
    
    # Callback that streams epoch results to a csv file.
    log = callbacks.CSVLogger('log.csv')
    cb.append(log)

    return cb
calls = callback()
n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    callbacks = calls,
    epochs=EPOCHS
)
def visualize_model_preds(model, indices=[0, 17, 1, 24]):
    comments = valid.comment_text.loc[indices].values.tolist()
    preds = model.predict(x_valid[indices].reshape(len(indices), -1))

    for idx, i in enumerate(indices):
        if y_valid[i] == 0:
            label = "Non-toxic"
            color = f'{Fore.GREEN}'
            symbol = '\u2714'
        else:
            label = "Toxic"
            color = f'{Fore.RED}'
            symbol = '\u2716'

        print('{}{} {}'.format(color, str(idx+1) + ". " + label, symbol))
        print(f'{Style.RESET_ALL}')
        print("ORIGINAL")
        print(comments[idx]); print("")
        print("TRANSLATED")
        print(translator.translate(comments[idx]).text)
        fig = go.Figure()
        if list.index(sorted(preds[:, 0]), preds[idx][0]) > 1:
            yl = [preds[idx][0], 1 - preds[idx][0]]
        else:
            yl = [1 - preds[idx][0], preds[idx][0]]
        fig.add_trace(go.Bar(x=['Non-Toxic', 'Toxic'], y=yl, marker=dict(color=["seagreen", "indianred"])))
        fig.update_traces(name=comments[idx])
        fig.update_layout(xaxis_title="Labels", yaxis_title="Probability", template="plotly_white", title_text="Predictions for validation comment #{}".format(idx+1))
        fig.show()
        
visualize_model_preds(model)
sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)
subResult = pd.read_csv('./submission.csv')
subResult.head()