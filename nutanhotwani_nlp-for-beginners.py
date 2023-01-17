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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, metrics, decomposition, model_selection, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
import seaborn as sns
%matplotlib inline
train=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
test=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
validation=pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
train.head()
train.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)
train.head()
train.shape
train=train.loc[:12000, :]
train.shape
train.comment_text.apply(lambda x:len(str(x).split())).max()
def roc_auc(predictions, target):
    fpr, tpr, threshold=metrics.roc_curve(target, predictions)
    roc_auc=metrics.auc(fpr, tpr)
    return roc_auc
x_train, x_valid, y_train, y_valid=train_test_split(train.comment_text.values, train.toxic.values, 
                                stratify=train.toxic.values, test_size=0.2, random_state=42, shuffle=True)
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
token=text.Tokenizer(num_words=None)
max_len=1500

token.fit_on_texts(list(x_train)+list(x_valid))

x_trainseq=token.texts_to_sequences(x_train)
x_validseq=token.texts_to_sequences(x_valid)
x_trainpad=sequence.pad_sequences(x_trainseq, maxlen=max_len)
x_validpad=sequence.pad_sequences(x_validseq, maxlen=max_len)
 
word_index=token.word_index
print(len(word_index))
model=Sequential()
model.add(Embedding(len(word_index)+1, 300))
model.add(SimpleRNN(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_trainpad, y_train, epochs=5, batch_size=64*strategy.num_replicas_in_sync)
scores = model.predict(x_validpad)
print("Auc: %.2f%%" % (roc_auc(scores,y_valid)))

score_model=[]
score_model.append({"Mode":"SimpleRNN", "AUC_Score":roc_auc(scores, y_valid)})
embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
from tqdm import tqdm
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
model=Sequential()
model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_trainpad, y_train, epochs=5, batch_size=64*strategy.num_replicas_in_sync)
pred=model.predict(x_validpad)
print("Acc: %.2f%%" %(roc_auc(pred, y_valid)))
score_model.append({"Model":"LSTM", "AUC_Score:":roc_auc(pred, y_valid)})
model=Sequential()
model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(GRU(300))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_trainpad, y_train, epochs=5, batch_size=64*strategy.num_replicas_in_sync)
scores=model.predict(x_validpad)
print("Acc: .%2f %%"%(roc_auc(scores, y_valid)))
score_model.append({"Model":"GRU", "AUC_Score":roc_auc(scores, y_valid)})
score_model
model=Sequential()
model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_trainpad, y_train, epochs=5, batch_size=64*strategy.num_replicas_in_sync)
scores=model.predict(x_validpad)
print("Acc: .%2f%%"%(roc_auc(scores, y_valid)))
score_model.append({"Model":"Bidirectional RNN", "AUC_Score":roc_auc(scores, y_valid)})
results = pd.DataFrame(scores_model).sort_values(by='AUC_Score',ascending=False)
results.style.background_gradient(cmap='Blues')
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
import transformers

from tokenizers import BertWordPieceTokenizer
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
from tqdm import tqdm
def fast_encode(text, tokenizer, chunk_size=256, maxlen=512):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids=[]
    for i in tqdm(range(0, len(text), chunk_size)):
        text_chunk = text[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)
AUTO = tf.data.experimental.AUTOTUNE

# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192
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
train_dataset=(tf.data.Dataset.from_tensor_slices((x_train, y_train))
              .repeat().shuffle(2048).batch(BATCH_SIZE).prefetch(AUTO))
valid_dataset=(tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
              .batch(BATCH_SIZE).cache().prefetch(AUTO))
test_dataset=(tf.data.Dataset.from_tensor_slices((x_test))
              .batch(BATCH_SIZE))
def build_model(transformer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained('distilbert-base-multilingual-cased')
    )
model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()
n_steps=x_train.shape[0]
history=model.fit(train_dataset, steps_per_epoch=n_steps, validation_data=valid_dataset, epochs=EPOCHS)
sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)