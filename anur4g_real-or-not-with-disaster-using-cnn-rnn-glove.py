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
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv') 
train_df.shape
train_df = train_df[['text', 'target']]
train_df.head()
train_labels = pd.get_dummies(train_df['target'])
train_labels
%%time
import os
import re
import string

import numpy as np 
from string import punctuation

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_auc_score

from tqdm import tqdm
from tqdm import tqdm_notebook
tqdm.pandas(desc="progress-bar")

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Reference : https://www.kaggle.com/sagar7390/nlp-on-disaster-tweets-eda-glove-bert-using-tfhub 

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
train_df['text']=train_df['text'].apply(lambda x : remove_URL(x))
train_df['text']=train_df['text'].apply(lambda x : remove_html(x))
train_df['text']=train_df['text'].apply(lambda x: remove_emoji(x))
train_df['text']=train_df['text'].apply(lambda x : remove_punct(x))
%%time
def get_coefs(word, *arr):
    try:
        return word, np.asarray(arr, dtype='float32')
    except:
        return None, None
    
embeddings_index = dict(get_coefs(*o.strip().split()) for o in tqdm_notebook(open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt')))

embed_size=300

for k in tqdm_notebook(list(embeddings_index.keys())):
    v = embeddings_index[k]
    try:
        if v.shape != (embed_size, ):
            embeddings_index.pop(k)
    except:
        pass

if None in embeddings_index:
  embeddings_index.pop(None)
  
values = list(embeddings_index.values())
all_embs = np.stack(values)

emb_mean, emb_std = all_embs.mean(), all_embs.std()
%%time
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.models import Sequential

from keras.layers import Input, Dense, Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.callbacks import Callback
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils.vis_utils import plot_model
%%time
MAX_NB_WORDS = 80000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(train_df['text'])
%%time
sen = 'Hi Kaggle'
print(tokenizer.texts_to_sequences([sen]))
# %%time
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
# find max length of text
def FindMaxLength(lst): 
    maxList = max(lst, key = lambda i: len(i)) 
    maxLength = len(maxList) 
    return maxLength

FindMaxLength(train_sequences)
%%time
MAX_LENGTH = 31
padded_train_sequences = pad_sequences(train_sequences, maxlen=MAX_LENGTH)
%%time
padded_train_sequences
%%time
padded_train_sequences.shape
%%time
word_index = tokenizer.word_index
nb_words = MAX_NB_WORDS
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

oov = 0
for word, i in tqdm_notebook(word_index.items()):
    if i >= MAX_NB_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        oov += 1

print(oov)
%%time
LABELS = 2
def get_rnn_cnn_model_with_glove_embedding():
    embedding_dim = 300 
    inp = Input(shape=(MAX_LENGTH, ))
    x = Embedding(MAX_NB_WORDS, embedding_dim, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=True)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(LABELS, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

get_rnn_cnn_model_with_embedding = get_rnn_cnn_model_with_glove_embedding()
filepath = 'weights-improvement-glove.hdf5'
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

batch_size = 10
epochs = 20

history = get_rnn_cnn_model_with_embedding.fit(x=padded_train_sequences, 
                    y=labels.values, 
                    validation_split = 0.33,
                    batch_size=batch_size, 
                    callbacks=[checkpoint], 
                    epochs=epochs, 
                    verbose=1)
get_rnn_cnn_model_with_embedding.load_weights('weights-improvement-glove.hdf5')
get_rnn_cnn_model_with_embedding.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Model is Loaded')
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test_df.head()
labels_ls = [0, 1]
X_test = test_df['text']
X_test.shape
maxlen = 31
token_sen = tokenizer.texts_to_sequences(X_test)
padded_test_sequences = pad_sequences(token_sen, maxlen=maxlen)
models = {}
MAX_LENGTH = 31
models['get_rnn_cnn_model_with_embedding'] = {"model": get_rnn_cnn_model_with_embedding,
                                              "process": lambda x: pad_sequences(tokenizer.texts_to_sequences(x), maxlen=MAX_LENGTH)}

y_pred_rnn_cnn_with_glove_embeddings = get_rnn_cnn_model_with_embedding.predict(
    padded_test_sequences, verbose=1, batch_size=2048)
result = list((y_pred_rnn_cnn_with_glove_embeddings == y_pred_rnn_cnn_with_glove_embeddings.max(axis=1, keepdims=True)).astype(int))
result_df = pd.DataFrame(result, columns=labels_ls)
result_df['target'] = result_df.idxmax(axis=1)
final_df = pd.DataFrame()
final_df['id'] = test_df['id']
final_df['target'] = result_df['target']
final_df.to_csv("submission.csv", index=False, header=True)
