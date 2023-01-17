# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np 
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_test = '/kaggle/input/nlp-getting-started/test.csv'
data_train = '/kaggle/input/nlp-getting-started/train.csv'
data_table = pd.read_csv(data_train)
data_table.head(10)
data_table_train = data_table.sample(frac=1)
data_table_train.head(10)
train_text_list = list(data_table_train['text'])
train_labels_list = list(data_table_train['target'])
print(len(train_text_list), len(train_labels_list))
print(train_text_list)
embedding_dim = 32
max_length = 25
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=len(train_text_list)
test_portion=.1
tokenizer = Tokenizer(oov_token=oov_tok, filters='!"#$%&()*+,-./:;<=>?@...')

tokenizer.fit_on_texts(train_text_list)
word_index = tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(train_text_list)
padded = pad_sequences(sequences=sequences, maxlen=max_length, padding=padding_type,truncating=trunc_type)

#split = int(test_portion * training_size
#train_seq = padded[0:split]
#test_seq = padded[split:]
#train_seq_lab = train_labels_list[0:split]
#test_seq_lab = train_labels_list[split:]

#train_seq = np.array(train_seq)
#test_seq = np.array(test_seq)
#train_seq_lab = np.array(train_seq_lab)
#test_seq_lab = np.array(test_seq_lab)

train_seq = np.array(padded)
train_seq_lab = np.array(train_labels_list)
print(len(train_seq), len(train_seq_lab))
LR_START = 0.00001
LR_MAX = 0.0001 
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(50)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
model = tf.keras.Sequential([
tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
tf.keras.layers.Conv1D(25,5,activation='relu'),
tf.keras.layers.MaxPooling1D(2),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(1, activation='sigmoid') 
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#history = model.fit(train_seq,train_seq_lab, epochs=50, verbose=1, validation_data=(test_seq,test_seq_lab), batch_size=100)
history = model.fit(train_seq,train_seq_lab, epochs=50, verbose=1, batch_size=100, callbacks=[lr_callback])
plt.figure(figsize=(20,7))
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
data_table_test = pd.read_csv(data_test)
data_table_test.head()
test_text_list = list(data_table_test['text'])
tokenizer = Tokenizer(oov_token=oov_tok, filters='!"#$%&()*+,-./:;<=>?@...')

tokenizer.fit_on_texts(test_text_list)
word_index_test = tokenizer.word_index
vocab_size_test = len(word_index)

sequences_test = tokenizer.texts_to_sequences(test_text_list)
padded_test = pad_sequences(sequences=sequences_test, maxlen=max_length, padding=padding_type,truncating=trunc_type)

test_seq = np.array(padded_test)
prediction = model.predict(test_seq)
y_predict = (prediction > 0.5).astype(int).reshape(data_table_test.shape[0])
print(len(y_predict))
out = pd.DataFrame({'Id': data_table_test['id'], 'target': y_predict})
out.to_csv('pr.csv', index=False)