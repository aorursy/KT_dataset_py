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
test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.info()
train_df['text'][5700]
text = np.asarray(train_df['text']) # Converting to numpy array
target = np.asarray(train_df['target'])
from keras.preprocessing.text import Tokenizer
max_vocab = 10000 # Defining max_vocab and max_len variable's
max_len = 500
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
from keras.preprocessing.sequence import pad_sequences
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=max_len)
data # Data is converted to vector
word_index
train_samples = int(len(text)*0.8) # Using 80% of data for training 
train_samples
text_train = data[:train_samples]
target_train = target[:train_samples]
text_test = data[train_samples:len(text)-2] # Using 20% data for evaluation, will be rebuilding model with full data
target_test = target[train_samples:len(text)-2]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU, SimpleRNN
embedding_mat_columns=32
model = Sequential()
model.add(Embedding(input_dim=max_vocab,
 output_dim=embedding_mat_columns,
 input_length=max_len))
model.add(LSTM(units=embedding_mat_columns))
model.add(Dropout(0.5)) # Adding Dropout layers to avoid overfitting
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
 metrics=['acc'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=2) # Using Early Stop to stop training
model.fit(text_train, target_train, epochs=10, batch_size=24, validation_data=(text_test,target_test), callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
acc = model.evaluate(text_test, target_test)
print("Test loss is {0:.2f} accuracy is {1:.2f} ".format(acc[0],acc[1]))
model.fit(data, target, epochs=5, batch_size=24,)
test_df
test_text = np.asarray(test_df['text'])
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(test_text)
test_sequences = tokenizer.texts_to_sequences(test_text)
word_index = tokenizer.word_index
test_data = pad_sequences(sequences, maxlen=max_len)
pred_label = model.predict(test_data)
pred_label
pred_label[3000]
test_df['text'][3000]
submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
probabilities = pd.DataFrame(data=pred_label)
submission['probabilities'] = probabilities
submission = submission.drop('target',axis=1)
submission[submission['probabilities'] < 0.5] =  0  # If probability is less than 0.5 it will return 0
submission[submission['probabilities'] > 0.5] = 1
submission
submission['target'] = submission['probabilities']
submission = submission.drop('probabilities', axis=1)
submission['target'] = submission['target'].astype('int64')
submission.to_csv('./submission.csv', index=False)