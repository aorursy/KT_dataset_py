# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

import nltk

# nltk.download('stopwords')

# nltk.download('punkt')

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english')) 

import tensorflow as tf

from sklearn.metrics import roc_auc_score

from keras import models, layers, Model

from keras.preprocessing import text, sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## Clean Punctuation & Stopwords

class clean_text:

	def __init__(self, text):

		self.text = text

	

	# Remove Punctuation

	def rm_punct(text):

		punct = set([p for p in "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'])

		text = [t for t in text if t not in punct]

			

		return "".join(text)



	# Remove Stopwords

	def rm_stopwords(text):

		word_tokens = word_tokenize(text)   

		result = [w for w in word_tokens if w not in stop_words]

				

		return " ".join(result)



def Embedding_CuDNNLSTM_model(max_words, max_len):

    sequence_input = layers.Input(shape=(None, ))

    x = layers.Embedding(max_words, 128, input_length=max_len)(sequence_input)

    x = layers.SpatialDropout1D(0.3)(x)

    x = layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))(x)

    x = layers.Bidirectional(layers.CuDNNLSTM(64, return_sequences=True))(x)

    

    avg_pool1d = layers.GlobalAveragePooling1D()(x)

    max_pool1d = layers.GlobalMaxPool1D()(x)

    

    x = layers.concatenate([avg_pool1d, max_pool1d])

    x = layers.Dense(32, activation='relu')(x)

    x = layers.BatchNormalization()(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    

    model = models.Model(sequence_input, output)

    

    return model

    



def auroc(y_true, y_pred):

	return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
## load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

print(train_data.shape)

print(test_data.shape)
train_data.head()
train_df = train_data[['id','comment_text','target']]

test_df = test_data.copy()



# set index

train_df.set_index('id', inplace=True)

test_df.set_index('id', inplace=True)



# y_label

train_y_label = np.where(train_df['target'] >= 0.5, 1, 0) # Label 1 >= 0.5 / Label 0 < 0.5

train_df.drop(['target'], axis=1, inplace=True)
# ratio by Class

Counter(train_y_label)
train_data['comment_text'].head(20)
# remove punctuation 

train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_text.rm_punct(x))

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_text.rm_punct(x))

# remove stopwords

X_train = train_df['comment_text'].apply(lambda x: clean_text.rm_stopwords(x))

X_test = test_df['comment_text'].apply(lambda x: clean_text.rm_stopwords(x))
## tokenize

max_words = 100000

tokenizer = text.Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(X_train)



# texts_to_sequences

sequences_text_train = tokenizer.texts_to_sequences(X_train)

sequences_text_test = tokenizer.texts_to_sequences(X_test)
sequences_text_train[:5]
# add padding

max_len = max(len(l) for l in sequences_text_train)

pad_train = sequence.pad_sequences(sequences_text_train, maxlen=max_len)

pad_test = sequence.pad_sequences(sequences_text_test, maxlen=max_len)
pad_train[:5]
## embedding_lstm models 

model = Embedding_CuDNNLSTM_model(max_words, max_len)



# model compile

model.compile(optimizer='adam',

			 loss='binary_crossentropy', metrics=['acc', auroc])

model.summary()
# keras.callbacks

callbacks_list = [

		ReduceLROnPlateau(

			monitor='val_auroc', patience=2, factor=0.1, mode='max'),	# val_loss가 patience동안 향상되지 않으면 학습률을 0.1만큼 감소 (new_lr = lr * factor)

		EarlyStopping(

			patience=5, monitor='val_auroc', mode='max', restore_best_weights=True)

]



history = model.fit(pad_train, train_y_label,

					epochs=7, batch_size=1024,

					callbacks=callbacks_list, 

					validation_split=0.3, verbose=2)
# plot score by epochs

auroc = history.history['auroc']

val_auroc = history.history['val_auroc']

epochs = range(1, len(auroc)+1)



plt.figure(figsize=(7,3))

plt.plot(epochs, auroc, 'b', label='auroc')

plt.plot(epochs, val_auroc, 'r', label='validation auroc')
## predict test_set

test_pred = model.predict(pad_test)
sample_result = pd.DataFrame()

sample_result['id'] = test_df.index

sample_result['prediction'] = test_pred



## submit sample_submission.csv

sample_result.to_csv('submission.csv', index=False)