import os

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt
base_dir = '../input/'

train_dir = os.path.join(base_dir, 'train.tsv')

test_dir = os.path.join(base_dir, 'test.tsv')



Train = pd.read_csv(train_dir, sep='\t')

Test = pd.read_csv(test_dir, sep='\t')



print('Training Shape: ', Train.shape)

print('Test Shape: ', Test.shape)

print('Class Distribution in Training Data: \n{}'.format(Train['Sentiment'].value_counts()/Train.shape[0]))
from keras.preprocessing.text import Tokenizer



# Tokenizer Object

tokenizer = Tokenizer(num_words=None,

                  filters='!"#$%&()*+-/<=>?@[\\]^_`{|}~\t\n',

                  lower=True,

                  split=' ')

tokenizer.fit_on_texts(np.append(Train['Phrase'].values, Test['Phrase'].values))



# Convert phrase to list of integers

Train['phrase_idx'] = pd.Series(np.array(tokenizer.texts_to_sequences(Train['Phrase'])))

Test['phrase_idx'] = pd.Series(np.array(tokenizer.texts_to_sequences(Test['Phrase'])))



word_index = tokenizer.word_index

index_word = tokenizer.index_word



# Number of words in vocab

num_words = len(word_index)+1



# One-hot encoding Labels

#Train['Sentiment'] = Train['Sentiment'].apply(lambda x: np.array([0 if x!=i else 1 for i in range(5)]))



Train['word_count'] = Train['phrase_idx'].apply(lambda x : len(x))

maxlen = Train['word_count'].max()
maxlen
from keras.preprocessing import sequence

Xtrain = sequence.pad_sequences(Train['phrase_idx'], maxlen)

Xtest= sequence.pad_sequences(Test['phrase_idx'], maxlen)

Ytrain = Train['Sentiment'].values
Train['Sentiment'].value_counts().plot.bar()
# Model

from keras.layers import (LSTM, Embedding, Dense, Input, Masking,

Dropout, Flatten)

from keras.models import Model, Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint



num_class = 5



model = Sequential()

# Embedding Layer

model.add(Embedding(input_dim=num_words,

                    output_dim = 250,

                    mask_zero=True,

                    name='Embedding')

         )

model.add(Masking())



# Lstm

model.add(LSTM(132,

               return_sequences=True,

               dropout=0.5,

               recurrent_dropout=0.5

               ))

model.add(LSTM(64,

               return_sequences=False,

               dropout=0.2,

               recurrent_dropout=0.2))



#model.add(Dense(64, activation='relu'))

#model.add(Dropout(0.4))

# Output layer

model.add(Dense(num_class, activation='softmax', name='output'))





          
model.summary()
model.compile(loss='sparse_categorical_crossentropy',

                        optimizer='adam',

                        metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_acc',

                               patience=3,

                               verbose=1)

model_checkpoint = ModelCheckpoint(filepath='Tsa_5.hdf5', save_best_only=True, verbose=1)
history = model.fit(Xtrain,

                    Ytrain,

                    epochs=10,

                    batch_size=1024,

                    validation_split=0.2,

                    shuffle=True,

                    callbacks=[model_checkpoint, early_stopping],

                    verbose=1

                    )
# load model

import keras

model = keras.models.load_model('Tsa_5.hdf5')

emb_layer = model.get_layer('Embedding')

(w,) = emb_layer.get_weights()

print(len(w))

#w = w[Train.index]

#df = Train
from sklearn.manifold import TSNE



tsne = TSNE(random_state=1,n_components=2, n_iter=1000, metric='cosine')

embs = tsne.fit_transform(w)

#df = pd.DataFrame()

df['X'] = embs[:, 0]

df['Y'] = embs[:, 1]

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(df['X'], df['Y'], alpha=0.1)
pred = model.predict(Xtest)

Sentiment = np.array([p.argmax() for p in pred])


output = pd.DataFrame({'PhraseId':Test.PhraseId,

                       'Sentiment':Sentiment})

output.to_csv('submission.csv', index=False)
output['Sentiment'].value_counts().plot.bar()