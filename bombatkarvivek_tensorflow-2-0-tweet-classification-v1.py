import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tensorflow as tf

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow_hub as hub
tf.__version__
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.T
train.target.value_counts().plot.pie()
test.T
X_train, X_test, y_train, y_test = train_test_split(train[['text']].values, train[['target']], test_size=0.2, random_state=2012)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)



print(type(X_train.shape))

print(type(X_test.shape))

print(type(y_train.shape))

print(type(y_test.shape))

print(X_train)

y_train.T
tokenizer = Tokenizer()



vocab_size = 10000

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

tokenizer.fit_on_texts(X_train[:,0])
print(tokenizer.get_config()['num_words'])

print(tokenizer.get_config()['document_count'])

print(tokenizer.get_config()['word_counts'][:1000])
text_length = []

for line in X_train[:,0]:

#     print(line)

#     print(len(line.split()))

    text_length.append(len(line.split()))

text_length = np.array(text_length)

text_length.max()

max_len = 20



X_seq_train = pad_sequences(tokenizer.texts_to_sequences(X_train[:, 0]), padding='post', maxlen=max_len)

X_seq_test = pad_sequences(tokenizer.texts_to_sequences(X_test[:, 0]), padding='post', maxlen=max_len)
print(type(X_seq_train))

X_seq_train
sequence = tf.keras.Input(shape=(max_len,))

embdeddings = layers.Embedding(vocab_size, 16)(sequence)



pooling = layers.GlobalAveragePooling1D()(embdeddings)



output = layers.Dense(1, activation='sigmoid')(pooling)



model = tf.keras.Model(inputs=sequence, outputs=output)



model.compile(optimizer=tf.optimizers.Adam(1e-3),

              loss='mae',

              metrics=['accuracy'])



model.summary()
tf.keras.utils.plot_model(model,show_layer_names=True, show_shapes=True)
model.fit(X_seq_train, y_train,

        epochs=21,

        validation_data=(X_seq_test, y_test))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model_loss = pd.DataFrame(model.history.history)

model_loss.head()
model_loss[['accuracy','val_loss']].plot()
sub_padded = pad_sequences(tokenizer.texts_to_sequences(test[['text']].values[:,0]), padding='post', maxlen=max_len)
test['target'] = model.predict(sub_padded)

test[['target']].T
test['target'] = test['target'].apply(lambda x: int(x > 0.5))

test[['target']].T
sub = test[['id', 'target']]
print(sub.shape)

sub.T
sub.target.value_counts().plot.pie()
sub.to_csv('submission.csv', index=False)