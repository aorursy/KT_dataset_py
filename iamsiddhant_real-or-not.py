import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
sample=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train.head()
train.info()
train.describe()
train.shape
train.head()
target = train['target']

sns.countplot(target)

train.drop(['target'], inplace =True,axis =1)
def concat_df(train, test):

    return pd.concat([train, test], sort=True).reset_index(drop=True)

df_all = concat_df(train, test)

print(train.shape)

print(test.shape)

print(df_all.shape)
df_all.head()
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = train['text']



train_size = int(7613*0.8)

train_sentences = sentences[:train_size]

train_labels = target[:train_size]



test_sentences = sentences[train_size:]

test_labels = target[train_size:]





vocab_size = 10000

embedding_dim = 16

max_length = 120

trunc_type='post'

oov_tok = "<OOV>"





tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sentences)

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



testing_sequences = tokenizer.texts_to_sequences(test_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(14, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10
train_labels = np.asarray(train_labels)

test_labels = np.asarray(test_labels)
history = model.fit(padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels))
def plot(history,string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string])

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string, 'val_'+string])

    plt.show()

plot(history, "accuracy") 
plot(history, 'loss')
tokenizer_1 = Tokenizer(num_words = vocab_size, oov_token=oov_tok)

tokenizer_1.fit_on_texts(train['text'])



word_index = tokenizer_1.word_index

sequences = tokenizer_1.texts_to_sequences(train['text'])

padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)



true_test_sentences = test['text']

testing_sequences = tokenizer_1.texts_to_sequences(true_test_sentences)

testing_padded = pad_sequences(testing_sequences,maxlen=max_length)
model_2 = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(24, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model_2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model_2.summary()
target = np.asarray(target)
num_epochs = 20

history = model_2.fit(padded, target, epochs=num_epochs, verbose=2)
output = model_2.predict(testing_padded)
predicted =  pd.DataFrame(output, columns=['target'])
final_output = []

for val in predicted.target:

    if val > 0.5:

        final_output.append(1)

    else:

        final_output.append(0)

sample['target'] = final_output



sample.to_csv("submission_1.csv", index=False)

sample.head()