from __future__ import absolute_import, division, print_function, unicode_literals

import datetime

import tensorflow as tf

from tensorflow.keras import regularizers

from tensorflow import keras

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import scipy as sp



from keras.callbacks import TensorBoard

# Load the TensorBoard notebook extension



import time



from tensorflow import feature_column

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split



import os
%load_ext tensorboard


!rm -rf ./logs/ 


df=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df1=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df3=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
df.head()
df.isnull().sum()
df1.isnull().sum()
df.describe()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False)
df1.isnull().sum()
tr=df['text']

tr1=df1['text']
tar=df['target']
sent=[]

for sentence in tr:

  sent.append(sentence)

print(sent[0])
sent1=[]

for sentence in tr1:

  sent1.append(sentence)

print(sent1[0])
label=[]

for sentence in tar:

  label.append(sentence)

print(label[0])
tokenizer = tf.keras.preprocessing.text.Tokenizer()

tokenizer.fit_on_texts(sent)

vocab_size = len(tokenizer.word_counts)

print("vocabulary size: {:d}".format(vocab_size))

word2idx = tokenizer.word_index

idx2word = {v:k for (k, v) in word2idx.items()}
seq_lengths = np.array([len(s.split()) for s in sent])

print([(p, np.percentile(seq_lengths, p)) for p

 in [75, 80, 90, 95, 99, 100]])
max_seqlen = 31



sentences_as_ints = tokenizer.texts_to_sequences(sent)

sentences_as_ints = tf.keras.preprocessing.sequence.pad_sequences(

 sentences_as_ints, maxlen=max_seqlen)

labels_as_ints = np.array(label)



dataset = tf.data.Dataset.from_tensor_slices(

 (sentences_as_ints, labels_as_ints))

label1=[]

for i in range(len(sent1)):

  label1.append(0)


sentences_as_ints = tokenizer.texts_to_sequences(sent1)

sentences_as_ints = tf.keras.preprocessing.sequence.pad_sequences(

 sentences_as_ints, maxlen=max_seqlen)

labels_as_ints = np.array(label1)

dataset1 = tf.data.Dataset.from_tensor_slices(

 (sentences_as_ints, labels_as_ints))




dataset = dataset.shuffle(300)











test_size = len(sent) // 8

val_size = (len(sent)) // 6

test_dataset = dataset.take(test_size)

val_dataset = dataset.skip(test_size).take(val_size)

train_dataset = dataset.skip( test_size + val_size)

batch_size = 64

train_dataset = train_dataset.batch(batch_size)

val_dataset = val_dataset.batch(batch_size)

test_dataset = test_dataset.batch(batch_size)

print(val_size)

print(len(sent))

print(test_size)
pred_dataset = dataset1.batch(batch_size)
class TextclassificationModel(tf.keras.Model):

  def __init__(self, vocab_size, max_seqlen, **kwargs):

    super(TextclassificationModel, self).__init__(**kwargs)

    self.embedding = tf.keras.layers.Embedding(

    vocab_size, max_seqlen)

    self.bilstm = tf.keras.layers.Bidirectional(

    tf.keras.layers.LSTM(max_seqlen)

    )

    

    self.dense = tf.keras.layers.Dense(64, activation="relu",kernel_regularizer=regularizers.l2(0.05))

    

    self.out = tf.keras.layers.Dense(1, activation="sigmoid")

  def call(self, x):

    x = self.embedding(x)

    x = self.bilstm(x)

    x = self.dense(x)



    x = self.out(x)

    return x

model = TextclassificationModel(vocab_size+1, max_seqlen)

model.build(input_shape=(batch_size, max_seqlen))

model.summary()

# compile

model.compile(

 loss="binary_crossentropy",

 optimizer="adam",

 metrics=["accuracy"]

)

data_dir = "./"

logs_dir = os.path.join("./logs")



best_model_file = os.path.join(data_dir, "best_model.h5")

checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model_file,

 save_weights_only=True,

 save_best_only=True)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

num_epochs = 10

history = model.fit(train_dataset, epochs=num_epochs,

 validation_data=val_dataset,

 callbacks=[checkpoint, tensorboard])

model = TextclassificationModel(vocab_size+1, max_seqlen)

model.build(input_shape=(batch_size, max_seqlen))

model.summary()

best_model = TextclassificationModel(vocab_size+1, max_seqlen)

best_model.build(input_shape=(batch_size, max_seqlen))

best_model.load_weights(best_model_file)

best_model.compile(

 loss="binary_crossentropy",

 optimizer="adam",

 metrics=["accuracy"]

)
test_loss, test_acc = best_model.evaluate(test_dataset)

print("test loss: {:.3f}, test accuracy: {:.3f}".format(

 test_loss, test_acc))
pred=best_model.predict(pred_dataset)
g=[]

for n in pred:

  if n>0.5:

     g.append([1])



  else:

    g.append([0])

 

 



print("p", g)





prediction=pd.DataFrame(g)
df3.pop('target')
df3.info()
df3=pd.concat([df3['id'],prediction],axis=1)
df3.columns=['id','target']
df3.head()
df3.to_csv('f5text_classification_submission.csv',index=False)