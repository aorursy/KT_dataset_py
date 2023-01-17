# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob
from sklearn import model_selection 
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import re
from keras.preprocessing.text import Tokenizer
from numpy import random
from random import randint
import matplotlib.pyplot as plt
#from tensorflow.python.keras.utils.data_utils import Sequence
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


   
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    

def _data_generator(x, y, num_features, batch_size):
    """Generates batches of vectorized texts for training/validation.

    # Arguments
        x: np.matrix, feature matrix.
        y: np.ndarray, labels.
        num_features: int, number of features.
        batch_size: int, number of samples per batch.

    # Returns
        Yields feature and label data in batches.
    """
    num_samples = x.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    while 1:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_samples:
                end_idx = num_samples
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            yield x_batch, y_batch
    
jigsaw_train_df = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
jigsaw_unbias_df = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
len(jigsaw_train_df)
len(jigsaw_unbias_df)
print(jigsaw_train_df.columns)
jigsaw_unbias_df = jigsaw_unbias_df.rename(columns={"severe_toxicity":"severe_toxic","identity_attack":"identity_hate"})
print(jigsaw_unbias_df.columns)
jigsaw_df = pd.concat([jigsaw_train_df,jigsaw_unbias_df])
print(len(jigsaw_df))
jigsaw_df = jigsaw_df.dropna(subset=["comment_text"])
jigsaw_df = jigsaw_df.drop_duplicates(subset=["comment_text"])
print(len(jigsaw_df))
jigsaw_df.insult = np.rint(jigsaw_df.insult)
jigsaw_df.toxic = np.rint(jigsaw_df.toxic)
jigsaw_df.severe_toxic = np.rint(jigsaw_df.severe_toxic)
jigsaw_df.obscene = np.rint(jigsaw_df.obscene)
jigsaw_df.identity_hate = np.rint(jigsaw_df.identity_hate)
jigsaw_df.threat = np.rint(jigsaw_df.threat)
jigsaw_df.insult.unique()
text_train, text_val, y_train, y_val = model_selection.train_test_split(jigsaw_df["comment_text"],jigsaw_df["insult"],test_size=0.3)
print("type(y_train)",type(y_train))
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
print("type(y_train)",type(y_train))
from keras.preprocessing.text import Tokenizer
# Tokenize and transform to integer index
tokenizer = Tokenizer(oov_token=True,num_words=10000)
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_val = tokenizer.texts_to_sequences(text_val)

vocab_size = 10000 #len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = 128 #max(len(x) for x in X_train) # longest text in train set
batch_size = 32

print("vocab_size",vocab_size)
print("maxlen",maxlen)

# Add pading to ensure all vectors have same dimensionality

print(len(X_train), "Training sequences")
print(len(X_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=maxlen)
print("type(x_train)",type(x_train))
x_train = np.asmatrix(x_train)
print("type(x_train)",type(x_train))
x_val = np.asmatrix(x_val)

training_generator = _data_generator(
    x_train, y_train, maxlen, batch_size)
validation_generator = _data_generator(
    x_val, y_val, maxlen, batch_size)


# Get number of training steps. This indicated the number of steps it takes
# to cover all samples in one epoch.
steps_per_epoch = x_train.shape[0] // batch_size
if x_train.shape[0] % batch_size:
    steps_per_epoch += 1


# Get number of validation steps.
validation_steps = x_val.shape[0] // batch_size
if x_val.shape[0] % batch_size:
    validation_steps += 1


embed_dim = 128  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

## Create classifier model using transformer layer

#Transformer layer outputs one vector for each time step of our input sequence.
#Here, we take the mean across all time steps and
#use a feed forward network on top of it to classify text.

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# Train and validate model.
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=2)  # Logs once per epoch.

#model.save("/content/drive/My Drive/Colab Notebooks/CB_Data/transformer_block_Jigsaw_insults")

%matplotlib inline
plot_history(history)

from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
predicted_labels = model.predict(x_val)
print("AUC score", roc_auc_score(np.rint(y_val),predicted_labels))
print("F1 score", f1_score(np.rint(y_val),np.rint(predicted_labels)))
##Toxicity
text_train, text_val, y_train, y_val = model_selection.train_test_split(jigsaw_df["comment_text"],jigsaw_df["toxic"],test_size=0.3)
print("type(y_train)",type(y_train))
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
print("type(y_train)",type(y_train))
from keras.preprocessing.text import Tokenizer
# Tokenize and transform to integer index
tokenizer = Tokenizer(oov_token=True,num_words=10000)
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_val = tokenizer.texts_to_sequences(text_val)

vocab_size = 10000 #len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = 200 #max(len(x) for x in X_train) # longest text in train set
batch_size = 32

print("vocab_size",vocab_size)
print("maxlen",maxlen)

# Add pading to ensure all vectors have same dimensionality

print(len(X_train), "Training sequences")
print(len(X_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=maxlen)
print("type(x_train)",type(x_train))
x_train = np.asmatrix(x_train)
print("type(x_train)",type(x_train))
x_val = np.asmatrix(x_val)

training_generator = _data_generator(
    x_train, y_train, maxlen, batch_size)
validation_generator = _data_generator(
    x_val, y_val, maxlen, batch_size)


# Get number of training steps. This indicated the number of steps it takes
# to cover all samples in one epoch.
steps_per_epoch = x_train.shape[0] // batch_size
if x_train.shape[0] % batch_size:
    steps_per_epoch += 1


# Get number of validation steps.
validation_steps = x_val.shape[0] // batch_size
if x_val.shape[0] % batch_size:
    validation_steps += 1


embed_dim = 128  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

## Create classifier model using transformer layer

#Transformer layer outputs one vector for each time step of our input sequence.
#Here, we take the mean across all time steps and
#use a feed forward network on top of it to classify text.

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# Train and validate model.
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=2)  # Logs once per epoch.

#model.save("/content/drive/My Drive/Colab Notebooks/CB_Data/transformer_block_Jigsaw_insults")

%matplotlib inline
plot_history(history)

from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
predicted_labels = model.predict(x_val)
print("AUC score", roc_auc_score(np.rint(y_val),predicted_labels))
print("F1 score", f1_score(np.rint(y_val),np.rint(predicted_labels)))
text_train, text_val, y_train, y_val = model_selection.train_test_split(jigsaw_df["comment_text"],jigsaw_df["obscene"],test_size=0.3)
print("type(y_train)",type(y_train))
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
print("type(y_train)",type(y_train))
from keras.preprocessing.text import Tokenizer
# Tokenize and transform to integer index
tokenizer = Tokenizer(oov_token=True,num_words=10000)
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_val = tokenizer.texts_to_sequences(text_val)

vocab_size = 10000 #len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = 200 #max(len(x) for x in X_train) # longest text in train set
batch_size = 32

print("vocab_size",vocab_size)
print("maxlen",maxlen)

# Add pading to ensure all vectors have same dimensionality

print(len(X_train), "Training sequences")
print(len(X_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=maxlen)
print("type(x_train)",type(x_train))
x_train = np.asmatrix(x_train)
print("type(x_train)",type(x_train))
x_val = np.asmatrix(x_val)

training_generator = _data_generator(
    x_train, y_train, maxlen, batch_size)
validation_generator = _data_generator(
    x_val, y_val, maxlen, batch_size)


# Get number of training steps. This indicated the number of steps it takes
# to cover all samples in one epoch.
steps_per_epoch = x_train.shape[0] // batch_size
if x_train.shape[0] % batch_size:
    steps_per_epoch += 1


# Get number of validation steps.
validation_steps = x_val.shape[0] // batch_size
if x_val.shape[0] % batch_size:
    validation_steps += 1


embed_dim = 128  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

## Create classifier model using transformer layer

#Transformer layer outputs one vector for each time step of our input sequence.
#Here, we take the mean across all time steps and
#use a feed forward network on top of it to classify text.

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# Train and validate model.
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=2)  # Logs once per epoch.

#model.save("/content/drive/My Drive/Colab Notebooks/CB_Data/transformer_block_Jigsaw_insults")

%matplotlib inline
plot_history(history)

from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
predicted_labels = model.predict(x_val)
print("AUC score", roc_auc_score(np.rint(y_val),predicted_labels))
print("F1 score", f1_score(np.rint(y_val),np.rint(predicted_labels)))
text_train, text_val, y_train, y_val = model_selection.train_test_split(jigsaw_df["comment_text"],jigsaw_df["threat"],test_size=0.3)
print("type(y_train)",type(y_train))
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
print("type(y_train)",type(y_train))
from keras.preprocessing.text import Tokenizer
# Tokenize and transform to integer index
tokenizer = Tokenizer(oov_token=True,num_words=10000)
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_val = tokenizer.texts_to_sequences(text_val)

vocab_size = 10000 #len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = 200 #max(len(x) for x in X_train) # longest text in train set
batch_size = 32

print("vocab_size",vocab_size)
print("maxlen",maxlen)

# Add pading to ensure all vectors have same dimensionality

print(len(X_train), "Training sequences")
print(len(X_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=maxlen)
print("type(x_train)",type(x_train))
x_train = np.asmatrix(x_train)
print("type(x_train)",type(x_train))
x_val = np.asmatrix(x_val)

training_generator = _data_generator(
    x_train, y_train, maxlen, batch_size)
validation_generator = _data_generator(
    x_val, y_val, maxlen, batch_size)


# Get number of training steps. This indicated the number of steps it takes
# to cover all samples in one epoch.
steps_per_epoch = x_train.shape[0] // batch_size
if x_train.shape[0] % batch_size:
    steps_per_epoch += 1


# Get number of validation steps.
validation_steps = x_val.shape[0] // batch_size
if x_val.shape[0] % batch_size:
    validation_steps += 1


embed_dim = 128  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

## Create classifier model using transformer layer

#Transformer layer outputs one vector for each time step of our input sequence.
#Here, we take the mean across all time steps and
#use a feed forward network on top of it to classify text.

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# Train and validate model.
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=2)  # Logs once per epoch.

#model.save("/content/drive/My Drive/Colab Notebooks/CB_Data/transformer_block_Jigsaw_insults")

%matplotlib inline
plot_history(history)

from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
predicted_labels = model.predict(x_val)
print("AUC score", roc_auc_score(np.rint(y_val),predicted_labels))
print("F1 score", f1_score(np.rint(y_val),np.rint(predicted_labels)))
text_train, text_val, y_train, y_val = model_selection.train_test_split(jigsaw_df["comment_text"],jigsaw_df["identity_hate"],test_size=0.3)
print("type(y_train)",type(y_train))
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
print("type(y_train)",type(y_train))
from keras.preprocessing.text import Tokenizer
# Tokenize and transform to integer index
tokenizer = Tokenizer(oov_token=True,num_words=10000)
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_val = tokenizer.texts_to_sequences(text_val)

vocab_size = 10000 #len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = 200 #max(len(x) for x in X_train) # longest text in train set
batch_size = 32

print("vocab_size",vocab_size)
print("maxlen",maxlen)

# Add pading to ensure all vectors have same dimensionality

print(len(X_train), "Training sequences")
print(len(X_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=maxlen)
print("type(x_train)",type(x_train))
x_train = np.asmatrix(x_train)
print("type(x_train)",type(x_train))
x_val = np.asmatrix(x_val)

training_generator = _data_generator(
    x_train, y_train, maxlen, batch_size)
validation_generator = _data_generator(
    x_val, y_val, maxlen, batch_size)


# Get number of training steps. This indicated the number of steps it takes
# to cover all samples in one epoch.
steps_per_epoch = x_train.shape[0] // batch_size
if x_train.shape[0] % batch_size:
    steps_per_epoch += 1


# Get number of validation steps.
validation_steps = x_val.shape[0] // batch_size
if x_val.shape[0] % batch_size:
    validation_steps += 1


embed_dim = 128  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

## Create classifier model using transformer layer

#Transformer layer outputs one vector for each time step of our input sequence.
#Here, we take the mean across all time steps and
#use a feed forward network on top of it to classify text.

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# Train and validate model.
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=2)  # Logs once per epoch.

#model.save("/content/drive/My Drive/Colab Notebooks/CB_Data/transformer_block_Jigsaw_insults")

%matplotlib inline
plot_history(history)

from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
predicted_labels = model.predict(x_val)
print("AUC score", roc_auc_score(np.rint(y_val),predicted_labels))
print("F1 score", f1_score(np.rint(y_val),np.rint(predicted_labels)))
text_train, text_val, y_train, y_val = model_selection.train_test_split(jigsaw_df["comment_text"],jigsaw_df["severe_toxic"],test_size=0.3)
print("type(y_train)",type(y_train))
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
print("type(y_train)",type(y_train))
from keras.preprocessing.text import Tokenizer
# Tokenize and transform to integer index
tokenizer = Tokenizer(oov_token=True,num_words=10000)
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train)
X_val = tokenizer.texts_to_sequences(text_val)

vocab_size = 10000 #len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = 200 #max(len(x) for x in X_train) # longest text in train set
batch_size = 32

print("vocab_size",vocab_size)
print("maxlen",maxlen)

# Add pading to ensure all vectors have same dimensionality

print(len(X_train), "Training sequences")
print(len(X_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=maxlen)
print("type(x_train)",type(x_train))
x_train = np.asmatrix(x_train)
print("type(x_train)",type(x_train))
x_val = np.asmatrix(x_val)

training_generator = _data_generator(
    x_train, y_train, maxlen, batch_size)
validation_generator = _data_generator(
    x_val, y_val, maxlen, batch_size)


# Get number of training steps. This indicated the number of steps it takes
# to cover all samples in one epoch.
steps_per_epoch = x_train.shape[0] // batch_size
if x_train.shape[0] % batch_size:
    steps_per_epoch += 1


# Get number of validation steps.
validation_steps = x_val.shape[0] // batch_size
if x_val.shape[0] % batch_size:
    validation_steps += 1


embed_dim = 128  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer

## Create classifier model using transformer layer

#Transformer layer outputs one vector for each time step of our input sequence.
#Here, we take the mean across all time steps and
#use a feed forward network on top of it to classify text.

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
# Train and validate model.
history = model.fit_generator(
    generator=training_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    epochs=2)  # Logs once per epoch.

#model.save("/content/drive/My Drive/Colab Notebooks/CB_Data/transformer_block_Jigsaw_insults")

%matplotlib inline
plot_history(history)

from sklearn.metrics import roc_auc_score,f1_score
import numpy as np
predicted_labels = model.predict(x_val)
print("AUC score", roc_auc_score(np.rint(y_val),predicted_labels))
print("F1 score", f1_score(np.rint(y_val),np.rint(predicted_labels)))