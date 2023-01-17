import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.head()
from collections import Counter

Counter(" ".join(df["review"]).lower().split()).most_common(100)
#import string as str

#df['review'] = [i.replace('<br>', '').str.replace('</br>', '') for i in df['review']]

df['review'] = df['review'].str.replace('<br />','')

df['review'] = df['review'].str.lower()
plt.figure()

plt.hist(df['review'].str.split().apply(len).value_counts())

plt.xlabel('number of words in sentence')

plt.ylabel('frequency')

plt.title('Words occurrence frequency')
print('The maximum length of a sentence is: ',np.max(df['review'].str.split().apply(len).value_counts()))

print('The average lenth of a sentence is: ', np.average(df['review'].str.split().apply(len).value_counts()))
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

df.head()
sentences = np.array(df['review'])

labels = np.array(df['sentiment'])
training_sentences, testing_sentences,training_labels, testing_labels = train_test_split(sentences, labels, test_size = 0.2)
# choose hyper parameters to tune

vocab_size = 20000 #(before 10000)

embedding_dim = 150 #(before 16)

max_length =  400 #(was 32)

trunc_type = 'post'

padding_type = 'post'

oov_tok = "<OOV>"





# import Tokenizer & fit on training test

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)

tokenizer.fit_on_texts(training_sentences)



word_index = tokenizer.word_index



# convert text to sequences

training_sequences = tokenizer.texts_to_sequences(training_sentences)

training_padded = pad_sequences(training_sequences, maxlen = max_length,

                                padding = padding_type,

                                truncating = trunc_type)



testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

testing_padded = pad_sequences(testing_sequences, maxlen = max_length,

                                padding = padding_type,

                                truncating = trunc_type)



    # modeling

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    

    # option 1: Flatten

    #tf.keras.layers.Flatten(),

    #tf.keras.layers.GlobalAveragePooling1D(),

    

    # option 2: LSTM

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.15),



    # option 3: GRU

    #tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),

    

    # option 4: Conv1D

    #tf.keras.layers.Conv1D(128,5,activation='relu'),

    #tf.keras.layers.GlobalAveragePooling1D(),

    

    tf.keras.layers.Dense(128, activation = 'relu'),

    tf.keras.layers.Dense(1, activation = 'sigmoid')

])



model.summary()
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import LearningRateScheduler



# compile model

model.compile(loss = 'binary_crossentropy',

            optimizer = Adam(learning_rate=0.001),

            metrics = ['accuracy'])



# add early stopping

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



# learning rate decay

def lr_decay(epoch, initial_learningrate = 0.001):#lrv

    return initial_learningrate * 0.9 ** epoch



# training model

num_epochs = 10

history = model.fit(training_padded, training_labels,

                    epochs=num_epochs,

                    callbacks=[LearningRateScheduler(lr_decay),

                              callback],

                    batch_size = 512,

                    validation_data = (testing_padded, testing_labels),

                    verbose=1)
def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history['val_'+string])

    plt.xlabel('Epochs')

    plt.ylabel(string)

    plt.title(print('vocab_size: ',vocab_size))

    plt.legend([string, 'val_' + string])

    plt.show()

    

plot_graphs(history, "accuracy")

plot_graphs(history,"loss")