import numpy as np

import pandas as pd 

import os

import seaborn as sns

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

import string, re

from bs4 import BeautifulSoup

from wordcloud import WordCloud

from keras.preprocessing import text, sequence

from nltk.tokenize.toktok import ToktokTokenizer

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM,Dropout, Bidirectional

from keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
np.random.seed(1)

tf.random.set_seed(1)
train_df = pd.read_csv('../input/nlp-getting-started/train.csv', index_col='id')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv', index_col='id')
train_df.head()
train_df.info()
ax = sns.barplot(x="target", y="target", data=train_df, estimator=lambda x: len(x) / len(train_df.index) * 100)

ax.set(ylabel="Percent")

plt.show()
train_df.keyword.unique()
stop = set(stopwords.words('english'))

# add punctuation to the list of stopwords

punctuation = list(string.punctuation)

stop.update(punctuation)
def strip_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text()



#Removing the square brackets

def remove_between_square_brackets(text):

    return re.sub('\[[^]]*\]', '', text)



# Removing URL's

def remove_url(text):

    return re.sub(r'http\S+', '', text)



def add_space(text):

    return re.sub('%20', ' ', text)



#Removing the stopwords from text

def remove_stopwords(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            final_text.append(i.strip())

    return " ".join(final_text)



#Removing the noisy text

def denoise_text(text):

    text = strip_html(text)

    text = remove_between_square_brackets(text)

    text = add_space(text)

    text = remove_url(text)

    text = remove_stopwords(text)

    return text

def preprocess_df(df):

    df = df.fillna("")

    df['text'] = df['location'] + " " + df['keyword'] + " " + df['text']

    del df['keyword']

    del df['location']

    df['text'] = df['text'].apply(denoise_text)

    return df
train_df = preprocess_df(train_df)

test_df = preprocess_df(test_df)
X_train, X_dev, y_train, y_dev = train_test_split(train_df.text.values, train_df.target.values)
# Set max words and max length hyperparameters

max_features = 10000

max_len = 300
# Fit the tokenizer on the training data

tokenizer = text.Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)
# Tokenize and pad each set of texts

tokenized_train = tokenizer.texts_to_sequences(X_train)

X_train = sequence.pad_sequences(tokenized_train, maxlen=max_len)



tokenized_dev = tokenizer.texts_to_sequences(X_dev)

X_dev = sequence.pad_sequences(tokenized_dev, maxlen=max_len)
EMBEDDING_FILE = '../input/glove-twitter/glove.twitter.27B.100d.txt'
# Create a dictionary of words and their feature vectors from the embedding file

def get_coefs(word, *arr): 

    return word, np.asarray(arr, dtype='float32')



embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))
all_embs = np.stack(list(embeddings_index.values()))

emb_mean,emb_std = all_embs.mean(), all_embs.std()



# Find dims of embedding matrix

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))



# Randomly initialize the embedding matrix

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



# Add each vector to the embedding matrix, corresponding to each token that we set earlier

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
batch_size = 1024

epochs = 15

embed_size = 100
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
#Defining Neural Network

model = Sequential()

#Non-trainable embeddidng layer

model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=max_len, trainable=False))

#LSTM 

model.add(LSTM(units=128 , return_sequences = False , recurrent_dropout = 0.3 , dropout = 0.3))

model.add(Dense(units=64 , activation = 'relu', kernel_regularizer='l2'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, batch_size = batch_size , 

                    validation_data = (X_dev,y_dev) , 

                    epochs = epochs , callbacks = [learning_rate_reduction])
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100)

print("Accuracy of the model on Dev Data is - " , model.evaluate(X_dev,y_dev)[1]*100)
plt.figure(figsize=(10, 10))



epochs = np.arange(epochs)

plt.subplot(2, 2, 1)

plt.xlabel('epochs')

plt.ylabel('loss')

plt.plot(epochs, history.history['loss'])



plt.subplot(2, 2, 2)

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.plot(epochs, history.history['accuracy'])



plt.subplot(2, 2, 3)

plt.xlabel('epochs')

plt.ylabel('val_loss')

plt.plot(epochs, history.history['val_loss'])



plt.subplot(2, 2, 4)

plt.xlabel('epochs')

plt.ylabel('val_accuracy')

plt.plot(epochs, history.history['val_accuracy'])



plt.show()
X_test = test_df.text.values
tokenized_dev = tokenizer.texts_to_sequences(X_test)

X_test = sequence.pad_sequences(tokenized_dev, maxlen=max_len)
classes = model.predict_classes(X_test)[:, 0]
submission = pd.DataFrame(

    {'id': list(test_df.index.values),

     'target': list(classes),

    }).set_index('id')
submission.to_csv('submission.csv')