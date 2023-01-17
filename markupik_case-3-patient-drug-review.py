import os

import math

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.utils import compute_class_weight

from sklearn.model_selection import train_test_split

from keras import preprocessing

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras import layers

from keras import regularizers

import seaborn as sb

from sklearn.metrics import confusion_matrix 



SAMPLE_SIZE = 8000

MAX_LEN = 170 #cuts off reviews after 170 words

MAX_WORDS = 10000 #considers only the top 10000 words in the dataset

EPOCHS = 25

BATCH_SIZE = 32
csv_folder = '../input/kuc-hackathon-winter-2018/'

train_df = pd.read_csv(csv_folder + 'drugsComTrain_raw.csv')

test_df = pd.read_csv(csv_folder + 'drugsComTest_raw.csv')



train_df.head(6)
sample_df = train_df.sample(SAMPLE_SIZE)

test_sample_df = test_df.sample(SAMPLE_SIZE)

print('Records: Train: {} Test: {}'.format(sample_df.shape[0],test_sample_df.shape[0]))



stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text



sample_df['processed_review'] = sample_df.review.apply(lambda x: clean_text(x))

test_sample_df['processed_review'] = test_sample_df.review.apply(lambda x: clean_text(x))
def ratingCov(rating):

    if rating > 8:

        return 4

    elif rating > 6:

        return 3

    elif rating > 4:

        return 2

    elif rating > 2:

        return 1

    else:

        return 0



sample_df['category'] = sample_df.rating.apply(lambda x: ratingCov(x))

test_sample_df['category'] = test_sample_df.rating.apply(lambda x: ratingCov(x))

sample_df['category'].tail(10)
sample_df['rating'].hist(bins= np.arange(1,10), align='left')
catNums = sorted(sample_df['category'].unique())



w = compute_class_weight('balanced', range(0,5), sample_df['category'])

classWeights = dict(zip(np.arange(0,5), w))

classWeights
tokenizer = preprocessing.text.Tokenizer(num_words=MAX_WORDS)



tokenizer.fit_on_texts(sample_df['processed_review'])



#tokenizes and pads out the tokenized strings

def toSequenceAndPad(series):

    sequences = tokenizer.texts_to_sequences(series)

    data = pad_sequences(sequences, maxlen=MAX_LEN)

    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    

    m = 0

    for i,arr in enumerate(sequences):

        m += len(arr)

    mean = m/len(sequences)

    s = 0

    for i,arr in enumerate(sequences):

        s += abs(mean - len(arr))

    std = s/len(sequences)

    print('Length of reviews. Mean: {:.2f} STD: {:.2f}'.format(mean,std))

    

    return (data, word_index)

data, word_index = toSequenceAndPad(sample_df['processed_review'])

test_data, test_word_index = toSequenceAndPad(test_sample_df['processed_review'])
labels = to_categorical(sample_df['category'])

test_labels = to_categorical(test_sample_df['category'])
x_train, x_val, y_train, y_val = train_test_split(data,

                                                  labels,

                                                  test_size = 0.33)
glove_dir = '../input/glove-global-vectors-for-word-representation/'



embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 100



embedding_matrix = np.zeros((MAX_WORDS, embedding_dim))

for word, i in word_index.items():

    if i < MAX_WORDS:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector

            
model = Sequential()

model.add(layers.Embedding(MAX_WORDS, embedding_dim, input_length=MAX_LEN))

model.add(layers.Bidirectional(

    layers.LSTM(8,

                dropout=0.1,

                recurrent_dropout=0.2,

                kernel_regularizer=regularizers.l2(0.002),

                return_sequences=True

               ))

         )

model.add(layers.Bidirectional(

    layers.LSTM(8,

                dropout=0.1,

                recurrent_dropout=0.2,

                kernel_regularizer=regularizers.l2(0.002),

                return_sequences=True

               ))

         )

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(20, kernel_regularizer=regularizers.l2(0.005), activation='sigmoid'))

model.add(layers.Dropout(0.11))

model.add(layers.Dense(5, activation='softmax'))
model.layers[0].set_weights([embedding_matrix])

model.layers[0].traiable = False
model.compile(loss='categorical_crossentropy',

              optimizer='rmsprop',

              metrics=['acc'])
hist = model.fit(x_train, y_train,

          validation_data = (x_val, y_val),

          epochs=EPOCHS,

          batch_size=BATCH_SIZE,

          class_weight=classWeights)
# We use the same plotting commands several times, so create a function for that purpose

def plot_history(history, info):

    

    f, ax = plt.subplots(1, 2, figsize = (14, 7))

    

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    f_acc = val_acc[len(val_acc)-1]

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    f_loss = val_loss[len(val_loss)-1]

    

    epochs = range(1, len(loss) + 1)



    #plot loss

    plt.sca(ax[0])

    plt.plot(epochs, loss, 'b', label='Training loss')

    plt.plot(epochs, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend(['training', 'validation'])

    plt.title('Loss, final validation loss: {:.2f}'.format(f_loss))

    

    #plot accuracy

    plt.sca(ax[1])

    plt.plot(epochs, acc, 'b', label='Training acc')

    plt.plot(epochs, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend(['training', 'validation'])

    plt.title('Accuracy, final validation acc: {:.2f}'.format(f_acc))

    

    #additional info from elsewhere

    plt.figtext(0.1,0, info)



    plt.savefig('results.png')

    plt.show()
res = model.evaluate(test_data, test_labels)
plot_history(hist, 'Test results:\nLoss: {:.2f} Acc: {:.2f}'.format(res[0],res[1]))
pred = model.predict(test_data)
cm = confusion_matrix(test_sample_df['category'], np.argmax(pred, axis=1))
heatMap = sb.heatmap(cm, annot=True, fmt='d')

heatMap.set(xlabel='pred',ylabel='actual')

plt.savefig('confusion_matrix.png')

plt.show()