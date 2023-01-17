import pickle

import pandas as pd

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras import layers

import gzip
#load datasets



with open("../input/text-classification-2-feature-engineering/df_train.pkl", 'rb') as data:

    df_train = pickle.load(data)





with open("../input/text-classification-2-feature-engineering/df_test.pkl", 'rb') as data:

    df_test = pickle.load(data)

    
train_reviews = df_train['review_parsed'].values

test_reviews = df_test['review_parsed'].values



y_train = df_train['condition'].values

y_test = df_test['condition'].values
tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(train_reviews)



X_train = tokenizer.texts_to_sequences(train_reviews)

X_test = tokenizer.texts_to_sequences(test_reviews)



vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index



#look at example

print(train_reviews[2])

print(X_train[2])
#nn'ne verirken review uzunluklarını eşitliyoruz

maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



print(X_train[0, :])
from keras.utils import to_categorical



# Convert the labels to one_hot_category values

y_train_1hot = to_categorical(y_train, num_classes = 10)

y_test_1hot = to_categorical(y_test, num_classes = 10)


def create_embedding_matrix(filepath, word_index, embedding_dim):

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    with open(filepath) as f:

        for line in f:

            word, *vector = line.split()

            if word in word_index:

                idx = word_index[word] 

                embedding_matrix[idx] = np.array(

                    vector, dtype=np.float32)[:embedding_dim]



    return embedding_matrix
embedding_dim = 50

embedding_matrix = create_embedding_matrix('../input/glove6b/glove.6B.50d.txt',tokenizer.word_index, embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

nonzero_elements / vocab_size
import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['categorical_accuracy']

    val_acc = history.history['val_categorical_accuracy']

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
model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))

model.add(layers.Conv1D(filters=50, kernel_size=3, activation='relu'))

model.add(layers.MaxPooling1D(3))

model.add(layers.Dropout(0.5))

model.add(layers.Flatten())

model.add(layers.Dense(units=50, activation='relu'))

model.add(layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['categorical_accuracy'])

model.summary()
history = model.fit(X_train, y_train_1hot,

                    epochs=20,

                    verbose=True,

                    validation_data=(X_test, y_test_1hot),

                    batch_size=512)



loss_train, accuracy_train = model.evaluate(X_train, y_train_1hot, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy_train))



loss_test, accuracy_test = model.evaluate(X_test, y_test_1hot, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy_test))



plot_history(history)
res_final_df=pd.DataFrame({'model_name': 'glove_CNN', 'train_acc': accuracy_train, 'test_acc': accuracy_test}, index=[0])
#save best model

model.save("glove_cnn.h5")



#pickle results

with gzip.open('glove_cnn_results.pkl', 'wb') as output:

    pickle.dump(res_final_df, output, protocol=-1)

    

#pickle tokenizer  

with open('keras-tokenizer.pkl', 'wb') as output:

    pickle.dump(tokenizer, output, protocol=-1)



#pickle embedding_matrix  

with open('embedding_matrix.pkl', 'wb') as output:

    pickle.dump(embedding_matrix, output, protocol=-1)



    