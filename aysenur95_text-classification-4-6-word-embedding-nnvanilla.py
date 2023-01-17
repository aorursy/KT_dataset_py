import pickle

import pandas as pd

import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

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
print(y_train)

print(type(train_reviews))
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
print(y_train[0])

print(y_train_1hot[0])
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
# word embedding + vanilla NN

from keras.models import Sequential

from keras import layers



embedding_dim = 50



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(50,activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

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
res_final_df=pd.DataFrame({'model_name': 'vanilla_NN', 'train_acc': accuracy_train, 'test_acc': accuracy_test}, index=[0])

#save best model

model.save("vanilla_nn.h5")



#pickle results

with gzip.open('vanilla_nn_results.pkl', 'wb') as output:

    pickle.dump(res_final_df, output, protocol=-1)



        

#pickle tokenizer  

with open('keras-tokenizer.pkl', 'wb') as output:

    pickle.dump(tokenizer, output, protocol=-1)