from keras import Model, Input

from keras import models

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.callbacks import ModelCheckpoint



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
data = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv",

                   names=['review', 'сlass'],

                   header=0,

                   encoding='utf-8')
x_train = data['review']

y_train = data['сlass']

del data
print(x_train[0])

print(y_train[0])

print(len(x_train))
max_words = 20_000
tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)

del tokenizer
def vectorize(sequences, dimension):

    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):

        for j in sequence:

            results[i, j] += 1

    return results/np.amax(results)
x_train = vectorize(x_train, max_words)
y_train = [1 if i == 'positive' else 0 for i in y_train]
x_test = x_train[:10_000]

y_test = y_train[:10_000]



x_train = x_train[10_000:]

y_train = y_train[10_000:]
x_test = np.array(x_test)

y_test = np.array(y_test)



x_train = np.array(x_train)

y_train = np.array(y_train)
branch = 256



dropout = 0.4



epochs = 500



onehot_input = Input(shape=(max_words,), name='onehot') 



features = layers.Dropout(dropout)(onehot_input)



features = layers.Dense(128, activation='sigmoid')(features)

features = layers.Dropout(dropout)(features)



features = layers.Dense(64, activation='sigmoid')(features)

features = layers.Dropout(dropout)(features)



output = layers.Dense(1, activation = "sigmoid", name="ispositive")(features)



model = Model(

    inputs=[onehot_input],

    outputs=[output],

)



model.compile(

    optimizer = 'rmsprop',

    loss = "binary_crossentropy",

    metrics = ["accuracy"]

)



print(model.summary())
model_save_path = 'best_model.h5'

checkpoint_callback = ModelCheckpoint(model_save_path, 

                                      monitor='val_accuracy',

                                      save_best_only=True,

                                      verbose=1)
results = model.fit(

    {"onehot": x_train},

    {"ispositive": y_train},

    epochs=epochs,

    batch_size=branch,

    validation_data=({"onehot": x_test},

                     {"ispositive": y_test}),

    callbacks=[checkpoint_callback],

)
model.load_weights(model_save_path)
plt.plot(results.history['accuracy'], label='Train')

plt.plot(results.history['val_accuracy'], label='Test')

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.legend()

plt.show()
scores = model.evaluate(x_test, y_test, verbose=1)
scores