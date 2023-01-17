import numpy as np

import pandas as pd

import plotly.express as px



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/chess/games.csv')
data
data['winner'].unique()
data.query("winner != 'draw'")
moves = np.array(

    data.query("winner != 'draw'")['moves']

)
labels = np.array(

    data.query("winner != 'draw'")['winner']

    .apply(lambda x: 1 if x == 'white' else 0)

)
moves.shape
labels.shape
moves
all_moves = set()



for move_list in moves:

    for move in move_list.split(" "):

        if move not in all_moves:

            all_moves.add(move)



max_vocab = len(all_moves)
max_len = 0



for move_list in moves:

    total = 0

    for move in move_list.split(" "):

        total += 1

    if total > max_len:

        max_len = total
print(max_vocab)

print(max_len)
tokenizer = Tokenizer(num_words=max_vocab)

tokenizer.fit_on_texts(moves)



sequences = tokenizer.texts_to_sequences(moves)



word_index = tokenizer.word_index



model_inputs = pad_sequences(sequences, maxlen=max_len)
model_inputs.shape
labels.shape
train_inputs, test_inputs, train_labels, test_labels = train_test_split(model_inputs, labels, train_size=0.7, random_state=24)
embedding_dim = 256



inputs = tf.keras.Input(shape=max_len)



embedding = tf.keras.layers.Embedding(

    input_dim=max_vocab,

    output_dim=embedding_dim,

    input_length=max_len

)(inputs)



gru = tf.keras.layers.GRU(units=embedding_dim)(embedding)



outputs = tf.keras.layers.Dense(1, activation='sigmoid')(gru)





model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.AUC(name='auc')

    ]

)





batch_size = 32

epochs = 3



history = model.fit(

    train_inputs,

    train_labels,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=2

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y': "Loss"},

    title="Loss Over Time"

)



fig.show()
fig = px.line(

    history.history,

    y=['auc', 'val_auc'],

    labels={'x': "Epoch", 'y': "AUC"},

    title="AUC Over Time"

)



fig.show()
model.evaluate(test_inputs, test_labels)