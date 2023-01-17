import numpy as np

import tensorflow as tf



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras import layers, models
# Read the file

fp = open('../input/christmas-carol/carol.txt','r')

data = fp.read().splitlines()        

fp.close()



# Encode the data

tokens = text.Tokenizer()

tokens.fit_on_texts(data)

data_sequences = tokens.texts_to_sequences(data)

vocab_size = len(tokens.word_counts) + 1



# generate the sequence

seq_list = list()

for item in data_sequences:

    l = len(item)

    for id in range(1, l):

        seq_list.append(item[: id+1])

        

max_length = max([len(seq) for seq in seq_list])

data_sequences_matrix = sequence.pad_sequences(seq_list, maxlen = max_length, padding = 'pre')

data_sequences_matrix = np.array(data_sequences_matrix)



# separate input data X and corresponding output y

X = data_sequences_matrix[:, :-1]

y = data_sequences_matrix[:, -1]

y = to_categorical(y, num_classes = vocab_size)
lstm_model = models.Sequential()

lstm_model.add(layers.Input(shape = [max_length-1]))

lstm_model.add(layers.Embedding(vocab_size, 10, input_length = max_length-1))

lstm_model.add(layers.LSTM(50))

lstm_model.add(layers.Dropout(0.1))               

lstm_model.add(layers.Dense(vocab_size, activation = 'softmax'))

lstm_model.summary()
lstm_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = lstm_model.fit(X, y, batch_size = 2, epochs = 50)
lstm_model.save('./saved_model/')
idx2word = {v:k for k,v in tokens.word_index.items()}

new_model = models.load_model('./saved_model/')



# function to make predictions, it takes text as input and predict *num_words* possible after this text

def predict_words(text, num_words):

    encoded_data = tokens.texts_to_sequences([text])[0]

    padded_data = sequence.pad_sequences([encoded_data], maxlen = max_length - 1, padding = 'pre')

    y_preds = new_model.predict(padded_data)

    y_preds = np.argsort(-y_preds)

    y_preds = y_preds[0][:num_words]

    possible_words = [idx2word[item] for item in y_preds]

    print(text, possible_words)
predict_words("how to", 2)

predict_words("find a", 2)

predict_words("Merry", 2)

predict_words("I am", 2)

predict_words("how", 2)