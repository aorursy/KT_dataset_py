import pandas as pd



million = pd.read_csv('../input/million-headlines/abcnews-date-text.csv', delimiter=',', nrows=200000)

data = million.drop(['publish_date'], axis=1).rename(columns={'headline_text': 'headline'})
START = '÷'

END = '■'
import numpy as np

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



def format_data(data, max_features, maxlen, shuffle=False):

    if shuffle:

        data = data.sample(frac=1).reset_index(drop=True)

    

    # Add start and end tokens

    data['headline'] = START + ' ' + data['headline'].str.lower() + ' ' + END



    text = data['headline']

    

    # Tokenize text

    filters = "!\"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n"

    tokenizer = Tokenizer(num_words=max_features, filters=filters)

    tokenizer.fit_on_texts(list(text))

    corpus = tokenizer.texts_to_sequences(text)

    

    # Build training sequences of (context, next_word) pairs.

    # Note that context sequences have variable length. An alternative

    # to this approach is to parse the training data into n-grams.

    X, Y = [], []

    for line in corpus:

        for i in range(1, len(line)-1):

            X.append(line[:i+1])

            Y.append(line[i+1])

    

    # Pad X and convert Y to categorical (Y consisted of integers)

    X = pad_sequences(X, maxlen=maxlen)

    Y = to_categorical(Y, num_classes=max_features)



    return X, Y, tokenizer
max_features, max_len = 3500, 20

X, Y, tokenizer = format_data(data, max_features, max_len)
tokenizer.word_index['trump'], tokenizer.word_index[START], tokenizer.word_index[END]
from keras.layers import Input, Dense, Bidirectional, GRU, Embedding, Dropout, LSTM

from keras.layers import concatenate, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.models import Model, Sequential



epochs = 3



model = Sequential()



# Embedding and GRU

model.add(Embedding(max_features, 300))

model.add(SpatialDropout1D(0.33))

model.add(Bidirectional(LSTM(30)))



# Output layer

model.add(Dense(max_features, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=epochs, batch_size=128, verbose=1)



model.save_weights('model{}.h5'.format(epochs))
model.evaluate(X, Y)
def sample(preds, temp=1.0):

    """

    Sample next word given softmax probabilities, using temperature.

    

    Taken and modified from:

    https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

    """

    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temp

    preds = np.exp(preds) / np.sum(np.exp(preds))

    probs = np.random.multinomial(1, preds, 1)

    return np.argmax(probs)
"""When sampling from the distribution, we do not know which word is being

sampled, only its index. We need a way to go from index to word. Unfortunately,

the tokenizer class only contains a dictionary of {word: index} items. We will

reverse that dictionary to get {index: word} items. That way, going from

indices to words is much faster."""

idx_to_words = {value: key for key, value in tokenizer.word_index.items()}





def process_input(text):

    """Tokenize and pad input text"""

    tokenized_input = tokenizer.texts_to_sequences([text])[0]

    return pad_sequences([tokenized_input], maxlen=max_len-1)





def generate_text(input_text, model, n=7, temp=1.0):

    """Takes some input text and feeds it to the model (after processing it).

    Then, samples a next word and feeds it back into the model until the end

    token is produced.

    

    :input_text: A string or list of strings to be used as a generation seed.

    :model:      The model to be used during generation.

    :temp:       A float that adjusts how 'volatile' predictions are. A higher

                 value increases the chance of low-probability predictions to

                 be picked."""

    if type(input_text) is str:

        sent = input_text

    else:

        sent = ' '.join(input_text)

    

    tokenized_input = process_input(input_text)

    

    while True:

        preds = model.predict(tokenized_input, verbose=0)[0]

        pred_idx = sample(preds, temp=temp)

        pred_word = idx_to_words[pred_idx]

        

        if pred_word == END:

            return sent

        

        sent += ' ' + pred_word

#         print(sent)

#         tokenized_input = process_input(sent[-n:])

        tokenized_input = process_input(sent)
text = generate_text(START, model, temp=0.01)

text[2:] # the first two elements are '÷ '
text = generate_text(START, model, temp=0.25)

text[2:] # the first two elements are '÷ '
text = generate_text(START, model, temp=0.5)

text[2:] # the first two elements are '÷ '
text = generate_text(START, model, temp=0.75)

text[2:] # the first two elements are '÷ '
text = generate_text(START, model, temp=1.0)

text[2:] # the first two elements are '÷ '