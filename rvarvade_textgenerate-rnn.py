from numpy import array

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding
def generate_seq(model, tokenizer, max_length, seed_text, n_words):

    in_text = seed_text

    for _ in range(n_words):

        encoded = tokenizer.texts_to_sequences([in_text])[0]

        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')

        yhat = model.predict_classes(encoded, verbose=0)

        out_word = ''

        for word, index in tokenizer.word_index.items():

            if index == yhat:

                out_word = word

                break

        in_text += ' ' + out_word

    return in_text

 

data = """ During recent years deep learning has become somewhat of a buzzword in the tech community. 

We always seem to hear about it in news regarding AI, and yet most people don’t actually know what it is! In this article, 

I’ll be demystifying the buzzword that is deep learning, and providing an intuition of how it works. """



tokenizer = Tokenizer()

tokenizer.fit_on_texts([data])

encoded = tokenizer.texts_to_sequences([data])[0]

vocab_size = len(tokenizer.word_index) + 1

print('Vocabulary Size: %d' % vocab_size)

sequences = list()

for i in range(2, len(encoded)):

    sequence = encoded[i-2:i+1]

    sequences.append(sequence)

print('Total Sequences: %d' % len(sequences))

max_length = max([len(seq) for seq in sequences])

sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

print('Max Sequence Length: %d' % max_length)

sequences = array(sequences)

X, y = sequences[:,:-1],sequences[:,-1]

y = to_categorical(y, num_classes=vocab_size)

#LSTM Model

model = Sequential()

model.add(Embedding(vocab_size, 10, input_length=max_length-1))

model.add(LSTM(50))

model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=300, verbose=2)
# Evaluation

print('Text Prediction based on 3 consecutive words:')

print(generate_seq(model, tokenizer, max_length-1, 'recent years deep', 1))



print('\n Generating Text for RNN')

print(generate_seq(model, tokenizer, max_length-1, '', 100))
