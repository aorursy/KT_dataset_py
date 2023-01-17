import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation,Bidirectional
from keras.callbacks import ModelCheckpoint

with open("/kaggle/input/sonnets/sonnets.txt") as corpus_file:
    corpus = corpus_file.read()
print("Loaded a corpus of {0} characters".format(len(corpus)))

# Get a unique identifier for each char in the corpus, then make some dicts to ease encoding and decoding
chars = sorted(list(set(corpus)))
num_chars = len(chars)
encoding = {c: i for i, c in enumerate(chars)}
decoding = {i: c for i, c in enumerate(chars)}
print("Our corpus contains {0} unique characters.".format(num_chars))

sentence_length = 50
skip = 1
X_data = []
y_data = []
for i in range (0, len(corpus) - sentence_length, skip):
    sentence = corpus[i:i + sentence_length]
    next_char = corpus[i + sentence_length]
    X_data.append([encoding[char] for char in sentence])
    y_data.append(encoding[next_char])

num_sentences = len(X_data)
print("Sliced our corpus into {0} sentences of length {1}".format(num_sentences, sentence_length))


print("Vectorizing X and y...")
X = np.zeros((num_sentences, sentence_length, num_chars), dtype=np.bool)
y = np.zeros((num_sentences, num_chars), dtype=np.bool)
for i, sentence in enumerate(X_data):
    for t, encoded_char in enumerate(sentence):
        X[i, t, encoded_char] = 1
    y[i, y_data[i]] = 1


print("Sanity check y. Dimension: {0} # Sentences: {1} Characters in corpus: {2}".format(y.shape, num_sentences, len(chars)))
print("Sanity check X. Dimension: {0} Sentence length: {1}".format(X.shape, sentence_length))


from keras.layers import Dropout
model = Sequential()
model.add(LSTM(256,input_shape=(sentence_length, num_chars)))
model.add(Dense(num_chars))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
architecture = model.to_yaml()
with open('model.yaml', 'a') as model_file:
    model_file.write(architecture)


file_path="weights-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks = [checkpoint]

model.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks)
num_chars = len(chars)
sentence_length = 50
corpus_length = len(corpus)
X_p= np.zeros((1, sentence_length,num_chars), dtype=np.bool)
seed = np.random.randint(0, corpus_length - sentence_length)
seed_pattern = corpus[seed:seed + sentence_length]
for i, character in enumerate(seed_pattern):
            X_p[0, i, encoding[character]] = 1
generated_text = ""
for i in range(500):
            prediction = np.argmax(model.predict(X_p, verbose=0))

            generated_text += decoding[prediction]

            activations = np.zeros((1, 1, num_chars), dtype=np.bool)
            activations[0, 0, prediction] = 1
            X_p = np.concatenate((X_p[:, 1:, :], activations), axis=1)
print(generated_text)
from keras.layers import Dropout
model_bi = Sequential()
model_bi.add(Bidirectional(LSTM(256,return_sequences=True), input_shape=(sentence_length, num_chars)))
model_bi.add(Dropout(0.2))
model_bi.add(Bidirectional(LSTM(256), input_shape=(sentence_length, num_chars)))
model_bi.add(Dropout(0.2))
model_bi.add(Dense(num_chars))
model_bi.add(Activation('softmax'))
model_bi.compile(loss='categorical_crossentropy', optimizer='adam')
model_bi.summary()
architecture = model_bi.to_yaml()
with open('model_bi.yaml', 'a') as model_file:
    model_file.write(architecture) 
                 
file_path="weights-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks = [checkpoint]
                 
model_bi.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks)

num_chars = len(chars)
sentence_length = 50
corpus_length = len(corpus)
X = np.zeros((1, sentence_length,num_chars), dtype=np.bool)
seed = np.random.randint(0, corpus_length - sentence_length)
seed_pattern = corpus[seed:seed + sentence_length]
for i, character in enumerate(seed_pattern):
            X[0, i, encoding[character]] = 1
generated_text = ""
for i in range(1000):
            prediction = np.argmax(model_bi.predict(X, verbose=0))

            generated_text += decoding[prediction]

            activations = np.zeros((1, 1, num_chars), dtype=np.bool)
            activations[0, 0, prediction] = 1
            X = np.concatenate((X[:, 1:, :], activations), axis=1)
print(generated_text)
