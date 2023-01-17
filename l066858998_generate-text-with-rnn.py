import pandas as pd
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
import matplotlib.pyplot as plt
dataset_path = "../input/nyt-comments/CommentsApril2017.csv"
dataset = pd.read_csv(dataset_path)
dataset.shape
# to prevent run out of memory, I only select part of dataset
dataset = dataset[:1000]
dataset.head(10)
sentences = dataset["commentBody"].values
sentences[0]
sentences[1]
# convert all words to lowercase
for idx, sentence in enumerate(sentences):
    sentences[idx] = sentence.lower()
# fit all sentences on tokenizer
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
# word index of tokenizer
tokenizer.word_index
# number of total words
total_word = len(tokenizer.word_index)+1
print("Total number of word: ", total_word)
# convert sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences)
# prepare training sequences
training_sequences = []

for seq in sequences:
    for i in range(2, len(seq)):
        training_sequences.append(seq[:i])
        
training_sequences = np.array(training_sequences)
print("Length of training_sequences: ", len(training_sequences))
# take a look on training_sequences
print("The first sequence in training sequences: ", training_sequences[0])
print("The second sequence in training sequences: ", training_sequences[1])
# pad all sequences to make them same length
longest_len = max([len(l) for l in training_sequences])
training_sequences = keras.preprocessing.sequence.pad_sequences(sequences=training_sequences,
                                           maxlen=longest_len,
                                           padding="pre")
# prepare x_train and y_train
x_train = training_sequences[:, :-1]
y_train = training_sequences[:, -1]
y_train = keras.utils.to_categorical(y=y_train, num_classes=total_word)
print("Shape of training_sequences: ", training_sequences.shape)
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
# model architechture
model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=total_word,
                                 output_dim=64,
                                 input_length=longest_len))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))
model.add(keras.layers.Dense(units=64, activation="relu"))
model.add(keras.layers.Dense(units=total_word, activation="softmax"))
model.summary()
# load model weight: model was trained to get the accuracy of 0.95
try:
    model.load_weights("../input/model-weight-generate-text-with-rnn/best_model_weight.h5")
except:
    print("ERROR")
# compile model
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["acc"])
# define custom callback for training
class CustomCallback(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs):
        if(logs["acc"] >= 0.95):
            self.model.stop_training = True

custome_callback = CustomCallback()
checkpoint = keras.callbacks.ModelCheckpoint(filepath="best_model.h5",
                                             monitor="acc",
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode="auto",
                                             save_freq="epoch")
# train model
history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=32,
                    epochs=500,
                    callbacks=[custome_callback, checkpoint])
first_word = "You"

generated_sentence = [first_word]
num_word_to_generate = 25
generated_sentence
tokenizer.texts_to_sequences(generated_sentence)
# create a dict to map idx to word
idx2word = {idx:word for word, idx in tokenizer.word_index.items()}
idx2word
for i in range(num_word_to_generate):
    
    x = tokenizer.texts_to_sequences(generated_sentence)
    
    if len(x[0]) > longest_len:
        x[0] = x[0][-1 * longest_len:]
    else:
        x = keras.preprocessing.sequence.pad_sequences(sequences=x,
                                                   maxlen=longest_len,
                                                   padding="pre")
    x = np.array(x)
    y = model.predict(x)[0]
    idx = np.argmax(y)
    
    generated_word = idx2word[idx]
    
    generated_sentence[0] += " " + generated_word
generated_sentence