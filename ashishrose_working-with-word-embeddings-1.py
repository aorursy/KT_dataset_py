import tensorflow as tf
print(tf.__version__)
#Download the dataset
#if tensorflow_datasets is not installed use this script: !pip install -q tensorflow-datasets
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
import numpy as np
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

#iterating over the data(train_data/test_data in plain text format) --> convert to numpy --> and then decode using utf-8  -->append to the list
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())
  
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000 #10000 unique words
embedding_dim = 16 # Total dimenions for classification
max_length = 120 #max length of the sequence
trunc_type='post' #truncate at the end of the line
oov_tok = "<OOV>" #Out of vocabulary 


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token=oov_tok)#creating a tokenizer
tokenizer.fit_on_texts(training_sentences)#tokenizer will fit on the training_texts(here the output will be number as you might have guessed !!)

sequences = tokenizer.texts_to_sequences(training_sentences)#this is basically numerical sequences of the original texts
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)#padding and the truncating is done so as to make the sentences symmetrical

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

word_index = tokenizer.word_index
print(word_index)#word index is basically a dictionary of word and its unique token.

#Doubt: vocab size and the len(word_index) should be same right ????
#The answer is vocab_size is not the necesary parameter for creating Tokenizer "Tokenizer(oov_token=oov_tok)"
len(word_index)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index), embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
