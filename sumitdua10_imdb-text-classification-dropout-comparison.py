import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/word2vec-nlp-tutorial"))
#Read the IMDB dataset with 25K reviews for training. 

df = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv", sep = '\t', 
                 error_bad_lines=False )
print("Total no. of reviews are ", df.shape[0])
print("cols are ", df.columns)
print("Sample reviews are ")
print(df.loc[:5,['review','sentiment']])

word2vec = {}
with open('../input/glove6b50dtxt/glove.6B.50d.txt', encoding="utf8") as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
MAX_VCOCAB_SIZE = 8000
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 1500

tokenizer = Tokenizer( filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
tokenizer.fit_on_texts(df['review'])
#print("Total Sequences: ", type(sequences))
word_index = tokenizer.word_index
documents = tokenizer.texts_to_sequences(df['review'])
print(list(word_index.items())[:5])#iloc[:10])
token_count = len(word_index)+1
print('Found {} unique tokens.'.format(token_count))

#print(t.word_counts)
print("Total documents ", tokenizer.document_count)
#print(t.word_index)
#print(t.word_docs)
print("max sequence length:", max(len(s) for s in documents))
print("min sequence length:", min(len(s) for s in documents))

# pad sequences so that we get a N x T matrix
data = pad_sequences(documents, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print('Shape of data tensor:', data.shape)
print(data[1])

print('Filling pre-trained embeddings...')
embedding_matrix = np.zeros((token_count, EMBEDDING_DIM))
for word, i in word_index.items():
  #if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word) #get(word) is used instead of [word] as it won't give exception in case word is not found
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i,:] = embedding_vector

print("Sample embedded dimension {}".format(embedding_matrix.shape))
print(embedding_matrix[10][:5])

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, GlobalAveragePooling1D 
from keras.layers import Embedding, Conv2D, GlobalMaxPooling1D 
from keras import regularizers

embedding_layer = Embedding(
  token_count,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False)
model = Sequential()
model.add(embedding_layer)#, input_shape= (token_count, EMBEDDING_DIM))
model.add(Conv1D(filters = 100, kernel_size = 3, padding = 'same', activation='relu'))
                 #input_shape=(token_count,EMBEDDING_DIM)))
model.add(Dropout(0.2))
model.add(MaxPooling1D())#kernel_size=500))
model.add(Conv1D(filters = 200, kernel_size = 4, padding = 'same',  activation='relu'))              
model.add(Dropout(0.5))
model.add(MaxPooling1D())
model.add(Conv1D(filters = 300, kernel_size = 5, padding = 'same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D())
model.add(Dense(192, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
#model.add(Conv1D(128, 3, activation='relu'))
model.add(Dropout(0.2))
model.add(GlobalMaxPooling1D())

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, df['sentiment'], 
                                                    test_size=0.2, random_state=42)
print(x_train.shape)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=4, mode = 'max')
model.fit(x_train, y_train , batch_size=96, epochs=50, validation_split = 0.25, 
          callbacks=[early_stopping])
#score = model.evaluate(x_test, y_test, batch_size=32)
print("Standalone CNN Result with dropout")
print("Loss & accuracy on test set is", model.evaluate(x_test, y_test))

