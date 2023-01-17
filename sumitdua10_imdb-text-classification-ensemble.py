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
from keras.layers import Input, Dense, Concatenate
from keras.models import Model


embedding_layer = Embedding(
  token_count,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False)
def conv_model(kernel, pool):
    
    inputs1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x1 = embedding_layer(inputs1)
    #print("After applying embeeding ", x1)
    num_filters=64
    if(kernel==3):
        num_filters=128
    x1 = Conv1D(num_filters, kernel_size = kernel, padding = 'same', activation='relu'
               ,kernel_regularizer=regularizers.l2(0.01))(x1)
                     #input_shape=(token_count,EMBEDDING_DIM)))
    #print("After applying conv1d on filter size 3 ", x1)
    x1 = MaxPooling1D(pool_size=pool)(x1)
    #print("After applying global max pooling ", x1)

    x1 = Conv1D(filters = 256, kernel_size = kernel, padding = 'same', activation='relu'
               ,kernel_regularizer=regularizers.l2(0.01))(x1)
    #print(x1)
    x1 = MaxPooling1D(pool_size=pool)(x1)

    x1 = Conv1D(filters = 128, kernel_size = kernel, padding = 'same', activation='relu'
               ,kernel_regularizer=regularizers.l2(0.01))(x1)
    x1 = GlobalMaxPooling1D()(x1)

    x1 = Dense(64, activation='relu')(x1)

    x1 = Dense(1, activation='sigmoid')(x1)

    model1 = Model(inputs=inputs1, outputs=x1)

    model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model1
    


model1 = conv_model(kernel=3, pool=2)
print(model1.summary())
model2 = conv_model(kernel=4, pool=3)
print(model2.summary())
model3 = conv_model(kernel=5, pool=4)
print(model3.summary())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, df['sentiment'], 
                                                    test_size=0.2, random_state=42)
print(x_train.shape)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='acc', patience=4, mode = 'max')
model1.fit(x_train, y_train , batch_size=96, epochs=50, validation_split = 0.25, 
           callbacks=[early_stopping])
#score = model.evaluate(x_test, y_test, batch_size=32)
model2.fit(x_train, y_train , batch_size=96, epochs=50, validation_split = 0.25,
           callbacks=[early_stopping])
model3.fit(x_train, y_train , batch_size=96, epochs=50, validation_split = 0.25, 
           callbacks=[early_stopping])
from sklearn.metrics import accuracy_score
#print("Concatenated CNN Result")
#print("Loss & accuracty on test set is", Fmodel.evaluate(x_test, y_test))

print("CNN Result")
print("Loss & accuracty on test set is", model1.evaluate(x_test, y_test))
print("Loss & accuracty on test set is", model2.evaluate(x_test, y_test))
print("Loss & accuracty on test set is", model3.evaluate(x_test, y_test))

y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)
y_pred = (y_pred1+y_pred2+y_pred3)/3
print("Accuracy score on y_test for Ensembel model of 3 is :")
print(accuracy_score(y_test,np.round(y_pred)))

