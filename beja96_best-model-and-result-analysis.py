#General packages

import os

import numpy as np #for linear algebra

import pandas as pd # for data processing

from tqdm import tqdm #for progress information

tqdm.pandas()
#Load "train" dataset

data = pd.read_csv("../input/train.csv")
#Load "glove.840B.300d" data. Glove is a data files which assigns to each possible word a vector to "explain" it in machine language

embeddings_index = {}    #creates empty list

glove = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt') #opens the test document for reading

for line in tqdm(glove): #for every line in this text do the following        (tqdm: and show the progress)

    values = line.split(" ")  #splits the string every time there is a space into seperate strings

    word = values[0] #the first string in this text file is always the word

    coefs = np.asarray(values[1:], dtype='float32') # the following strings are the "explanation"

    embeddings_index[word] = coefs #the list is now filled with entries consisting of the word and the respective "explanations" (word vectors)

glove.close() #closes the file such that is not possible to read it anymore



print('The dictionary contains %s word vectors.' % len(embeddings_index))
from sklearn.model_selection import train_test_split

#this time some portion of the data is used to test the model in the end

train, test = train_test_split(data, 

                 test_size = 0.1, shuffle = True)

train = train.reset_index(drop=True) 

test = test.reset_index(drop = True)

print ("The training dataset has the shape:" , train.shape)

print ("The test dataset has the shape:", test.shape)
train2, validate = train_test_split(train, 

                 test_size = 0.05, shuffle = True)

train = train2.reset_index(drop=True) 

validate = validate.reset_index(drop = True)

print ("The training dataset has the shape:" , train.shape)

print ("The validation dataset has the shape:", validate.shape)
X_train = train.iloc[:,1] #Takes all rows of the first column as new dataset

Y_train = np.array(train.iloc[:, 2]) #Takes all rows of the second column as new dataset



X_validate = validate.iloc[:,1]

Y_validate = np.array(validate.iloc[:, 2])



X_test = test.iloc[:,1]

Y_test = np.array(test.iloc[:, 2])



print(X_train.shape)

print(Y_train.shape)
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
tokenizer_2 = Tokenizer(filters = puncts, lower = False)

tokenizer_2.fit_on_texts(list(data["question_text"]))

word_index_2 = tokenizer_2.word_index

print('Found %s unique tokens.' % len(word_index_2))



sequences_2 = tokenizer_2.texts_to_sequences(data["question_text"])

maxlen_2 = len(max(sequences_2, key = len)) #max number of words in a question (längste sequenz aus tokenisierten wörtern)



X_train_seq_2 = tokenizer_2.texts_to_sequences(X_train)

X_train_seq_2 = pad_sequences(X_train_seq_2, maxlen=maxlen_2)



X_validate_seq_2 = tokenizer_2.texts_to_sequences(X_validate)

X_validate_seq_2 = pad_sequences(X_validate_seq_2, maxlen=maxlen_2)



X_test_seq_2 = tokenizer_2.texts_to_sequences(X_test)

X_test_seq_2 = pad_sequences(X_test_seq_2, maxlen=maxlen_2)
#Compute Embedding Matrix for the tweaked data

embed_dim = 300 #da glove.840B.300d.txt bedeutet, dass 300d. vektor

embedding_matrix_2 = np.zeros((len(word_index_2) + 1, embed_dim)) #creation of the numpy array

for word, i in tqdm(word_index_2.items()): #loop going through each word in word_index

    embedding_vector = embeddings_index.get(word) #for each word the programm takes the respective vector and calls it embedding_vector

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix_2[i] = embedding_vector # vector gets inserted into the numpy array at the place where the word would stand according to the index.
#Load into Keras Embedding layer

from keras.layers.embeddings import Embedding

embedding_layer_2 = Embedding(len(word_index_2) + 1,

                            embed_dim,

                            weights=[embedding_matrix_2],

                            input_length=maxlen_2,

                            trainable=False)
#Metric: F1 score: F1: wikipedia, umsetzung https://github.com/keras-team/keras/blob/53e541f7bf55de036f4f5641bd2947b96dd8c4c3/keras/metrics.py

import keras.backend as K #to use math functions like "keras.backend.sum"

def fmeasure (y_true, y_pred):

    

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    

    f1 = 5 * (precision*recall) / (4*precision+recall+K.epsilon())

    

    return f1
from keras.models import Model #to build the Model

from keras.layers import Dense, Input, Dropout, Activation, Bidirectional, CuDNNGRU



# Third Model: Preprocessed Data, Improved Structure

sequence_input_3 = Input(shape=(maxlen_2,), dtype='int32')

embedded_sequences_3 = embedding_layer_2(sequence_input_3)

X_3 = Bidirectional(CuDNNGRU(128, return_sequences=True))(embedded_sequences_3)

X_3 = Dropout(0.5)(X_3)

X_3 = Bidirectional(CuDNNGRU(128, return_sequences=False))(X_3)

X_3 = Dropout(0.5)(X_3)

X_3 = Dense(1)(X_3)

X_3 = Activation('sigmoid')(X_3)
model_3 = Model(inputs = sequence_input_3, outputs=X_3)

model_3.summary()
# Callbacks

from keras.callbacks import EarlyStopping

#Early Stopping -> Stop if the results do not improve anymore

early_stopping = EarlyStopping(monitor='val_fmeasure', min_delta=0.0001, patience=1, mode='max')
model_3.compile(loss= "binary_crossentropy", optimizer='adam', metrics= [fmeasure])

history_3 = model_3.fit(X_train_seq_2, Y_train, validation_data=(X_validate_seq_2, Y_validate),

          epochs= 8, batch_size = 512, shuffle = True, callbacks = [early_stopping])
# Da nicht mehr im Keras framework, umgeschrieben für Numpy Arrays

def f_measure (y_true, y_pred):

    

    true_positives = np.sum(np.round(np.dot(y_true, y_pred)))

    predicted_positives = np.sum(y_pred)

    actual_positives = np.sum(y_true)

    

    precision = true_positives / (predicted_positives + 2e-7)

    recall = true_positives / (actual_positives + 2e-7)

    f1 = 5 * (precision*recall) / (4*precision+recall+ 2e-7)

    

    return f1
Y_validate.shape
prediction = model_3.predict([X_validate_seq_2], batch_size=1024, verbose=1)



j = 0

thresh_results = np.zeros(51)

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    value = np.zeros(11)

    for i in range(11):

        value[i] = f_measure(Y_validate[i*5000:(i+1)*5000], (prediction[i*5000:(i+1)*5000]>thresh).astype(int))

    print("F1 score at threshold {0} is {1}".format(thresh, np.mean(value)))

    thresh_results[j] = np.mean(value)

    j = j+1

    

best_thresh = 0.10 + np.argmax(thresh_results)*0.01

print("Best threshhold :", best_thresh)
prediction_final = model_3.predict([X_test_seq_2], batch_size=1024, verbose=1)

prediction_final = (prediction_final>best_thresh).astype(int).flatten()
prediction_final.shape
value = np.zeros(26)

for i in range(26):

     value[i] = f_measure(Y_test[i*5000:(i+1)*5000], (prediction_final[i*5000:(i+1)*5000]))

np.mean(value)
m=pd.crosstab(prediction_final,Y_test, rownames = ["prediction"], colnames = ["actual value"])

print("Confusion matrix")

print()

print(m)
#building a dataframe in which for each test example there is question, actual class and predicted class

results = test

results["pred"] = prediction_final

true_pos = results.loc[(results['pred'] == 1) & (results['target'] == 1)]

true_neg = results.loc[(results['pred'] == 0) & (results['target'] == 0)]

false_pos = results.loc[(results['pred'] == 1) & (results['target'] == 0)]

false_neg = results.loc[(results['pred'] == 0) & (results['target'] == 1)]

pd.options.display.max_colwidth = 300
true_pos.head(10)
true_neg.head(10)
false_neg.head(10)
false_pos.head(10)