#Import all required Lobraries



from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras import layers

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate



import pandas as pd

import numpy as np

import re
csv_train=pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')

csv_train.shape

csv_test=pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')

csv_test.shape

#csv.head()

#check if there is any null column

#csv_test.shape

csv_train['question_input']=csv_train['question_title']+'. '+csv_train['question_body']

#Defining a function to preprocess the inout data using regular expression.

def preprocess_text(sen):

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)



    return sentence



#define a empty list and run a for loop to process each and every record in the train data set. and move the processed

#data to the list

QI_train = []

sentences = list(csv_train['question_input'])

for sen in sentences:

    QI_train.append(preprocess_text(sen))



#Defining the target variables related to question

QO_train=csv_train[['question_asker_intent_understanding','question_body_critical', 'question_conversational','question_expect_short_answer','question_fact_seeking',

       'question_has_commonly_accepted_answer','question_interestingness_others','question_interestingness_self','question_multi_intent','question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written']].values



#if you are locally running and like to see how does the preprocessed the data comment out the below line

#QO_train[0]



#Lets preprocess the data from the test dataset as well for prediction to run once the model is built

csv_test['question_input']=csv_test['question_title']+'. '+csv_test['question_body']

def preprocess_text(sen):

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)



    return sentence

#same as above. define a empty list and run a for loop to process each and every record in the test data set...

#..and move the processed data to the list

QI_test = []

sentences = list(csv_test['question_input'])

for sen in sentences:

    QI_test.append(preprocess_text(sen))

QI_test[0]



#Lets consider the 40000 unique words. Use the text_to_sequence method to convert the words to vector

tokenizer = Tokenizer(num_words=40000)

tokenizer.fit_on_texts(QI_train)



QI_train = tokenizer.texts_to_sequences(QI_train)

QI_test = tokenizer.texts_to_sequences(QI_test)



vocab_size = len(tokenizer.word_index) + 1



maxlen = 6079



#Padding - Adding the zeros post to the vector to equalize the length of all input rows to 2000

QI_train = pad_sequences(QI_train, padding='post', maxlen=maxlen)

QI_test = pad_sequences(QI_test, padding='post', maxlen=maxlen)



from numpy import array

from numpy import asarray

from numpy import zeros



embeddings_dictionary = dict()



glove_file = open('/kaggle/input/glove6b100d/glove.6B.100d.txt', encoding="utf8")



for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = asarray(records[1:], dtype='float32')

    embeddings_dictionary[word] = vector_dimensions

glove_file.close()

# Feature extraction

embedding_matrix = zeros((vocab_size, 100))

for word, index in tokenizer.word_index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector



#Using one embedding layer, one LSTM layer and 21 neurons for output

deep_inputs = Input(shape=(maxlen,))

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)

model = Sequential()

model.add(layers.Dense(32, activation='relu', input_shape=(maxlen,)))

model.add(layers.Dense(21, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])



print(model.summary())

#Lets go for prediction using the test data set

history = model.fit(QI_train, QO_train, batch_size=128, epochs=2, verbose=1,validation_split=0.1)

QO_test=model.predict(QI_test, batch_size=128, verbose=0)
# I am just repeating the same set of code here to train the model on the answer part

def preprocess_text(sen):

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)



    return sentence



AI_train = []

sentences = list(csv_train['answer'])

for sen in sentences:

    AI_train.append(preprocess_text(sen))

AI_train[0]



AO_train=csv_train[['answer_helpful','answer_level_of_information', 'answer_plausible', 'answer_relevance','answer_satisfaction', 'answer_type_instructions','answer_type_procedure', 'answer_type_reason_explanation','answer_well_written']].values

AO_train[0]



def preprocess_text(sen):

    # Remove punctuations and numbers

    sentence = re.sub('[^a-zA-Z]', ' ', sen)



    # Single character removal

    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)



    # Removing multiple spaces

    sentence = re.sub(r'\s+', ' ', sentence)



    return sentence



AI_test = []

sentences = list(csv_test['answer'])

for sen in sentences:

    AI_test.append(preprocess_text(sen))

AI_test[0]



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(AI_train)



AI_train = tokenizer.texts_to_sequences(AI_train)

AI_test = tokenizer.texts_to_sequences(AI_test)



vocab_size = len(tokenizer.word_index) + 1



maxlen = 6079



AI_train = pad_sequences(AI_train, padding='post', maxlen=maxlen)

AI_test = pad_sequences(AI_test, padding='post', maxlen=maxlen)



from numpy import array

from numpy import asarray

from numpy import zeros



embeddings_dictionary = dict()

glove_file = open('/kaggle/input/glove6b100d/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:

    records = line.split()

    word = records[0]

    vector_dimensions = asarray(records[1:], dtype='float32')

    embeddings_dictionary[word] = vector_dimensions

glove_file.close()



embedding_matrix = zeros((vocab_size, 100))

for word, index in tokenizer.word_index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector



deep_inputs = Input(shape=(maxlen,))

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)

model = Sequential()

model.add(layers.Dense(32, activation='relu', input_shape=(maxlen,)))

model.add(layers.Dense(9, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])



print(model.summary())



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())



history = model.fit(AI_train, AO_train, batch_size=128, epochs=2, verbose=1,validation_split=0.1)

AO_test=model.predict(AI_test, batch_size=128, verbose=0)





outanscol=['answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']

outquescol=['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written']

dfoutans=pd.DataFrame(AO_test,columns=outanscol)

dfoutque=pd.DataFrame(QO_test,columns=outquescol)

horizontal_stack = pd.concat([dfoutque,dfoutans], axis=1)

horizontal_stack['qa_id']=csv_test['qa_id']

cols = horizontal_stack.columns.tolist()

cols = cols[-1:] + cols[:-1]

submission=horizontal_stack[cols]

submission.to_csv('submission.csv', index=False)

submission.head()