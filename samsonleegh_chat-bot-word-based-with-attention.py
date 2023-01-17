from __future__ import print_function

import sys
import os
import pandas as pd
import numpy as np
import re
import nltk

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Bidirectional
from keras.models import Model, load_model

INPUT_LENGTH = 20
OUTPUT_LENGTH = 20

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the data
lines = open('../input/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('../input/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
# Create a dictionary to map each line's id with its text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
# Create a list of all of the conversations' lines' ids.
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))
#id and conversation sample
for k in convs[300]:
    print (k, id2line[k])
# Sort the sentences into questions (inputs) and answers (targets)
questions = []
answers = []
for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])
        
# Compare lengths of questions and answers
print(len(questions))
print(len(answers))
def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|]", "", text)
#     text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = " ".join(text.split())
    return text
# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
clean_answers = []    
for answer in answers:
    clean_answers.append(clean_text(answer))
# Find the length of sentences (not using nltk due to processing speed)
lengths = []
# lengths.append([len(nltk.word_tokenize(sent)) for sent in clean_questions]) #nltk approach
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))
# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])
print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
# Remove questions and answers that are shorter than 1 word and longer than 20 words.
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

for i, question in enumerate(clean_questions):
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

for i, answer in enumerate(short_answers_temp):
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
        
print(len(short_questions))
print(len(short_answers))
r = np.random.randint(1,len(short_questions))

for i in range(r, r+3):
    print(short_questions[i])
    print(short_answers[i])
    print()
#choosing number of samples
num_samples = 30000  # Number of samples to train on.
short_questions = short_questions[:num_samples]
short_answers = short_answers[:num_samples]
#tokenizing the qns and answers
short_questions_tok = [nltk.word_tokenize(sent) for sent in short_questions]
short_answers_tok = [nltk.word_tokenize(sent) for sent in short_answers]
#train-validation split
data_size = len(short_questions_tok)

# We will use the first 0-80th %-tile (80%) of data for the training
training_input  = short_questions_tok[:round(data_size*(80/100))]
training_input  = [tr_input[::-1] for tr_input in training_input] #reverseing input seq for better performance
training_output = short_answers_tok[:round(data_size*(80/100))]

# We will use the remaining for validation
validation_input = short_questions_tok[round(data_size*(80/100)):]
validation_input  = [val_input[::-1] for val_input in validation_input] #reverseing input seq for better performance
validation_output = short_answers_tok[round(data_size*(80/100)):]

print('training size', len(training_input))
print('validation size', len(validation_input))
# Create a dictionary for the frequency of the vocabulary
# Create 
vocab = {}
for question in short_questions_tok:
    for word in question:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

for answer in short_answers_tok:
    for word in answer:
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1            
# Remove rare words from the vocabulary.
# We will aim to replace fewer than 5% of words with <UNK>
# You will see this ratio soon.
threshold = 15
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1
print("Size of total vocab:", len(vocab))
print("Size of vocab we will use:", count)
#we will create dictionaries to provide a unique integer for each word.
WORD_CODE_START = 1
WORD_CODE_PADDING = 0


word_num  = 2 #number 1 is left for WORD_CODE_START for model decoder later
encoding = {}
decoding = {1: 'START'}
for word, count in vocab.items():
    if count >= threshold: #get vocabularies that appear above threshold count
        encoding[word] = word_num 
        decoding[word_num ] = word
        word_num += 1

print("No. of vocab used:", word_num)
#include unknown token for words not in dictionary
decoding[len(encoding)+2] = '<UNK>'
encoding['<UNK>'] = len(encoding)+2
dict_size = word_num+1
dict_size
def transform(encoding, data, vector_size=20):
    """
    :param encoding: encoding dict built by build_word_encoding()
    :param data: list of strings
    :param vector_size: size of each encoded vector
    """
    transformed_data = np.zeros(shape=(len(data), vector_size))
    for i in range(len(data)):
        for j in range(min(len(data[i]), vector_size)):
            try:
                transformed_data[i][j] = encoding[data[i][j]]
            except:
                transformed_data[i][j] = encoding['<UNK>']
    return transformed_data
#encoding training set
encoded_training_input = transform(
    encoding, training_input, vector_size=INPUT_LENGTH)
encoded_training_output = transform(
    encoding, training_output, vector_size=OUTPUT_LENGTH)

print('encoded_training_input', encoded_training_input.shape)
print('encoded_training_output', encoded_training_output.shape)
#encoding validation set
encoded_validation_input = transform(
    encoding, validation_input, vector_size=INPUT_LENGTH)
encoded_validation_output = transform(
    encoding, validation_output, vector_size=OUTPUT_LENGTH)

print('encoded_validation_input', encoded_validation_input.shape)
print('encoded_validation_output', encoded_validation_output.shape)
import tensorflow as tf
tf.keras.backend.clear_session()
INPUT_LENGTH = 20
OUTPUT_LENGTH = 20

encoder_input = Input(shape=(INPUT_LENGTH,))
decoder_input = Input(shape=(OUTPUT_LENGTH,))
from keras.layers import SimpleRNN

encoder = Embedding(dict_size, 128, input_length=INPUT_LENGTH, mask_zero=True)(encoder_input)
encoder = LSTM(512, return_sequences=True, unroll=True)(encoder)
encoder_last = encoder[:,-1,:]

print('encoder', encoder)
print('encoder_last', encoder_last)

decoder = Embedding(dict_size, 128, input_length=OUTPUT_LENGTH, mask_zero=True)(decoder_input)
decoder = LSTM(512, return_sequences=True, unroll=True)(decoder, initial_state=[encoder_last, encoder_last])

print('decoder', decoder)

# For the plain Sequence-to-Sequence, we produced the output from directly from decoder
# output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(decoder)
from keras.layers import Activation, dot, concatenate

# Equation (7) with 'dot' score from Section 3.1 in the paper.
# Note that we reuse Softmax-activation layer instead of writing tensor calculation
attention = dot([decoder, encoder], axes=[2, 2])
attention = Activation('softmax', name='attention')(attention)
print('attention', attention)

context = dot([attention, encoder], axes=[2,1])
print('context', context)

decoder_combined_context = concatenate([context, decoder])
print('decoder_combined_context', decoder_combined_context)

# Has another weight + tanh layer as described in equation (5) of the paper
output = TimeDistributed(Dense(512, activation="tanh"))(decoder_combined_context)
output = TimeDistributed(Dense(dict_size, activation="softmax"))(output)
print('output', output)
model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
training_encoder_input = encoded_training_input
training_decoder_input = np.zeros_like(encoded_training_output)
training_decoder_input[:, 1:] = encoded_training_output[:,:-1]
training_decoder_input[:, 0] = WORD_CODE_START
training_decoder_output = np.eye(dict_size)[encoded_training_output.astype('int')]

validation_encoder_input = encoded_validation_input
validation_decoder_input = np.zeros_like(encoded_validation_output)
validation_decoder_input[:, 1:] = encoded_validation_output[:,:-1]
validation_decoder_input[:, 0] = WORD_CODE_START
validation_decoder_output = np.eye(dict_size)[encoded_validation_output.astype('int')]
model.fit(x=[training_encoder_input, training_decoder_input], y=[training_decoder_output],
          validation_data=([validation_encoder_input, validation_decoder_input], [validation_decoder_output]),
          #validation_split=0.05,
          batch_size=64, epochs=100)

model.save('model_attention.h5')
def prediction(raw_input):
    clean_input = clean_text(raw_input)
    input_tok = [nltk.word_tokenize(clean_input)]
    input_tok = [input_tok[0][::-1]]  #reverseing input seq
    encoder_input = transform(encoding, input_tok, 20)
    decoder_input = np.zeros(shape=(len(encoder_input), OUTPUT_LENGTH))
    decoder_input[:,0] = WORD_CODE_START
    for i in range(1, OUTPUT_LENGTH):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return output

def decode(decoding, vector):
    """
    :param decoding: decoding dict built by word encoding
    :param vector: an encoded vector
    """
    text = ''
    for i in vector:
        if i == 0:
            break
        text += ' '
        text += decoding[i]
    return text
for i in range(20):
    seq_index = np.random.randint(1, len(short_questions))
    output = prediction(short_questions[seq_index])
    print ('Q:', short_questions[seq_index])
    print ('A:', decode(decoding, output[0]))
raw_input = input()
output = prediction(raw_input)
print (decode(decoding, output[0]))
