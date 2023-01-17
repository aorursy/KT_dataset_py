# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing packages
import tensorflow as tf
import re
import time
#loading data
conv_lines=open("../input/movie-dialog-corpus/movie_conversations.tsv", encoding ='utf-8-sig', errors='ignore').read().split('\n')
lines=open("../input/movie-dialog-corpus/movie_lines.tsv",encoding='utf-8-sig', errors='ignore').read().split('\n')
print(conv_lines[:10])
print(lines[:10])
#preparing an id to line dictionary
id2line={}
for line in lines:
    _line=line.split('\t')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]

#collecting line ids for each conversation 
conv=[]
for line in conv_lines[:-1]:
    _line=line.split('\t')[-1][1:-1].replace("'","").replace(" ",",")
    conv.append(_line.split(','))
print(conv[:10])
#preparing questions and corresponding answers
#from each conversation
questions=[]
answers=[]
for con in conv:
    for i in range(len(con)-1):
        if con[i] not in id2line:
            id2line[con[i]]=""
        if con[i+1] not in id2line:
            id2line[con[i+1]]=""
        questions.append(id2line[con[i]])
        answers.append(id2line[con[i+1]])
print(questions[:10])
print(answers[:10])
print(len(questions))
print(len(answers))
#function to make the text simpler
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
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

#cleaning questions
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))

#cleaning answers
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))
'''
for i in range(0,5):
    print(clean_questions[i])
    print(clean_answers[i])
    print()
    '''
#defining a max length
max_length=20
min_length=1

#eliminating long questions-answers
short_questions=[]
short_answers=[]
i=0
for question in clean_questions:
    if (len(question.split())<=max_length
        and len(question.split())>=min_length
        and len(clean_answers[i].split())<=max_length
        and len(clean_answers[i].split())>=min_length):
        short_questions.append(question)
        short_answers.append(clean_answers[i])
    i+=1
print(len(short_questions))
print(len(short_answers))
print(len(short_questions)/len(clean_questions))
#free some space
del lines
del conv_lines
del conv
del clean_answers
del clean_questions
#creating vocabulary
# we create different vocabulary for questions and answers
vocab={}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word]=1
        else:
            vocab[word]+=1

for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word]=1
        else:
            vocab[word]+=1

#words that appear less than threshold are dropped from vocab
# we will replace them with <UNK> token
threshold=10

question_vocab_to_int={}
word_num=0
for word, count in vocab.items():
    if count>=threshold:
        question_vocab_to_int[word]=word_num
        word_num+=1

answer_vocab_to_int={}
word_num=0
for word, count in vocab.items():
    if count>=threshold:
        answer_vocab_to_int[word]=word_num
        word_num+=1

#free more space
del vocab
#additional required tokens
codes=['<Pad>','<EOS>','<UNK>','<GO>']
for code in codes:
    question_vocab_to_int[code]=len(question_vocab_to_int)
    answer_vocab_to_int[code]=len(answer_vocab_to_int)

question_int_to_vocab={v_i:v for v,v_i in question_vocab_to_int.items()}
answer_int_to_vocab={v_i:v for v,v_i in answer_vocab_to_int.items()}

print(len(question_vocab_to_int))
print(len(question_int_to_vocab))
print(len(answer_vocab_to_int))
print(len(answer_int_to_vocab))
#turning sentences into numerical arrays 
questions_int=[]
for question in short_questions:
    ints=[]
    for word in question.split():
        if word not in question_vocab_to_int:
            ints.append(question_vocab_to_int['<UNK>'])
        else:
            ints.append(question_vocab_to_int[word])
    questions_int.append(ints)

answers_int=[]
for answer in short_answers:
    ints=[]
    ints.append(answer_vocab_to_int['<GO>'])
    for word in answer.split():
        if word not in answer_vocab_to_int:
            ints.append(answer_vocab_to_int['<UNK>'])
        else:
            ints.append(answer_vocab_to_int[word])
    ints.append(answer_vocab_to_int['<EOS>'])
    answers_int.append(ints)

print(len(questions_int))
print(len(answers_int))
#free more space
del short_questions
del short_answers
#defining encoder and decoder input and target data
encoder_input_data=np.zeros(
    (len(questions_int)//20, max_length),
    dtype='float32')
decoder_input_data=np.zeros(
    (len(questions_int)//20, max_length+2),
    dtype='float32')
decoder_output_data=np.zeros(
    (len(questions_int)//20, max_length+2, len(answer_vocab_to_int)),
    dtype='float32')

print(encoder_input_data.shape)
print(decoder_input_data.shape)
print(decoder_output_data.shape)
for i, (input_t,target_t) in enumerate(zip(questions_int,answers_int)):
    for t, num in enumerate(input_t):
        encoder_input_data[i,t]= num
    encoder_input_data[i,t+1:]=question_vocab_to_int['<Pad>']
    for t,num in enumerate(target_t):
        decoder_input_data[i,t]=num
        if t>0:
            decoder_output_data[i,t-1,num]=1.
    decoder_input_data[i,t+1:]=answer_vocab_to_int['<Pad>']
    decoder_output_data[i,t:,answer_vocab_to_int['<Pad>']]=1.
    if i==int(len(questions_int)//20)-1:
        break

print(encoder_input_data[0,:])
print(decoder_input_data[0,:])
print(decoder_output_data[0,:,:])
#creating encoder
encoder_inputs=tf.keras.Input(shape=(None, ))
encoder_embedding=tf.keras.layers.Embedding(len(question_vocab_to_int), 200, mask_zero=True)(encoder_inputs)
encoder=(tf.keras.layers.LSTM(200, return_state=True))
encoder_outputs, state_h, state_c=encoder(encoder_embedding)
encoder_states=[state_h, state_c]

#creating decoder
decoder_inputs=tf.keras.Input(shape=(None, ))
decoder_embedding=tf.keras.layers.Embedding(len(answer_vocab_to_int), 200, mask_zero=True)(decoder_inputs)
decoder=(tf.keras.layers.LSTM(200, return_sequences=True, return_state=True))
decoder_outputs,_,_=decoder(decoder_embedding, initial_state=encoder_states)
decoder_dense=tf.keras.layers.Dense(len(answer_vocab_to_int)
                                    ,activation='softmax')
decoder_outputs=decoder_dense(decoder_outputs)

#creating training model
model=tf.keras.Model([encoder_inputs,decoder_inputs], decoder_outputs)
#fitting data on model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data,decoder_input_data],decoder_output_data,
          batch_size=64,
          epochs=25,
          validation_split=0.2)

#saving model
model.save('conv_model')
#inference model
#get states from encoder and input by encoder model
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

#get output by decoder model
decoder_state_input_h = tf.keras.Input(shape=(200,))
decoder_state_input_c = tf.keras.Input(shape=(200,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
#function to get output for input sentence
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = answer_vocab_to_int['<GO>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = answer_int_to_vocab[sampled_token_index]
        decoded_sentence += sampled_char + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<EOS>' or
           len(decoded_sentence) > max_length+2):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence
#function to get sentence from numerical array
def get_sentence(input_seq):
    sent=''
    for seq in input_seq[0]:
        sent+= question_int_to_vocab[seq] + ' '
    return sent


for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', get_sentence(input_seq))
    print('Decoded sentence:', decoded_sentence)
for _ in range(10):
    stat = input('Enter question : ')
    seq = np.zeros((1,max_length))
    i=0
    for word in stat.split():
        seq[0][i]=(question_vocab_to_int[word])
        i+=1
    while i<max_length:
        seq[0][i]=(question_vocab_to_int['<Pad>'])
        i+=1
    decoded_sentence = decode_sequence(seq)
    print( decoded_sentence)
    print(' ')
