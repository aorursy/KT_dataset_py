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
import re                       # to clean text and replace charactors to simplfy the conversation
import tensorflow as tf 
# read 'movie_lines.txt' file  
lines = open('/kaggle/input/cornell-moviedialog-corpus/movie_lines.txt', encoding = 'UTF-8', errors = 'ignore').read().split('\n')
# read 'conversations.txt' file  
conversations = open('/kaggle/input/cornell-moviedialog-corpus/movie_conversations.txt', encoding = 'UTF-8', errors = 'ignore').read().split('\n')
# display first lines of file 'movie_lines.txt'
with open('/kaggle/input/cornell-moviedialog-corpus/movie_lines.txt') as m_lines:
        head = [next(m_lines) for x in range(1)]
        print(head)
# display first lines of file 'movie_conversations.txt'
with open('/kaggle/input/cornell-moviedialog-corpus/movie_conversations.txt') as m_conv:
        head = [next(m_conv) for x in range(1)]
        print(head)
# create a dictionary that maps each lines and id
id2line = {} 
for line in lines:
    _line = line.split(' +++$+++ ')                 # split the line with sep ' +++$+++ '
    if len(_line) == 5:                             # len of line should be 5    
        id2line[_line[0]] = _line[4]                # assign id of line (index'0') with movie line (index'4')  
# print id2line value
print(dict(list(id2line.items())[0: 5]))  
# create list of all the conversations
conversations_ids = []
for conversation in conversations[:-1]:    # last row of conversations is empty
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ","")
    conversations_ids.append(_conversation.split(','))
# print first 10 conversations_ids values 
conversations_ids[0:10]
# getting seperately questions and answers 
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):           # iterate through all the ids of a conversation_ids list
        questions.append(id2line[conversation[i]])  
        answers.append(id2line[conversation[i+1]])
# dataframe showing seperate questions ans anwers obtained from above lines 
QA_df = pd.DataFrame({'Question': list(questions), 'Answers': list(answers)} )
QA_df.head(5)
# text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text) 
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-{}+=();*&^%$#@!~`><:]","", text)
    return text
# clean questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers[0:5]
# clean answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
# create a dict that maps a word with number of its occurance
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] =  1
        else:
            word2count[word] +=1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] =  1
        else:
            word2count[word] +=1
# print word2count value
print(dict(list(word2count.items())[0: 5])) 
# get frequency of questions word  and answers word 

threshold = 20
word_number = 0 
questionword2int = {}
for word,count in word2count.items():
    if count>threshold:
        questionword2int[word] = word_number
        word_number +=1

answerword2int = {}
for word,count in word2count.items():
    if count>threshold:
        answerword2int[word] = word_number
        word_number +=1
dict(list(answerword2int.items())[-5:])
# adding  last tokens to above dict
tokens = ['<PAD>','<EOS>','<OUT>', '<SOS>']
for token in tokens:
    questionword2int[token] = len(questionword2int)+1

for token in tokens:
    answerword2int[token] = len(answerword2int)+1
#creating inverse of dict answerword2int
answerint2word = {w_i:w for w, w_i in answerword2int.items() }
sorted(dict(list(answerint2word.items())[-5:]))
# adding End of String token to to end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

clean_answers[0:2]
# Translating questions and answers  into integers. Replacing word which are filtered out by <OUT> token

question_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionword2int:
            ints.append(questionword2int['<OUT>'])
        else:
            ints.append(questionword2int[word])
    question_to_int.append(ints)

answer_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerword2int:
            ints.append(answerword2int['<OUT>'])
        else:
            ints.append(answerword2int[word])
    answer_to_int.append(ints)
#question_to_int[0:2]
# sort question and answer by length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25+1):
    for i in enumerate(question_to_int):
        if len(i[1])== length:
            sorted_clean_questions.append(question_to_int[i[0]])
            sorted_clean_answers.append(answer_to_int[i[0]])
            
sorted_clean_questions[-1:]
# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
 

# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
# Creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 

# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
