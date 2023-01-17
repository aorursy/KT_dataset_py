import numpy as np

import pandas as pd 

import tensorflow as tf

import re

import zipfile
tf.__version__
from subprocess import check_output

print(check_output(["ls", "../input/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus"]).decode("utf8"))
lines = open("../input/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")

conversations = open("../input/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus/movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")
print("Lines : \n")

for i in range(7,11):

    print(lines[23-i])

print("\n\nConversations : \n")

print(conversations[19])
def text_preprocessor(line):

    x = line.lower()

    x = re.sub(r"won't", "will not", x)

    x = re.sub(r"i ain't", "i am not", x)

    x = re.sub(r"he ain't", "he is not", x)

    x = re.sub(r"can't", "cannot", x)

    x = re.sub(r"n't", " not", x)

    x = re.sub(r"let's", "let us", x)

    x = re.sub(r"'s ", " is ", x)

    x = re.sub(r"'re ", " are ", x)

    x = re.sub(r"'m ", " am ", x)

    x = re.sub(r"'ll ", " will ", x)

    x = re.sub(r"'d ", " would ", x)

    x = re.sub(r"'s ", " is ", x)

    x = re.sub(r"'ve ", " have ", x)

    x = re.sub(r"wanna", "want to", x)

    x = re.sub(r"gonna", "going to", x)

    x = re.sub(r"\W", " ", x)

    x = re.sub(r"\d", " ", x)

    x = re.sub(r"\s+$", "", x)

    x = re.sub(r"^\s+", "", x)

    x = re.sub(r"\s+", " ", x)

    

    return x
Id_to_line = {}

for line in lines:

    l = line.split(" +++$+++ ")

    if len(l)==5:

        Id_to_line[l[0]] = text_preprocessor(l[4])
print("No. of lines: ",len(Id_to_line))

print("Sample Mappings : \nL862: ",Id_to_line['L862'],"\nL863: ",Id_to_line['L863'],"\nL864: ",Id_to_line['L864'],"\nL865: ",Id_to_line['L865'])
conversation_lists = []

for conv in conversations[:-1]:

    c = conv.split(" +++$+++ ")

    c = c[3]

    c = re.sub(r"[\[\]\'\,]", " ", c)

    c = re.sub(r"^\s+", "",c)

    c = re.sub(r"\s+$", "",c)

    c = re.sub(r"\s+", " ",c)

    c = c.split(" ")

    conversation_lists.append(c)

    
print(conversation_lists[19])

for i in range(len(conversation_lists[19])):

    print(conversation_lists[19][i],"  :  ",Id_to_line[conversation_lists[19][i]])
messages = []

responses = []

for conv in conversation_lists:

    for i in range(len(conv)-1):

        messages.append(Id_to_line[conv[i]])

        responses.append(Id_to_line[conv[i+1]])
print("Message: ", messages[114],"\nResponse: ",responses[114])
word_frequency = {}

for message in messages:

    for word in message.split(" "):

        if word not in word_frequency.keys():

            word_frequency[word] = 1

        else:

            word_frequency[word] += 1



for response in responses:

    for word in response.split(" "):

        if word not in word_frequency.keys():

            word_frequency[word] = 1

        else:

            word_frequency[word] += 1

        
threshold = 20

unique_id = 0

word_to_num = {}

for word in word_frequency.keys():

    if word_frequency[word]>=threshold:

        word_to_num[word] = unique_id

        unique_id += 1



# Adding extra tokens to this dictionary

tokens = ["<PAD>","<EOS>","<SOS>","<UNK>"]

for token in tokens:

    word_to_num[token] = unique_id

    unique_id += 1

    print(token, "  :  ",word_to_num[token])
num_to_word = { num: word for word, num in word_to_num.items()}
print(word_to_num["what"]," : ",num_to_word[word_to_num["what"]])
for i in range(len(responses)):

    responses[i] += " <EOS>"
responses[6736]
for i in range(len(messages)):

    message = messages[i].split(" ")

    for j in range(len(message)):

        if message[j] in word_to_num.keys():

            message[j] = word_to_num[message[j]]

        else :

            message[j] = word_to_num["<UNK>"]

    messages[i] = message



for i in range(len(responses)):

    response = responses[i].split(" ")

    for j in range(len(response)):

        if response[j] in word_to_num.keys():

            response[j] = word_to_num[response[j]]

        else :

            response[j] = word_to_num["<UNK>"]

    responses[i] = response
print(messages[114],responses[114])
sorted_messages = []

sorted_responses = []

for length in range(1, 20):         # Smaller range of length would decrease the number of messages and responses

    for pair in enumerate(messages):

        if len(pair[1])==length:

            sorted_messages.append(pair[1])

            sorted_responses.append(responses[pair[0]])
print([num_to_word[sorted_messages[70065][i]] for i in range(len(sorted_messages[70065]))])
def model_inputs(batch_size):

    '''Create palceholders for inputs to the model'''

    input_data = tf.placeholder(tf.int32, [batch_size, None], name='current_input_batch')

    output_data = tf.placeholder(tf.int32, [batch_size, None], name='current_output_batch')

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return input_data, output_data, learning_rate, keep_prob
def preprocessed_targets(targets, word_to_num, batch_size):

    left_side = tf.fill([batch_size, 1], word_to_num["<SOS>"])

    new_targets = tf.concat([left_side, targets], axis=1)

    return new_targets
def encoder_rnn(embedded_message_batch, num_units_in_LSTM, num_layers, keep_prob, sequence_length):

    """Takes embedded inputs and returns an embedding sequence that becomes input for the decoder"""

    lstm = tf.contrib.rnn.BasicLSTMCell(num_units_in_LSTM)

    lstm_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)

    

    stacked_encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_with_dropout for n in range(num_layers)])

    

    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = stacked_encoder_cell,

                                                       cell_bw = stacked_encoder_cell,

                                                       inputs = embedded_message_batch,

                                                       sequence_length = sequence_length,

                                                       dtype = tf.float32)

    return encoder_output,encoder_state
def decoding_layer_train(encoder_state, encoder_output, decoder_cell, decoder_embedded_input, sequence_length, output_function, keep_prob, batch_size): # decoding_scope

    with tf.variable_scope('shared_attention_mechanism'):

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units = decoder_cell.output_size,

                                                                   memory = tf.concat(encoder_output, 2))  # memory = encoder_output

    decoder_cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(cell = decoder_cell,

                                                                      attention_mechanism = attention_mechanism,

                                                                      attention_layer_size = decoder_cell.output_size)

    

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_embedded_input,

                                                        sequence_length = sequence_length)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell_with_attention,

                                                       helper = training_helper,

                                                       initial_state = decoder_cell_with_attention.zero_state(batch_size, tf.float32).clone(cell_state = encoder_state))

    with tf.variable_scope('decode_with_shared_attention'):

        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder = training_decoder,

                                                                          impute_finished = True,

                                                                          maximum_iterations = sequence_length)

    training_decoder_output_dropout = tf.nn.dropout(training_decoder_output, keep_prob)

    

    return output_function(training_decoder_output_dropout)
def decoding_layer_predict(encoder_state, encoder_output, decoder_cell, decoder_embeddings, SOS_id, EOS_id, maximum_length, vocab_size, beam_width, output_function, keep_prob, batch_size): # decoding_scope

    

    # Beam Search Tile

    encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier = beam_width)

    encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier = beam_width)

    

    # Attention (Predicting)

    with tf.variable_scope('shared_attention_mechanism', reuse=True):

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units = decoder_cell.output_size,

                                                                   memory = tf.concat(encoder_output, 2)) # memory = encoder_output

        

        decoder_cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(cell = decoder_cell,

                                                                          attention_mechanism = attention_mechanism,

                                                                          attention_layer_size = decoder_cell.output_size)

        

    

    # Decoder (Predicting)

    predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = decoder_cell_with_attention,

                                                              embedding = decoder_embeddings,

                                                              start_tokens = tf.tile(tf.constant([SOS_id], dtype=tf.int32), [batch_size]),

                                                              end_token = EOS_id,

                                                              initial_state = decoder_cell_with_attention.zero_state(batch_size*beam_width, tf.float32).clone(cell_state = encoder_state),

                                                              beam_width = beam_width,

                                                              output_layer = output_function,

                                                              length_penalty_weight = 0.0)

    

    with tf.variable_scope('decode_with_shared_attention', reuse = True):

        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decoder(decoder = predicting_decoder,

                                                                             impute_finished = False,

                                                                             maximum_iterations = 2* maximum_length)

    

    predicted_logits = predicting_decoder_output.predicted_ids[:, :, 0]

    return predicted_logits
def decoder_rnn(decoder_embedded_input, decoder_embeddings, encoder_state, encoder_output, vocab_size, sequence_length, num_units, num_layers, word_to_num, keep_prob, beam_width, batch_size):

    lstm = tf.contrib.rnn.BasicLSTMCell(num_units)

    lstm_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)

    stacked_decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_with_dropout for n in range(num_layers)])

    

    weights = tf.truncated_normal_initializer(stddev=0.1)

    biases = tf.zeros_initializer()

    

    output_function = lambda x: tf.contrib.layers.fully_connected(inputs = x,

                                                                  num_outputs = vocab_size,

                                                                  activation_fn = None,

                                                                  weights_initializer = weights,

                                                                  biases_initializer = biases)

    

    train_logits = decoding_layer_train(encoder_state,

                                        encoder_output,

                                        stacked_decoder_cell,

                                        decoder_embedded_input,

                                        sequence_length,

                                        output_function,

                                        keep_prob,

                                        batch_size)

    

    prediction_logits = decoding_layer_predict(encoder_state,

                                               encoder_output,

                                               stacked_decoder_cell,

                                               decoder_embeddings,

                                               word_to_num["<SOS>"],

                                               word_to_num["<EOS>"],

                                               sequence_length,

                                               vocab_size,

                                               beam_width,

                                               output_function,

                                               keep_prob,

                                               batch_size)

    return train_logits, prediction_logits
def seq2seq_model(input_data, target_data, vocab_size, encoder_embedding_size, decoder_embedding_size, num_units, num_layers, word_to_num, sequence_length, keep_prob, beam_width, batch_size):

    

    embedded_message_batch = tf.contrib.layers.embed_sequence(ids = input_data,

                                                              vocab_size = vocab_size+1,

                                                              embed_dim = encoder_embedding_size,

                                                              initializer = tf.random_uniform_initializer(0,1))

    encoder_output, encoder_state = encoder_rnn(embedded_message_batch, num_units, num_layers, keep_prob, sequence_length)

    

    decoder_input = preprocessed_targets(target_data, word_to_num, batch_size)

    decoder_embeddings = tf.Variable(tf.random_uniform([vocab_size+1, decoder_embedding_size], 0,1))

    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    

    train_logits, prediction_logits = decoder_rnn(decoder_embedded_input,

                                                  decoder_embeddings,

                                                  encoder_state,

                                                  encoder_output,

                                                  vocab_size,

                                                  sequence_length,

                                                  num_units,

                                                  num_layers,

                                                  word_to_num,

                                                  keep_prob,

                                                  beam_width,

                                                  batch_size)

    return train_logits, prediction_logits
epochs = 100

batch_size = 128

num_units = 512         # = rnn_size =  no. of units in one layer

num_layers = 2

encoder_embedding_size = 512

decoder_embedding_size = 512

beam_width = 3

learning_rate = 0.004

learning_rate_decay = 0.9

min_learning_rate = 0.0001

keep_probability = 0.8
tf.reset_default_graph()

sess.close()

sess = tf.InteractiveSession()



# Loading the model inputs

input_data, targets, learn_rate, keep_prob = model_inputs(batch_size)

sequence_length = tf.placeholder_with_default([threshold]*batch_size, [batch_size], name='sequence_length')



input_shape = tf.shape(input_data)



train_logits, prediction_logits = seq2seq_model(tf.reverse(input_data,[-1]),

                                                targets,

                                                len(word_to_num),

                                                encoder_embedding_size,

                                                decoder_embedding_size,

                                                num_units,

                                                num_layers,

                                                word_to_num,

                                                sequence_length,

                                                keep_prob,

                                                beam_width,

                                                batch_size)



# ????? Tensor for prediction_logits, needed when loading model from a checkpoint



with tf.name_scope('optimization'):

    # Loss Function

    cost = tf.contrib.seq2seq.sequence_loss(train_logits, targets, tf.ones([input_shape[0], sequence_length]))

    # Optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)

    # Gradient Clipping

    gradients = optimizer.compute_gradients(cost)

    clipped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad,var in gradients if grad is not None]

    

    train_op = optimizer.apply_gradients(clipped_gradients)
def pad_sentence_batch(sentence_batch, word_to_num):    

    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""

    max_len = max([len(sentence) for sentence in sentence_batch])

    return [sentence + [word_to_num["<PAD>"]]*(max_len-len(sentence)) for sentence in sentence_batch]
def batch_data(messages, responses, word_to_num, batch_size):

    """Batch questions and answers together"""

    for batch_i in range(0, len(messages)//batch_size):

        start_i = batch_i*batch_size

        end_i = start_i + batch_size

        # Generating i_th batch

        messages_batch_i = messages[start_i : end_i]

        responses_batch_i = responses[start_i : end_i]

        # Padding this batch to same length using previous pad_sentence_batch() function

        pad_messages_batch_i = np.array(pad_sentence_batch(messages_batch_i, word_to_num))

        pad_responses_batch_i = np.array(pad_sentence_batch(responses_batch_i, word_to_num))

        yield pad_messages_batch_i, pad_responses_batch_i
validation_ratio = 0.15

start_index = len(sorted_messages)*3//5

end_index = start_index + ( len(sorted_messages) * validation_ratio)

val_index_range = np.arange(start_index, end_index, 1)

# Validation Set

validation_messages = sorted_messages[val_index_range]

validation_responses = sorted_responses[val_index_range]

# Train Set

train_messages = [sorted_messages[i] for i in range(0, len(sorted_messages)) if i not in val_index_range]

train_responses = [sorted_responses[i] for i in range(0, len(sorted_responses)) if i not in val_index_range]
for var in tf.trainable_variables():

    print(var)