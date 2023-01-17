import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import tensorflow as tf
from tensorflow import keras
df = pd.read_csv('/kaggle/input/moviedialog.csv')
df.head()
df.shape
input_texts = []
target_texts = []

input_vocabulary = set()
output_vocabulary = set()
start_token = '\t'
stop_token = '\n'
max_training_examples = min(25000 , len(df) - 1)

for input_text , target_text in zip(df.statement , df.reply):
    target_text = start_token + target_text + stop_token
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_vocabulary:
            input_vocabulary.add(char)
    for char in target_text:
        if char not in output_vocabulary:
            output_vocabulary.add(char)
len(output_vocabulary)
input_vocabulary = sorted(list(input_vocabulary))
output_vocabulary = sorted(list(output_vocabulary))

input_vocabulary_size = len(input_vocabulary)
output_vocabulary_size = len(output_vocabulary)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char , i) for i,char in enumerate(input_vocabulary)])
target_token_index = dict([(char , i) for i,char in enumerate(output_vocabulary)])

reverse_input_char_index = dict([(i,char) for char,i in input_token_index.items()])
reverse_target_char_index = dict([(i,char) for char,i in target_token_index.items()])
input_token_index
encoder_input_data = np.zeros((len(input_texts) , max_encoder_seq_length , input_vocabulary_size) , dtype = 'float32')
decoder_input_data = np.zeros((len(target_texts) , max_decoder_seq_length , output_vocabulary_size) , dtype = 'float32')
decoder_target_data = np.zeros((len(target_texts) , max_decoder_seq_length , output_vocabulary_size) , dtype = 'float32')


for i ,(input_text , target_text) in enumerate(zip(input_texts,target_texts)):
    
    for t,char in enumerate(input_text):
        encoder_input_data[i , t , input_token_index[char]] = 1
    
    for t,char in enumerate(target_text):
        decoder_input_data[i , t , target_token_index[char]] = 1
        
    for t,char in enumerate(target_text):
        decoder_target_data[i , t-1 , target_token_index[char]] = 1
        
#building the model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input , LSTM , Dense
batch_size = 64
epochs = 100
num_neurons = 256

encoder_inputs = Input(shape = (None , input_vocabulary_size))
encoder = LSTM(num_neurons , return_state = True)
encoder_outputs , state_h , state_c = encoder(encoder_inputs)
encoder_states = [state_h , state_c]


decoder_inputs = Input(shape = (None , output_vocabulary_size))
decoder_lstm = LSTM(num_neurons , return_state = True , return_sequences = True)
decoder_outputs , _ , _ = decoder_lstm(decoder_inputs , initial_state = encoder_states)
decoder_dense = Dense(output_vocabulary_size , activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs , decoder_inputs] , decoder_outputs)
model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
checkpoint_filepath = '/kaggle/input/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
history = model.fit([encoder_input_data , decoder_input_data] , decoder_target_data , batch_size = 64 , epochs = epochs,
         validation_split = 0.1 , callbacks=[model_checkpoint_callback])
type(history)
model.save('./trained3.h5')

encoder_model = Model(encoder_inputs, encoder_states)
thought_input = [
     Input(shape=(num_neurons,)), Input(shape=(num_neurons,))]
decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=thought_input)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
      [decoder_inputs] + thought_input,
        [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    thought = encoder_model.predict(input_seq)
    target_seq = np.zeros((1 , 1 , output_vocabulary_size))
    target_seq[0 , 0 , target_token_index[stop_token]] = 1
    stop_condition = False
    generated_sequence = ''
    
    while not stop_condition:
        output_tokens , h , c = decoder_model.predict([target_seq] + thought)
        generated_token_idx = np.argmax(output_tokens[0, -1, :])
        generated_char = reverse_target_char_index[generated_token_idx]
        generated_sequence += generated_char
        
        if (generated_char == stop_token) or (len(generated_sequence) > max_decoder_seq_length):
            stop_condition = True
            
        target_seq = np.zeros((1 , 1 , output_vocabulary_size))
        target_seq[0, 0, generated_token_idx] = 1
        thought = [h, c]
        
    return generated_sequence
def response(input_text):
    input_seq = np.zeros((1 , max_encoder_seq_length , input_vocabulary_size) ,
                                  dtype = 'float32')
    for t, char in enumerate(input_text):
        input_seq[0, t, input_token_index[char]] = 1
    decoded_sentence = decode_sequence(input_seq)
    print('Bot Reply (Decoded sentence):', decoded_sentence)
response("hello nice here")
encoder_model.save('./encodermodel3.h5')
decoder_model.save('./decodermodel3.h5')
response("wow")
