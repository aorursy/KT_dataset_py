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
import tensorflow as tf
from tensorflow.keras import layers, preprocessing, utils
import yaml

print( tf.version)
questions=list()
answers=list()

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file = open(os.path.join(dirname, filename), 'rb')
        docs = yaml.safe_load(file)
        conversations = docs['conversations']
        for con in conversations:
            if len(con)>2:
                questions.append(con[0])
                replies=con[1:]
                ans=""
                for rep in replies:
                    ans+=' '+rep
                answers.append(ans)
            elif len(con)>1:
                questions.append(con[0])
                answers.append(con[1])
                
answers_with_tags=list()
for i in range( len(answers)):
    if type(answers[i])==str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)

answers=list()
for ans in answers_with_tags:
    answers.append('<START> '+ans+' <END>')

tokenizer=preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions+answers)
VOCAB_SIZE= len(tokenizer.word_index)+1
print('VOCAB SIZE : {}'.format(VOCAB_SIZE))
                
from gensim.models import Word2Vec
import re

vocab=[]
for word in tokenizer.word_index:
    vocab.append(word)
    
def tokenize(sentences):
    tokens_list=[]
    vocabulary=[]
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub( '^a-zA-Z',' ',sentence)
        tokens = sentence.split()
        vocabulary+=tokens
        tokens_list.append(tokens)
    return tokens_list,vocabulary

p=tokenize(questions+answers)
model = Word2Vec(p[0])
'''
embedding_matrix = np.zeros((VOCAB_SIZE,100))
for i in range(len(tokenizer.word_index)):
    embedding_matrix[i] = model[vocab[i]]
   ''' 
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
encoder_input_data = np.array( padded_questions )
print(encoder_input_data.shape, maxlen_questions)

tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_input_data = np.array(padded_answers)
print(decoder_input_data.shape, maxlen_answers)

tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)):
    tokenized_answers[i]=tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
one_hot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
decoder_output_data = np.array(one_hot_answers)
print(decoder_output_data.shape)

encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=100 ) 
model.save( 'model.h5' )
def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model
def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')
enc_model , dec_model = make_inference_models()

for _ in range(10):
    states_values = enc_model.predict( str_to_tokens( input( 'Enter question : ' ) ) )
    empty_target_seq = np.zeros( ( 1 , 1 ) )
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition :
        dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
        sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
        sampled_word = None
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += ' {}'.format( word )
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros( ( 1 , 1 ) )  
        empty_target_seq[ 0 , 0 ] = sampled_word_index
        states_values = [ h , c ] 

    print( decoded_translation )
