import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train_t = pd.read_csv('/kaggle/input/language-translation-englishfrench/eng_-french.csv')
train_t.columns

df = train_t[0:100000]
df
english_text = df['English words/sentences']
french_text = df['French words/sentences']
import re
english = []
french = []
for i in range(len(english_text)):
    text = english_text[i].lower()
    text = re.sub('[^a-zA-Z]',' ',text)
    english.append(text)
    

for i in range(len(french_text)):
    ftext = french_text[i].lower()
    ftext = (re.sub("[^a-zA-Z' àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]",' ',ftext))
    french.append("START_ " + ftext + " _END")

french
#Vocabulary of English
all_eng_words = set()
for i in english:
    for j in i.split():
        all_eng_words.add(j)

#vocabulary of french
all_fre_words = set()
for i in french:
    for j in i.split():
        all_fre_words.add(j)

#maxlen of the source sequence
max_length_src = 0
for i in english:
    a = len(i.split())
    if a>max_length_src:
        max_length_src = a
        
#maxlen of the target sequence
max_length_tar = 0
for j in french:
    b = len(j.split())
    if b>max_length_tar:
        max_length_tar = b
        

input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_fre_words))

# Calculate Vocab size for both source and targe
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_fre_words)


#indexs for input and target sequences
input_index = dict([(words,i) for i,words in enumerate(input_words)])
target_index = dict([(word, i) for i, word in enumerate(target_words)])

reverse_input_index = dict((i, word) for word, i in input_index.items())
reverse_target_index = dict((i, word) for word, i in target_index.items())
print(max_length_src)
print(max_length_tar)
print(num_encoder_tokens)
print(num_decoder_tokens)
encoder_input_data = np.zeros((100000, max_length_src, num_encoder_tokens),dtype='float32')
decoder_input_data = np.zeros((100000, max_length_tar, num_decoder_tokens),dtype='float32')
decoder_target_data = np.zeros((100000, max_length_tar, num_decoder_tokens),dtype='float32')
for j in range(100000):
    for i,text in enumerate(english[j].split()):
        encoder_input_data[j,i,input_index[text]] = 1.

for j in range(100000):
    for i,text in enumerate(french[j].split()):
        decoder_input_data[j,i,target_index[text]] = 1.
        if i>0:
            decoder_target_data[j,i-1,target_index[text]] = 1.
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional
batch_size = 64
epochs = 100
latent_dim = 256 #size of the lstms hidden state
#Time to bulid the model

#inputs for the encoder
encoder_inputs = Input(shape=(None,num_encoder_tokens))
#encoder lstm
encod_lstm = (LSTM(latent_dim,return_state = True))
encoder_output,state_h,state_c = encod_lstm(encoder_inputs)

#hidden from encoder to pass to the decoder as initial hidden state
encoder_states = [state_h,state_c]

#inputs for the decoder
decoder_inputs = Input(shape=(None,num_decoder_tokens))
#decoder lstm 
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_output,_,_= decoder_lstm(decoder_inputs,initial_state = encoder_states)
#The decoder output is passed through the softmax layer that will learn to classify the correct french character
#Activation functions are used to transform vectors before computing the loss in the training phase
#for more on softmax https://gombru.github.io/2018/05/23/cross_entropy_loss/
dense_layer = Dense(num_decoder_tokens, activation='softmax')
decoder_output = dense_layer(decoder_output)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_output)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
model.fit([encoder_input_data,decoder_input_data],decoder_target_data,batch_size= 64,epochs= 50,validation_split=0.2)
encoder_model = Model(encoder_inputs,encoder_states)

decoder_state_h = Input(shape=(latent_dim,))
decoder_state_c = Input(shape=(latent_dim,))
decode_state = [decoder_state_h,decoder_state_c]

decoder_outputs,state_h,state_c = decoder_lstm(decoder_inputs,initial_state = decode_state)
decoder_states = [state_h, state_c]
decoder_outputs = dense_layer(decoder_outputs)

decoder_model = Model([decoder_inputs] + decode_state,[decoder_outputs] + decoder_states)
from keras.utils.vis_utils import plot_model
plot_model(decoder_model, to_file='model.png', show_shapes=True)
def decode_sequence(input_seq):
    # encode the input sequence to get the internal state vectors.
    states_value = encoder_model.predict(input_seq)
  
    # generate empty target sequence of length 1 with only the start character
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_index['START_']] = 1.
  
    # output sequence loop
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    
        # sample a token and add the corresponding character to the 
        # decoded sequence
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_index[sampled_token_index]
        
        if (sampled_char == "_END" or len(decoded_sentence) > max_length_tar):
            stop_condition = True
            break
            
        decoded_sentence += sampled_char
        decoded_sentence +=' '
      
        # update the target sequence (length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
    
        # update states
        states_value = [h, c]
    
    return decoded_sentence
            
toks = ['i love you','run fast','she is the client','my name is tom']
for t in toks:
    input_sentence = t
    test_sentence_tokenized = np.zeros((1, max_length_src, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(input_sentence.split()):
        test_sentence_tokenized[0, t, input_index[char]] = 1.
    print(input_sentence)
    print(decode_sequence(test_sentence_tokenized))
    print(' ')
#result je vous aime is i love you in english
#result un fait vite is move fast in english
#result elle est dans le tu is she’s in the you 
#result mon nom est tom is my name is tom