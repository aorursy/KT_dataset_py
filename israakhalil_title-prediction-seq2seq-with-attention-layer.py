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
df=pd.read_csv("/kaggle/input/cleaned-title-abstract/cleaned_title_abstract.csv")

df.shape
for i in range(2):

    print("abstract:",df['abstract'][i])

    print("title:",df['title'][i])

    print("\n")
import matplotlib.pyplot as plt



abstract_count = []

title_count = []

longest_abstract = 0

longest_title=0



for sent in df['abstract']:

    abstract_count.append(len(sent.split()))

    if len(sent.split()) > longest_abstract:

        longest_abstract = len(sent.split())

        

for sent in df['title']:

    title_count.append(len(sent.split()))

    if len(sent.split()) > longest_title:

        longest_title = len(sent.split())



length_df = pd.DataFrame({'abstract':abstract_count, 'title':title_count})

length_df.hist(bins = 10)

plt.show()
longest_abstract,longest_title
#Check how much % of abstract have 0-300 words

cnt=0

for i in df['abstract']:

    if(len(i.split())<=320):

        cnt=cnt+1

print(cnt/len(df['abstract']))
#Check how much % of title have 0-20 words

cnt=0

for i in df['title']:

    if(len(i.split())<=20):

        cnt=cnt+1

print(cnt/len(df['title']))
max_len_abstract=longest_abstract 

max_len_title=longest_title
val_df = df.sample(frac=0.1, random_state=1007)

train_df = df.drop(val_df.index)

test_df = train_df.sample(frac=0.1, random_state=1007)

train_df.drop(test_df.index, inplace=True)

(train_df.shape,val_df.shape,test_df.shape)
x_train=train_df['abstract']

y_train=train_df['title']

x_val=val_df['abstract']

y_val=val_df['title']
from keras.preprocessing.text import Tokenizer 

from keras.preprocessing.sequence import pad_sequences
#prepare a tokenizer for abstract on training data

x_tokenizer = Tokenizer()

x_tokenizer.fit_on_texts(list(x_train))



#convert abstract sequences into integer sequences

x_train   =   x_tokenizer.texts_to_sequences(x_train) 

x_val   =   x_tokenizer.texts_to_sequences(x_val)



#padding zero upto maximum length

x_train    =   pad_sequences(x_train,  maxlen=max_len_abstract, padding='post',truncating='post') 

x_val     =   pad_sequences(x_val, maxlen=max_len_abstract, padding='post',truncating='post')



x_voc_size   =  len(x_tokenizer.word_index) +1

print("Size of vocabulary in x = {}".format(x_voc_size))
#preparing a tokenizer for title on training data 

y_tokenizer = Tokenizer()

y_tokenizer.fit_on_texts(list(y_train))



#convert title sequences into integer sequences

y_train   =   y_tokenizer.texts_to_sequences(y_train) 

y_val   =   y_tokenizer.texts_to_sequences(y_val) 



#padding zero upto maximum length

y_train    =   pad_sequences(y_train, maxlen=max_len_title, padding='post',truncating='post')

y_val   =   pad_sequences(y_val, maxlen=max_len_title, padding='post',truncating='post')



y_voc_size  =   len(y_tokenizer.word_index) +1

#len(y_tokenizer.word_index) +1

print("Size of vocabulary in Y = {}".format(y_voc_size))
x_train[:1]
from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/attention/attention.py", dst = "../working/attention.py")
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional,dot,Activation

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import plot_model

from attention import AttentionLayer

import warnings

from keras import backend as K 

pd.set_option("display.max_colwidth", 200)

warnings.filterwarnings("ignore")



K.clear_session()



latent_dim = 300

embedding_dim=100



# Encoder

encoder_inputs = Input(shape=(max_len_abstract,))



#embedding layer

enc_emb =  Embedding(x_voc_size, embedding_dim,trainable=True)(encoder_inputs)



#encoder lstm 1

encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)

encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)



#encoder lstm 2

encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)

encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)



#encoder lstm 3

encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)

encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)



# Set up the decoder, using `encoder_states` as initial state.

decoder_inputs = Input(shape=(None,))



#embedding layer

dec_emb_layer = Embedding(y_voc_size, embedding_dim,trainable=True)

dec_emb = dec_emb_layer(decoder_inputs)



decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)

decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])



# Attention layer

attn_layer = AttentionLayer(name='attention_layer')

attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])



# Concat attention input and decoder LSTM output

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])



#dense layer

decoder_dense =  TimeDistributed(Dense(y_voc_size, activation='softmax'))

decoder_outputs = decoder_dense(decoder_concat_input)



# Define the model 

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)



model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=["accuracy"])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

history=model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,

                  epochs=50,

                  callbacks=[es],

                  batch_size=128, 

                  validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
model.save("my_model")
from tensorflow import keras

model = keras.models.load_model("my_model")
from matplotlib import pyplot 

pyplot.plot(history.history['loss'], label='train') 

pyplot.plot(history.history['val_loss'], label='val') 

pyplot.legend() 

pyplot.show()
#build the dictionary to convert the index to word for target and source vocabulary

reverse_target_word_index=y_tokenizer.index_word 

reverse_source_word_index=x_tokenizer.index_word 

target_word_index=y_tokenizer.word_index

# Encode the input sequence to get the feature vector

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])



# Decoder setup

# Below tensors will hold the states of the previous time step

decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_hidden_state_input = Input(shape=(max_len_abstract,latent_dim))



# Get the embeddings of the decoder sequence

dec_emb2= dec_emb_layer(decoder_inputs) 

# To predict the next word in the sequence, set the initial states to the states from the previous time step

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])



#attention inference

attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])

decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])



# A dense softmax layer to generate prob dist. over the target vocabulary

decoder_outputs2 = decoder_dense(decoder_inf_concat) 



# Final decoder model

decoder_model = Model(

    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],

    [decoder_outputs2] + [state_h2, state_c2])
# encoder inference

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])



# decoder inference

# Below tensors will hold the states of the previous time step

decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_hidden_state_input = Input(shape=(max_len_abstract,latent_dim))



# Get the embeddings of the decoder sequence

dec_emb2= dec_emb_layer(decoder_inputs)



# To predict the next word in the sequence, set the initial states to the states from the previous time step

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])



#attention inference

attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])

decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])



# A dense softmax layer to generate prob dist. over the target vocabulary

decoder_outputs2 = decoder_dense(decoder_inf_concat)



# Final decoder model

decoder_model = Model(

[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],

[decoder_outputs2] + [state_h2, state_c2])
def decode_sequence(input_seq):

    # Encode the input as state vectors.

    e_out, e_h, e_c = encoder_model.predict(input_seq)

    

    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1,1))

    

    # Populate the first word of target sequence with the start word.

    target_seq[0, 0] = target_word_index['start']



    stop_condition = False

    decoded_sentence = ''

    while not stop_condition:

      

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])



        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_token = reverse_target_word_index[sampled_token_index]

        

        if(sampled_token!='end'):

            decoded_sentence += ' '+sampled_token



        # Exit condition: either hit max length or find stop word.

        if (sampled_token == 'end'  or len(decoded_sentence.split()) >= (max_len_title-1)):

            stop_condition = True



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1,1))

        target_seq[0, 0] = sampled_token_index



        # Update internal states

        e_h, e_c = h, c



    return decoded_sentence
def seq2summary(input_seq):

    newString=''

    for i in input_seq:

        if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):

            newString=newString+reverse_target_word_index[i]+' '

    return newString



def seq2text(input_seq):

    newString=''

    for i in input_seq:

        if(i!=0):

            newString=newString+reverse_source_word_index[i]+' '

    return newString
for i in range(5):

    print("abstract:",seq2text(x_val[i]))

    print("Original title:",seq2summary(y_val[i]))

    print("Predicted title:",decode_sequence(x_val[i].reshape(1,max_len_abstract)))

    print("\n")
test_df

x_test=test_df["abstract"]

y_test=test_df["title"]

x_test[:1]
x_test   =   x_tokenizer.texts_to_sequences(x_test) 

x_test   =   pad_sequences(x_test,  maxlen=max_len_abstract, padding='post')



y_test   =   y_tokenizer.texts_to_sequences(y_test) 

y_test   =   pad_sequences(y_test,  maxlen=max_len_title, padding='post') 

from nltk.translate.bleu_score import sentence_bleu

for i in range(5):

    print("abstract:",seq2text(x_test[i]))

    original_title=seq2summary(y_test[i])

    predicted_title=decode_sequence(x_test[i].reshape(1,max_len_abstract))

    

    print("Original title:",original_title)

    print("Predicted title:",predicted_title)

    

    

    print("Match:",sentence_bleu([original_title.split()], predicted_title.split()))

    print("\n")
x_test[:200]
from nltk.translate.bleu_score import corpus_bleu

for i in range (len(x_test[:200])):

    original_title=seq2summary(y_test[i])

    predicted_title=decode_sequence(x_test[i].reshape(1,max_len_abstract))

    hypothesis=predicted_title.split()

    reference=original_title.split()

    references=[reference]

    list_of_references = [references]

    list_of_hypotheses = [hypothesis]

    print(corpus_bleu(list_of_references, list_of_hypotheses))

    

    '''import nltk

>>> hypothesis = ['This', 'is', 'cat'] 

>>> reference = ['This', 'is', 'a', 'cat']

>>> references = [reference] # list of references for 1 sentence.

>>> list_of_references = [references] # list of references for all sentences in corpus.

>>> list_of_hypotheses = [hypothesis] # list of hypotheses that corresponds to list of references.

>>> nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses)

0.6025286104785453

>>> nltk.translate.bleu_score.sentence_bleu(references, hypothesis)

0.6025286104785453'''