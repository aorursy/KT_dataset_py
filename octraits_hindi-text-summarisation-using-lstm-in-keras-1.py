from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = 10000 # specify 'None' if want to read whole file

# test.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/test.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'test.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
nRowsRead = 100000 # specify 'None' if want to read whole file

# train.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/train.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'train.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
df.describe()
df.info()
import re

def preprocess_tokenize(text):

      # for removing punctuation from sentencesc

    text = str(text)

    text = re.sub(r'(\d+)', r'', text)

    

    text = text.replace('\n', '')

    text = text.replace('\r', '')

    text = text.replace('\t', '')

    text = text.replace('\u200d', '')

    text=re.sub("(__+)", ' ', str(text)).lower()   #remove _ if it occors more than one time consecutively

    text=re.sub("(--+)", ' ', str(text)).lower()   #remove - if it occors more than one time consecutively

    text=re.sub("(~~+)", ' ', str(text)).lower()   #remove ~ if it occors more than one time consecutively

    text=re.sub("(\+\++)", ' ', str(text)).lower()   #remove + if it occors more than one time consecutively

    text=re.sub("(\.\.+)", ' ', str(text)).lower()   #remove . if it occors more than one time consecutively

        

    text=re.sub(r"[<>()|&©@#ø\[\]\'\",;:?.~*!]", ' ', str(text)).lower() #remove <>()|&©ø"',;?~*!

    text = re.sub("([a-zA-Z])",' ',str(text)).lower()

    text = re.sub("(\s+)",' ',str(text)).lower()

    #text = text.split(' ')

    #text = [x for x in text if(x!='')]

    #text.insert(0,'<sos>')

    #text.insert(len(text),'<eos>')

    return text
import pandas as pd

import torch

import random

import numpy as np


#ensuring device is gpu



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
train_data_src = df['headline']

train_data_trg = df['article']
tokenized_corpus_src = [preprocess_tokenize(x) for x in train_data_src]  #these are headlines

tokenized_corpus_trg = [preprocess_tokenize(x) for x in train_data_trg] # these are articles

import matplotlib.pyplot as plt



text_word_count = []

summary_word_count = []



# populate the lists with sentence lengths

for i in tokenized_corpus_trg:

      text_word_count.append(len(i.split()))



for i in tokenized_corpus_src:

      summary_word_count.append(len(i.split()))



length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})



length_df.hist(bins = 30)

plt.show()
df['Text_Cleaned'] = tokenized_corpus_trg  

print("::::: Text_Cleaned :::::")

print(df['Text_Cleaned'][0:5], "\n")





df['Summary_Cleaned'] =  tokenized_corpus_src 

print("::::: Summary :::::")

print(df['Summary_Cleaned'][0:5], "\n")



corpus = list(df['Text_Cleaned'])
print(df['Text_Cleaned'][0])

print(df['Summary_Cleaned'][0])

text_count = []

summary_count = []



for sent in df['Text_Cleaned']:

    text_count.append(len(sent.split()))

for sent in df['Summary_Cleaned']:

    summary_count.append(len(sent.split()))



graph_df = pd.DataFrame()

graph_df['text'] = text_count

graph_df['summary'] = summary_count
graph_df['text'].describe()

graph_df['summary'].describe()

graph_df['text'].hist(bins = 25, range=(0, 300))

plt.show()
graph_df['summary'].hist(bins = 15, range=(0, 20))

plt.show()
count = 0

for i in graph_df['text']:

    if i > 10 and i <= 300:

        count = count + 1

print(count / len(graph_df['text']))
count = 0

for i in graph_df['summary']:

    if i > 1 and i <= 16:

        count = count + 1

print(count / len(graph_df['summary']))
max_text_len = 300

max_summary_len = 16

cleaned_text = np.array(df['Text_Cleaned'])

cleaned_summary = np.array(df['Summary_Cleaned'])



short_text = []

short_summary = []



for i in range(len(cleaned_text)):

    if(len(cleaned_summary[i].split()) <= max_summary_len 

       and len(cleaned_summary[i].split()) > 1 

       and len(cleaned_text[i].split()) <= max_text_len ):

        short_text.append(cleaned_text[i])

        short_summary.append(cleaned_summary[i])

        

post_pre = pd.DataFrame({'text':short_text,'summary':short_summary})
post_pre['summary'] = post_pre['summary'].apply(lambda x : 'sostok '+ x + ' eostok')
post_pre.shape
post_pre
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer 

from keras.preprocessing.sequence import pad_sequences



# train test split

x_tr,x_test,y_tr,y_test = train_test_split(np.array(post_pre['text']),

                                         np.array(post_pre['summary']),

                                         test_size = 0.2,

                                         random_state = 0,

                                         shuffle = True)

# train validation split

x_tr,x_val,y_tr,y_val = train_test_split(x_tr,

                                         y_tr,

                                         test_size = 0.2,

                                         random_state = 0,

                                         shuffle = True)
x_tr.shape
x_test.shape
x_val.shape

# Tokenize text to get the vocab count

#prepare a tokenizer for training data

x_tokenizer = Tokenizer() 

x_tokenizer.fit_on_texts(list(x_tr))



#prepare a tokenizer for reviews on training data

y_tokenizer = Tokenizer()   

y_tokenizer.fit_on_texts(list(y_tr))
thresh=4



cnt=0

tot_cnt=0

freq=0

tot_freq=0



for key,value in x_tokenizer.word_counts.items():

    tot_cnt=tot_cnt+1

    tot_freq=tot_freq+value

    if(value<thresh):

        cnt=cnt+1

        freq=freq+value

    

print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)

print("Total Coverage of rare words:",(freq/tot_freq)*100)
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from time import time

#prepare a tokenizer for reviews on training data

x_tokenizer = Tokenizer(num_words = tot_cnt - cnt) 

x_tokenizer.fit_on_texts(list(x_tr))



#convert text sequences into integer sequences (i.e one-hot encodeing all the words)

x_tr_seq    =   x_tokenizer.texts_to_sequences(x_tr) 

x_val_seq   =   x_tokenizer.texts_to_sequences(x_val)

x_test_seq = x_tokenizer.texts_to_sequences(x_test)



#padding zero upto maximum length

x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')

x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

x_test = pad_sequences(x_test_seq, maxlen=max_text_len, padding='post')



#size of vocabulary ( +1 for padding token)

x_voc   =  x_tokenizer.num_words + 1



print("Size of vocabulary in X = {}".format(x_voc))
thresh=6



cnt=0

tot_cnt=0

freq=0

tot_freq=0



for key,value in y_tokenizer.word_counts.items():

    tot_cnt=tot_cnt+1

    tot_freq=tot_freq+value

    if(value<thresh):

        cnt=cnt+1

        freq=freq+value

    

print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)

print("Total Coverage of rare words:",(freq/tot_freq)*100)
#prepare a tokenizer for reviews on training data

y_tokenizer = Tokenizer(num_words = tot_cnt-cnt) 

y_tokenizer.fit_on_texts(list(y_tr))



#convert text sequences into integer sequences (i.e one hot encode the text in Y)

y_tr_seq    =   y_tokenizer.texts_to_sequences(y_tr) 

y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 

y_test_seq = y_tokenizer.texts_to_sequences(y_test) 



#padding zero upto maximum length

y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')

y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

y_test = pad_sequences(y_test_seq, maxlen=max_summary_len, padding='post')



#size of vocabulary

y_voc  =   y_tokenizer.num_words +1

print("Size of vocabulary in Y = {}".format(y_voc))
y_tokenizer.word_counts['sostok'],len(y_tr)

from tensorflow.keras.backend import clear_session

import tensorflow as tf

import gensim

from numpy import *

import numpy as np

import pandas as pd 

import re

from bs4 import BeautifulSoup

from keras.preprocessing.text import Tokenizer 

from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

import warnings

warnings.filterwarnings("ignore")

ind=[]

for i in range(len(y_tr)):

    cnt=0

    for j in y_tr[i]:

        if j!=0:

            cnt=cnt+1

    if(cnt==2):

        ind.append(i)



y_tr=np.delete(y_tr,ind, axis=0)

x_tr=np.delete(x_tr,ind, axis=0)
ind=[]

for i in range(len(y_val)):

    cnt=0

    for j in y_val[i]:

        if j!=0:

            cnt=cnt+1

    if(cnt==2):

        ind.append(i)



y_val=np.delete(y_val,ind, axis=0)

x_val=np.delete(x_val,ind, axis=0)
import tensorflow as tf

import os

from tensorflow.python.keras.layers import Layer

from tensorflow.python.keras import backend as K





class AttentionLayer(Layer):

    """

    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).

    There are three sets of weights introduced W_a, U_a, and V_a

     """



    def __init__(self, **kwargs):

        super(AttentionLayer, self).__init__(**kwargs)



    def build(self, input_shape):

        assert isinstance(input_shape, list)

        # Create a trainable weight variable for this layer.



        self.W_a = self.add_weight(name='W_a',

                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),

                                   initializer='uniform',

                                   trainable=True)

        self.U_a = self.add_weight(name='U_a',

                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),

                                   initializer='uniform',

                                   trainable=True)

        self.V_a = self.add_weight(name='V_a',

                                   shape=tf.TensorShape((input_shape[0][2], 1)),

                                   initializer='uniform',

                                   trainable=True)



        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end



    def call(self, inputs, verbose=False):

        """

        inputs: [encoder_output_sequence, decoder_output_sequence]

        """

        assert type(inputs) == list

        encoder_out_seq, decoder_out_seq = inputs

        if verbose:

            print('encoder_out_seq>', encoder_out_seq.shape)

            print('decoder_out_seq>', decoder_out_seq.shape)



        def energy_step(inputs, states):

            """ Step function for computing energy for a single decoder state """



            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))

            assert isinstance(states, list) or isinstance(states, tuple), assert_msg



            """ Some parameters required for shaping tensors"""

            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]

            de_hidden = inputs.shape[-1]



            """ Computing S.Wa where S=[s0, s1, ..., si]"""

            # <= batch_size*en_seq_len, latent_dim

            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))

            # <= batch_size*en_seq_len, latent_dim

            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))

            if verbose:

                print('wa.s>',W_a_dot_s.shape)



            """ Computing hj.Ua """

            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            if verbose:

                print('Ua.h>',U_a_dot_h.shape)



            """ tanh(S.Wa + hj.Ua) """

            # <= batch_size*en_seq_len, latent_dim

            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))

            if verbose:

                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)



            """ softmax(va.tanh(S.Wa + hj.Ua)) """

            # <= batch_size, en_seq_len

            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))

            # <= batch_size, en_seq_len

            e_i = K.softmax(e_i)



            if verbose:

                print('ei>', e_i.shape)



            return e_i, [e_i]



        def context_step(inputs, states):

            """ Step function for computing ci using ei """

            # <= batch_size, hidden_size

            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)

            if verbose:

                print('ci>', c_i.shape)

            return c_i, [c_i]



        def create_inital_state(inputs, hidden_size):

            # We are not using initial states, but need to pass something to K.rnn funciton

            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim

            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)

            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)

            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim

            return fake_state



        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])

        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim



        """ Computing energy outputs """

        # e_outputs => (batch_size, de_seq_len, en_seq_len)

        last_out, e_outputs, _ = K.rnn(

            energy_step, decoder_out_seq, [fake_state_e],

        )



        """ Computing context vectors """

        last_out, c_outputs, _ = K.rnn(

            context_step, e_outputs, [fake_state_c],

        )



        return c_outputs, e_outputs



    def compute_output_shape(self, input_shape):

        """ Outputs produced by the layer """

        return [

            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),

            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))

        ]
K.clear_session()



latent_dim = 300

embedding_dim=100



# Encoder

encoder_inputs = Input(shape=(max_text_len,))



#embedding layer

enc_emb =  Embedding(x_voc, embedding_dim,trainable=True)(encoder_inputs)



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

dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True)

dec_emb = dec_emb_layer(decoder_inputs)



decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)

decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])



# Attention layer

attn_layer = AttentionLayer(name='attention_layer')

attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])



# Concat attention input and decoder LSTM output

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])



#dense layer

decoder_dense =  TimeDistributed(Dense(y_voc, activation='softmax'))

decoder_outputs = decoder_dense(decoder_concat_input)



# Define the model 

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)



model.summary()

tf.keras.utils.plot_model(

    model, to_file='model.png', show_shapes=True, show_layer_names=True,

    rankdir='TB', expand_nested=False, dpi=96

)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)



history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))
from matplotlib import pyplot

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()
reverse_target_word_index=y_tokenizer.index_word

reverse_source_word_index=x_tokenizer.index_word

target_word_index=y_tokenizer.word_index



# Encode the input sequence to get the feature vector

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])



# Decoder setup

# Below tensors will hold the states of the previous time step

decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))



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

    target_seq[0, 0] = target_word_index['sostok']



    stop_condition = False

    decoded_sentence = ''

    while not stop_condition:

      

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])



        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_token = reverse_target_word_index[sampled_token_index]

        

        if(sampled_token!='eostok'):

            decoded_sentence += ' '+sampled_token



        # Exit condition: either hit max length or find stop word.

        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):

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

        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):

            newString=newString+reverse_target_word_index[i]+' '

    return newString



def seq2text(input_seq):

    newString=''

    for i in input_seq:

        if(i!=0):

            newString=newString+reverse_source_word_index[i]+' '

    return newString
for i in range(0,100):

    print("Review:",seq2text(x_tr[i]))

    print("Original summary:",seq2summary(y_tr[i]))

    print("Predicted summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))

    print("\n")