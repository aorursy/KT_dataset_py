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
import pandas as pd

from tqdm.notebook import tqdm

import string

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from PIL import Image

from tensorflow.keras.utils import to_categorical

import numpy as np

import pandas as pd
with open('/kaggle/input/dlassignment4d2/train.en') as f:

    train_en = f.readlines()

train_en = [x.strip() for x in train_en] 

with open('/kaggle/input/dlassignment4d2/train.ta') as f:

    train_ta = f.readlines()

train_ta = [x.strip() for x in train_ta] 



with open('/kaggle/input/dlassignment4d2/dev.en') as f:

    val_en = f.readlines()

val_en = [x.strip() for x in val_en] 

with open('/kaggle/input/dlassignment4d2/dev.ta') as f:

    val_ta = f.readlines()

val_ta = [x.strip() for x in val_ta] 



with open('/kaggle/input/finaltest/test.en') as f:

    test_en = f.readlines()

test_en = [x.strip() for x in test_en] 

with open('/kaggle/input/finaltest/test.ta') as f:

    test_ta = f.readlines()

test_ta = [x.strip() for x in test_ta] 



print("Train eng",len(train_en))

print("Train tam",len(train_ta))

print("Val eng",len(val_en))

print("Val tam",len(val_ta))
start_token = 'starttoken'

end_token = 'endtoken'
tkzr_en = Tokenizer()

# tkzr_en.fit_on_texts([start_token,end_token])

tkzr_en.fit_on_texts(train_en+val_en+test_ta)

vocab_len_en = len(tkzr_en.word_index)+1



tkzr_ta = Tokenizer()

tkzr_ta.fit_on_texts([start_token,end_token])

tkzr_ta.fit_on_texts(train_ta+val_ta+test_ta)

vocab_len_ta = len(tkzr_ta.word_index)+1

vocab_len_ta
# start_token_t = tkzr_en.word_index[start_token]

# end_token_t = tkzr_en.word_index[end_token]

train_en_t = tkzr_en.texts_to_sequences(train_en)

val_en_t = tkzr_en.texts_to_sequences(val_en)

test_en_t = tkzr_en.texts_to_sequences(test_en)

# train_en_t = [[start_token_t]+i+[end_token_t] for i in train_en_t]



start_token_t = tkzr_ta.word_index[start_token]

end_token_t = tkzr_ta.word_index[end_token]

train_ta_t = tkzr_ta.texts_to_sequences(train_ta)

train_ta_t = [[start_token_t]+i+[end_token_t] for i in train_ta_t]
f = open("/kaggle/input/glove6b200d/glove.6B.200d.txt", encoding="utf-8")

dictionary = set(tkzr_en.word_index.keys())

dic_em = {}

pbar = tqdm(total=4*10e4)

while True:

    pbar.update(1)

    line = f.readline() 

    if not line: 

        break

    values = line.split()

    word = values[0]

    if(word in dictionary):

        dic_em[word] = np.asarray(values[1:], dtype='float32')

        dictionary.remove(word)

# #     embeddings_index[word] = coefs

f.close()

pbar.close()



for rem_word in tqdm(dictionary):

    dic_em[rem_word] = np.ones(200)

    

dic_em_csv = pd.DataFrame.from_dict(dic_em,orient='index')

# dic_em_csv.to_csv('glove_mapping_dl_assignment.csv')
en_words_set = set(tkzr_en.word_index.keys())

embedding_weights = np.zeros((vocab_len_en,200))

for word in tqdm(tkzr_en.word_index):

    try:

        embedding_weights[tkzr_en.word_index[word]] = dic_em_csv.loc[word][:200]

    except Exception as e:

        print(word,e)

        pass
max_en_len = max(len(i) for i in train_en_t)

max_ta_len = max(len(i) for i in train_ta_t)

print(max_en_len,max_ta_len)
to_categorical([0,1,2,3],4)
encoder_input_batch = list()

decoder_input_batch = list()

decoder_output_batch = list()

for (i,(en_t,ta_t)) in enumerate(zip(train_en_t,train_ta_t)):

    encoder_input_batch.append(en_t)

#     ta_t = pad_sequences(ta_t,max_ta_len,padding='post')

    decoder_input_batch.append(ta_t)

#     decoder_output_batch.append(ta_t,num_classes=vocab_len_ta

    

    if(i==3):

        break

encoder_input_batch = pad_sequences(encoder_input_batch,max_en_len,padding='post')

decoder_input_batch = pad_sequences(decoder_input_batch,max_ta_len,padding='post')

for ta_item in decoder_input_batch:

    decoder_output_batch.append(to_categorical(ta_item,num_classes=vocab_len_ta))

    



def generator(batch_size=1):

    num = 0

    while(True):

        encoder_input_batch,decoder_input_batch,decoder_output_batch = list(),list(),list()

        for (i,(en_t,ta_t)) in enumerate(zip(train_en_t,train_ta_t)):

            encoder_input_batch.append(en_t)

            decoder_input_batch.append(ta_t)

            num+=1

            if(num==batch_size):

                encoder_input_batch = pad_sequences(encoder_input_batch,max_en_len,padding='post')

                decoder_input_batch = pad_sequences(decoder_input_batch,max_ta_len,padding='post')

                for ta_item in decoder_input_batch:

                    decoder_output_batch.append(to_categorical(ta_item,num_classes=vocab_len_ta))

                yield([encoder_input_batch,decoder_input_batch],np.asarray(decoder_output_batch))

                encoder_input_batch,decoder_input_batch,decoder_output_batch = list(),list(),list()

                num=0

gen = generator(5)
t1 = next(gen)
print(t1[0][0].shape)

print(t1[0][1].shape)

print(t1[1].shape)
from keras import Sequential

from keras.layers import Input, Dense, Dropout, Conv2D,AveragePooling2D,Flatten, InputLayer, BatchNormalization,LSTM,Embedding,Concatenate,RepeatVector,Attention

from keras.layers.merge import add

from keras.models import Model

from keras.utils import plot_model

import tensorflow as tf

from keras.losses import sparse_categorical_crossentropy

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

latent_dim = 512



encoder_inp = Input(shape = (max_en_len,))

encoder_em = Embedding(vocab_len_en, 200,weights=[embedding_weights],trainable=False)(encoder_inp)

encoder_lstm = LSTM(latent_dim, return_state=True)

encoder_out,enc_h,enc_c = encoder_lstm(encoder_em)



encoded_state = [enc_h,enc_c]



decoder_inp = Input(shape = (None,),dtype='int64')

decoder_em_layer = Embedding(vocab_len_ta, 200)

decoder_em = decoder_em_layer(decoder_inp)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_out,dec_h,dec_c = decoder_lstm(decoder_em, initial_state  = encoded_state)

decoder_dense = Dense(vocab_len_ta, activation = 'softmax')

decoder_out = decoder_dense(decoder_out)



model = Model([encoder_inp, decoder_inp], decoder_out)



# model.compile(optimizer='rmsprop', loss=sparse_loss,target_tensors=[decoder_target])

# model.compile(optimizer='Adam', loss=)



print(model.summary())

plot_model(model,show_shapes=True)
opt = Adam(learning_rate=2*1e-4)

model.compile(optimizer=opt, loss='categorical_crossentropy')

mcp_save = ModelCheckpoint('/kaggle/working/mdl_wts.hdf5', save_best_only=True, monitor='loss', mode='min')

early_stopping = EarlyStopping(monitor='loss',min_delta=0,patience=5,verbose=1, mode='auto',restore_best_weights=True)



batch_size = 8

his = model.fit(generator(batch_size),epochs=10,verbose=1,

#           steps_per_epoch=len(train_en_t)//batch_size

          steps_per_epoch=21

         ,callbacks=[mcp_save])
from matplotlib import pyplot as plt

plt.plot(his.history['loss'])

# plt.plot(his.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.savefig('/kaggle/working/Q3_model.png', bbox_inches='tight')

plt.show()
encoder_model = Model(encoder_inp,encoded_state)



decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]



dec_emb2= decoder_em_layer(decoder_inp) # Get the embeddings of the decoder sequence



# To predict the next word in the sequence, set the initial states to the states from the previous time step

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)

decoder_states2 = [state_h2, state_c2]

decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary



# Final decoder model

decoder_model = Model(

    [decoder_inp] + decoder_states_inputs,

    [decoder_outputs2] + decoder_states2)
plot_model(decoder_model,show_shapes=True)
plot_model(encoder_model,show_shapes=True)
tkzr_ta.sequences_to_texts([[1,2,3],[1,2,3,4,5]])
y_pred_all = list()

for (i,en_t) in enumerate(tqdm(test_en_t)):

    X1_test = np.array(pad_sequences([en_t],max_en_len,padding='post'))

    curr_state = encoder_model.predict(X1_test)

    inp_word = np.array([[start_token_t]])

    y_pred_ix = []

    while True:

        pred_word,h,c = decoder_model.predict([inp_word]+curr_state)

        pred_index = np.argmax(pred_word, axis = -1)[0][0]

        if(pred_index == end_token_t or len(y_pred_ix)>max_ta_len):

            break

        y_pred_ix.append(pred_index)

        inp_word = np.array([[pred_index]])

        curr_state = [h,c]

    y_pred_all.append(y_pred_ix)

y_pred_all_txt = tkzr_ta.sequences_to_texts(y_pred_all)
pd.DataFrame({'x':test_en, 'y_pred':y_pred_all_txt, 'y_actual':test_ta}).to_csv('/kaggle/working/final_test_res.csv')



from nltk.translate.bleu_score import sentence_bleu



df = pd.DataFrame({'x':test_en, 'y_pred':y_pred_all_txt, 'y_actual':test_ta})

df.to_csv('/kaggle/working/test_res_final.csv')

# df['BLEU'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred']), axis=1)

df['BLEUone'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(1, 0, 0, 0)), axis=1)

df['BLEUtwo'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.5, 0.5, 0, 0)), axis=1)

df['BLEUthr'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.33, 0.33, 0.33, 0)), axis=1)

df['BLEUfou'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.25, 0.25, 0.25, 0.25)), axis=1)



df.to_csv('/kaggle/working/test_res-BLEU.csv')



from keras import Sequential

from keras.layers import Input, Dense, Dropout, Conv2D,AveragePooling2D,Flatten, InputLayer, BatchNormalization,LSTM,Embedding,Concatenate,RepeatVector,Attention

from keras.layers.merge import add

from keras.models import Model

from keras.utils import plot_model

import tensorflow as tf

from keras.losses import sparse_categorical_crossentropy,categorical_crossentropy

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

dec_h = tf.zeros(shape=(2,512))

enc_h = tf.zeros(shape=(2,46,512))

dec_h = tf.expand_dims(dec_h, 1)

con_v = tf.matmul(enc_h,dec_h,transpose_b=True)

weights = tf.ones(shape=(2,46,1))

con_v*=weights

print(con_v.shape)

sof_score = tf.nn.softmax(con_v)

print(sof_score.shape)

(sof_score*enc_h).shape
score = tf.matmul(enc_h, dec_h, transpose_b=True)

score*=weights

print(score.shape)
class LuongAttention(tf.keras.layers.Layer):

    def __init__(self, use_scale=False):

        super(LuongAttention, self).__init__()

        self.use_scale = use_scale    



    def build(self,input_shape):

        if self.use_scale:

            self.scale = self.add_weight(

              name='scale',

              shape=(input_shape[1][-2],1),

              initializer=tf.keras.initializers.Ones(),

              dtype=self.dtype,

              trainable=True)

        else:

            self.scale = None

        

    def call(self, inputs):

        query = inputs[0]

        values = inputs[1]

        # query hidden state shape == (batch_size, hidden size)

        # query_with_time_axis shape == (batch_size, 1, hidden size)

        # values shape == (batch_size, max_len, hidden size)

        # we are doing this to broadcast addition along the time axis to calculate the score

        query_with_time_axis = tf.expand_dims(query, 1)



        # score shape == (batch_size, max_length, 1)

        # we get 1 at the last axis because we are applying score to self.V

        # the shape of the tensor before applying self.V is (batch_size, max_length, units)

        #     score = self.V(tf.nn.tanh(

        #         self.W1(query_with_time_axis) + self.W2(values)))

        score = tf.matmul(values, query_with_time_axis, transpose_b=True)

        if(self.use_scale):

            score*=self.scale





        # attention_weights shape == (batch_size, max_length, 1)

        attention_weights = tf.nn.softmax(score, axis=1)



        # context_vector shape after sum == (batch_size, hidden_size)

        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)



        return context_vector
latent_dim = 256
encoder_inp = Input(shape = (max_en_len,))

# initial_state = Input(shape = (latent_dim,))#

encoder_em = Embedding(vocab_len_en, 200,weights=[embedding_weights],trainable=False)(encoder_inp)

encoder_lstm = LSTM(latent_dim, return_state=True,return_sequences=True)

encoder_out,enc_h,enc_c = encoder_lstm(encoder_em)

#                                        ,initial_state=[initial_state,initial_state])#



encoded_state = [encoder_out,enc_h,enc_c]



# model_enc = Model([encoder_inp,initial_state],encoded_state)

model_enc = Model(encoder_inp,encoded_state)





decoder_inp = Input(shape = (1,),dtype='int64')

decoder_em_layer = Embedding(vocab_len_ta, 200)

decoder_em = decoder_em_layer(decoder_inp)



decoder_enc_hidden_outs = Input(shape=(max_en_len,latent_dim,))

decoder_dec_hidden_out = Input(shape=(latent_dim,))

decoder_dec_h = Input(shape=(latent_dim,))

decoder_dec_c = Input(shape=(latent_dim,))



decoder_attention = LuongAttention(use_scale=True)

context_vector = decoder_attention([decoder_dec_hidden_out,decoder_enc_hidden_outs])

context_vector = RepeatVector(1)(context_vector)



comb_input = Concatenate(axis=-1)( [context_vector,decoder_em])

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_state_inputs = [decoder_dec_h,decoder_dec_c]

decoder_out,dec_h,dec_c = decoder_lstm(comb_input, initial_state  = decoder_state_inputs)

decoder_dense = Dense(vocab_len_ta, activation = 'softmax')

decoder_out = decoder_dense(decoder_out)



# model = Model([encoder_inp, decoder_inp], decoder_out)



# # model.compile(optimizer='rmsprop', loss=sparse_loss,target_tensors=[decoder_target])

# # model.compile(optimizer='Adam', loss=)



# print(model.summary())

model_dec = Model([decoder_inp,decoder_enc_hidden_outs,decoder_dec_hidden_out,decoder_dec_h,decoder_dec_c],[decoder_out,dec_h,dec_c])



print(model_enc.summary())

print(model_dec.summary())
plot_model(model_enc,show_shapes=True)
plot_model(model_dec,show_shapes=True)
start_token_t
optimizer = Adam()
@tf.function

def train_step(inp_enc, inp_dec,targ, enc_hidden):

    loss = 0

    dec_h_out = tf.zeros(shape=(BATCH_SIZE,latent_dim))



    with tf.GradientTape() as tape:

#         print("DE",inp.shape,enc_hidden)

        enc_outs,enc_h,enc_c= model_enc(inp_enc, enc_hidden)

        dec_states = [enc_h,enc_c]

#         dec_input = tf.expand_dims([start_token_t] * BATCH_SIZE, 1)

        dec_input = inp_dec[:,0]



        # Teacher forcing - feeding the target as the next input

        for t in range(1, inp_dec.shape[1]):

#             print(inp_dec.shape[1],t)

            # passing enc_output to the decoder

            predictions, dec_h,dec_c = model_dec([dec_input,enc_outs,dec_h_out]+dec_states)

            dec_states = [dec_h,dec_c]

            loss += categorical_crossentropy(targ[:, t:t+1,:], predictions)



            # using teacher forcing

            dec_input = tf.expand_dims(inp_dec[:, t], 1)

#         print("ones")

        batch_loss = (loss / int(inp_dec.shape[1]))

        variables = model_enc.trainable_variables + model_dec.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))



        return batch_loss
EPOCHS = 10

BATCH_SIZE=2

steps_per_epoch = len(train_en_t)//BATCH_SIZE

# steps_per_epoch = 5





for epoch in tqdm(range(EPOCHS)):



    enc_hidden = tf.zeros(( BATCH_SIZE, latent_dim))

    train_machine = generator(BATCH_SIZE)

    total_loss = 0



    for (batch,_) in enumerate(range(steps_per_epoch)):

        input_batch,target = next(train_machine)

        inp_enc,inp_dec = input_batch

        

#         print("batch",batch)

        batch_loss = train_step(inp_enc, inp_dec, target,enc_hidden)

        total_loss += batch_loss



    if batch % 100 == 0:

        print(f"Epoch {epoch+1} Batch {batch} Loss {batch_loss}")#format(epoch + 1,

#                                                    batch,

#                                                    batch_loss.numpy()))



    print(f'Epoch {epoch+1} Loss {(total_loss.numpy()/steps_per_epoch)}')

y_pred_all = list()

for (i,en_t) in enumerate(tqdm(test_en_t)):

    X1_test = np.array(pad_sequences([en_t],max_en_len,padding='post'))

    dec_h_out = tf.zeros(( 1, latent_dim))

    enc_outs_pred,enc_h_pred,enc_c_pred = model_enc(X1_test, enc_hidden)

#     curr_state = encoder_model.predict(X1_test)

    curr_state = [enc_h_pred,enc_c_pred]

    inp_word = np.array([[start_token_t]])

    y_pred_ix = []

    while True:

        pred_word, dec_h,dec_c = model_dec([inp_word,enc_outs_pred,dec_h_out]+curr_state)

#         pred_word,h,c = decoder_model.predict([inp_word]+curr_state)

        pred_index = np.argmax(pred_word, axis = -1)[0][0]

        if(pred_index == end_token_t or len(y_pred_ix)>max_ta_len):

            break

        y_pred_ix.append(pred_index)

        inp_word = np.array([[pred_index]])

        curr_state = [dec_h,dec_c]

    y_pred_all.append(y_pred_ix)

y_pred_all_txt = tkzr_ta.sequences_to_texts(y_pred_all)





pd.DataFrame({'x':test_en, 'y':y_pred_all_txt, 'y_actual':test_ta}).to_csv('/kaggle/working/final_test_res-attention.csv')



from nltk.translate.bleu_score import sentence_bleu



df = pd.DataFrame({'x':test_en, 'y_pred':y_pred_all_txt, 'y_actual':test_ta})

df.to_csv('/kaggle/working/final_test_res-attention.csv')

# df['BLEU'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred']), axis=1)

df['BLEUone'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(1, 0, 0, 0)), axis=1)

df['BLEUtwo'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.5, 0.5, 0, 0)), axis=1)

df['BLEUthr'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.33, 0.33, 0.33, 0)), axis=1)

df['BLEUfou'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.25, 0.25, 0.25, 0.25)), axis=1)

df.to_csv('/kaggle/working/final_test_res-attention-BLEU.csv')



# (5, 46)

# (5, 28)

# (5, 28, 19254)

def generator_transformer(batch_size=1):

    num = 0

    while(True):

        encoder_input_batch,decoder_input_batch,decoder_output_batch = list(),list(),list()

        for (i,(en_t,ta_t)) in enumerate(zip(train_en_t,train_ta_t)):

            encoder_input_batch.append(en_t)

            decoder_input_batch.append(ta_t)

            num+=1

            if(num==batch_size):

                encoder_input_batch = pad_sequences(encoder_input_batch,max_en_len,padding='post')

                decoder_input_batch = pad_sequences(decoder_input_batch,max_ta_len,padding='post')

                for ta_item in decoder_input_batch:

                    decoder_output_batch.append(to_categorical(ta_item,num_classes=vocab_len_ta))

                yield([encoder_input_batch,decoder_input_batch[:,:-1]],np.asarray(decoder_output_batch)[:,1:,:])

                encoder_input_batch,decoder_input_batch,decoder_output_batch = list(),list(),list()

                num=0
gen = generator_transformer()

t5 = next(gen)

print(t5[0][0].shape)

print(t5[0][1].shape)

print(t5[1].shape)
from keras import Sequential

from keras.layers import Add,Input,Lambda, Dense, Dropout, Conv2D,AveragePooling2D,Flatten, InputLayer, BatchNormalization,LSTM,Embedding,Concatenate,RepeatVector,Attention,LayerNormalization

from keras.layers.merge import add

from keras.models import Model

from keras.utils import plot_model

import tensorflow as tf

from keras.losses import sparse_categorical_crossentropy,categorical_crossentropy

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam
import sys

import os

sys.path.append(os.path.abspath("/kaggle/input/kerastransformerfour/"))

# import kaggle.input.keras-transformer.keras_transformer

# __import__("/kaggle/input/keras-transformer/keras_transformer")

from keras_transformer.attention import MultiHeadAttention, MultiHeadSelfAttention

from keras_transformer.transformer import TransformerTransition, TransformerBlock

from keras_transformer.position import AddPositionalEncoding

import keras_transformer
num_layers = 2

num_heads = 4

droput_rate=0



trans_enc_inp = Input(shape=(max_en_len,))

trans_enc_em = Embedding(vocab_len_en, 200,weights=[embedding_weights],trainable=False)(trans_enc_inp)

enc_output = AddPositionalEncoding()(trans_enc_em)





# output = trans_enc_pos_inp

for _ in range(num_layers):

    mhsa_out = MultiHeadSelfAttention(num_heads,use_masking=True)(enc_output)

    add_out = Add()([enc_output,Dropout(droput_rate)(mhsa_out)])

    norm_out = LayerNormalization()(add_out)

    transition_out = TransformerTransition(activation='relu')(norm_out)

    add_out2 = Add()([norm_out,Dropout(droput_rate)(transition_out)])

    norm_out2 = LayerNormalization()(add_out2)

    enc_output = LayerNormalization()(norm_out2)

    

# EncoderStack = Model(trans_enc_inp,enc_output)



trans_dec_inp = Input(shape = (max_ta_len-1,),dtype='int64')

trans_dec_em = Embedding(vocab_len_ta, 200)(trans_dec_inp)

dec_output = AddPositionalEncoding()(trans_dec_em)



for i in range(num_layers):

    mhsa_dec_out = MultiHeadSelfAttention(num_heads, use_masking = True)(dec_output)

    add_out_dec = Add()([dec_output,Dropout(droput_rate)(mhsa_dec_out)])

    norm_out_dec = LayerNormalization()(add_out_dec)

    corss_att_inp = [enc_output, norm_out_dec]

    mha_dec_out = MultiHeadAttention(num_heads, use_masking = False)(corss_att_inp)

    add_out2_dec = Add()([norm_out_dec,Dropout(droput_rate)(mha_dec_out)])

    norm_out2_dec = LayerNormalization()(add_out2_dec)

    transition_out_dec = TransformerTransition(activation = 'relu')(mha_dec_out)

    add_out3_dec = Add()([norm_out2_dec,Dropout(droput_rate)(transition_out_dec)])

    norm_out3_dec = LayerNormalization()(add_out3_dec)

    dec_output = LayerNormalization()(norm_out3_dec)



# DecoderStack = Model([input_dec,input_enc],dec_output)



output = Dense(vocab_len_ta,activation='softmax')(dec_output)



# trans_enc_model = Model(Encoder_inputs,layer_out)

# trans_enc_model = Model(trans_enc_inp,enc_output)

# plot_model(trans_enc_model,show_shapes=True)



transformer_model = Model([trans_enc_inp,trans_dec_inp],output)

plot_model(transformer_model,show_shapes=True)

opt = Adam(learning_rate=9*1e-5)

transformer_model.compile(optimizer=opt, loss='categorical_crossentropy')

mcp_save = ModelCheckpoint('/kaggle/working/mdl_wts-trans.hdf5', save_best_only=True, monitor='loss', mode='min')

early_stopping = EarlyStopping(monitor='loss',min_delta=0,patience=5,verbose=1, mode='auto',restore_best_weights=True)



batch_size = 8

# his = transformer_model.fit(generator_transformer(batch_size),epochs=10,verbose=1,

#           steps_per_epoch=len(train_en_t)//batch_size

# #           steps_per_epoch=21

#          ,callbacks=[mcp_save]

#          )
from matplotlib import pyplot as plt

plt.plot(his.history['loss'])

# plt.plot(his.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.savefig('/kaggle/working/Q5_model.png', bbox_inches='tight')

plt.show()
[start_token_t]
max_ta_len
y_pred_all = list()

for (i,en_t) in enumerate(tqdm(test_en_t)):

    X1_test = np.array(pad_sequences([en_t],max_en_len,padding='post'))

    inp_word = np.array([start_token_t])

    y_pred_ix = []

    i = 0

    try:

        while True:

#             print(X1_test.shape,i,inp_word,y_pred_ix)

            pred_words = transformer_model.predict([X1_test,

                                      pad_sequences([inp_word],max_ta_len-1,padding='post')])

            pred_index = np.argmax(pred_words, axis = -1)[0][i]

    #         print(pred_index)

            if(pred_index == end_token_t or i>=max_ta_len-2):

                break

            y_pred_ix.append(pred_index)

            inp_word = np.append(inp_word,pred_index)

            i+=1

    #     print(tkzr_ta.sequences_to_texts([y_pred_ix]))

        y_pred_all.append(y_pred_ix)

    except:

        y_pred_all.append([])

        print("skipped",tkzr_en.sequences_to_texts([en_t]))

y_pred_all_txt = tkzr_ta.sequences_to_texts(y_pred_all)
from nltk.translate.bleu_score import sentence_bleu



df = pd.DataFrame({'x':test_en, 'y_pred':y_pred_all_txt, 'y_actual':test_ta})

df.to_csv('/kaggle/working/final_test_res-transformer.csv')

df['BLEUone'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(1, 0, 0, 0)), axis=1)

df['BLEUtwo'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.5, 0.5, 0, 0)), axis=1)

df['BLEUthr'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.33, 0.33, 0.33, 0)), axis=1)

df['BLEUfou'] = df.apply(lambda row: sentence_bleu(row['y_actual'],row['y_pred'],weights=(0.25, 0.25, 0.25, 0.25)), axis=1)

df.to_csv('/kaggle/working/final_test_res-transformer-BLEU.csv')


