# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# IMPORTS

import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing import text, sequence



from tensorflow.keras.models import Model

from tensorflow.keras.layers import Layer, Input, Dense, LSTM, Embedding, Concatenate



import tensorflow.keras.backend as K



from sklearn.model_selection import train_test_split
# Set the size of the vocabulary.

# the bigger the vocabulary the better the translation but the longer the model will take to train.

VOCAB_SIZE = 20000




dataframe = pd.read_csv('/kaggle/input/europarl-parallel-corpus-19962011/english_french.csv')



dataframe = dataframe.dropna()





dataframe['English'] = dataframe['English'].map(lambda x: 'ssss ' + str(x) + ' eeee')



english_tokenizer = text.Tokenizer(num_words=VOCAB_SIZE)

english_tokenizer.fit_on_texts(dataframe['English'])



dataframe['English_sequences'] = english_tokenizer.texts_to_sequences(dataframe['English'])



dataframe['French'] = dataframe['French'].astype('str')



french_tokenizer = text.Tokenizer(num_words=VOCAB_SIZE)

french_tokenizer.fit_on_texts(dataframe['French'])



dataframe['French_sequences'] = french_tokenizer.texts_to_sequences(dataframe['French'])

#dataframe['French_sequences'] = dataframe['French_sequences'].map(lambda x: x[::-1])  #reversed sequences



dataframe
# the tokeniser dictionary does not assign a word to the key'0'  just added a placeholder word

# as when I was originally testing and making errors the model would not throw an error. 

english_tokenizer.index_word[0] = '<****>'

french_tokenizer.index_word[0] = '<****>'
class Data_generator(keras.utils.Sequence):

    

    def __init__(self, input_sequences, output_sequences, batch_size=128, shuffle=True):

        self.input_sequences = input_sequences

        self.output_sequences = output_sequences

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indexes = np.arange(len(input_sequences))

        self.on_epoch_end()

        

    def __len__(self):

        return len(self.input_sequences)//self.batch_size

    

    def __getitem__(self, index):

        batch_indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        

        input_seq = self.__get_input_sequences(batch_indexes)

        output_seq = self.__get_output_sequences(batch_indexes)



        output_seq_in = output_seq[:,:-1]

        output_seq_out = np.expand_dims(output_seq[:,1:], axis=-1)

        

        return [input_seq, output_seq_in], output_seq_out    

        

        

    def __get_input_sequences(self, indexes):

        input_seq = self.input_sequences[indexes]

        input_seq = sequence.pad_sequences(input_seq, maxlen=50, padding='pre', truncating='pre')

        return input_seq

    

    def __get_output_sequences(self, indexes):

        output_seq = self.output_sequences[indexes]

        output_seq = sequence.pad_sequences(output_seq, maxlen=50, padding='post', truncating='post')

        return output_seq

        

    

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(self.indexes)

            

    

# split the data into training and testing



train, test = train_test_split(dataframe, test_size=0.05, random_state=42)



train_generator = Data_generator(train['French_sequences'].values, train['English_sequences'].values)

test_generator = Data_generator(test['French_sequences'].values, test['English_sequences'].values)

EMBEDDING_SIZE = 128  # size of the word embedding

LATENT_SIZE = 256 # size of the LSTM latent size



# these sizes can be played around with
# this is quite compliciated

    

    

class Encoder_layer(Layer):

    

    def __init__(self, vocab_size, embedding_size, latent_size, **kwargs):

        super(Encoder_layer, self).__init__(**kwargs)

        self.embedding_layer = Embedding(vocab_size, embedding_size)

        self.alignment_lstm = LSTM(latent_size//2, return_sequences=True, return_state=True) # concentrating on alignment/attention  tried half size

        self.encoder_lstm =  LSTM(latent_size, return_sequences=True, return_state=True)  # concnetrating on decoding the sentence

        

           

    def call(self, inputs):

        embedding_out = self.embedding_layer(inputs)

        alignment_seq, alignment_h, alignment_c = self.alignment_lstm(embedding_out)

        encoder_seq, encoder_h, encoder_c = self.encoder_lstm(embedding_out)

        

        return [alignment_seq, alignment_h, alignment_c, encoder_seq, encoder_h, encoder_c]

    

    

class Decoder_layer(Layer):

    

    def __init__(self, vocab_size, embedding_size, latent_size, **kwargs):

        super(Decoder_layer, self).__init__(**kwargs)

        

        self.embedding_layer = Embedding(vocab_size, embedding_size)

        self.alignment_lstm = LSTM(latent_size//2, return_sequences=True, return_state=True)

        self.decoder_lstm =  LSTM(latent_size, return_sequences=True, return_state=True)

        

        #self.context_layer = Attention_layer(latent_size)

        

        self.concat_layer = Concatenate()

        self.dense_layer = Dense(vocab_size, activation='softmax')  

    

        

    def call(self, inputs):

        lang_in, encoder_alignment_seq, encoder_alignment_h, encoder_alignment_c, encoder_out, encoder_h, encoder_c = inputs

        

        embedding_out = self.embedding_layer(lang_in)

        alignment_seq, alignment_h, alignment_c = self.alignment_lstm(embedding_out, initial_state=[encoder_alignment_h, encoder_alignment_c])

        decoder_seq, decoder_h, decoder_c = self.decoder_lstm(embedding_out, initial_state=[encoder_h, encoder_c])

        

        scores = tf.matmul(alignment_seq, encoder_alignment_seq, transpose_b=True)

        alignment = tf.nn.softmax(scores, axis=-1)

        context = tf.matmul(alignment, encoder_out)

        concat = self.concat_layer([decoder_seq, context])

        prediction = self.dense_layer(concat)

        

        return [prediction, alignment_h, alignment_c, decoder_h, decoder_c]

        

        

        

    
# Complicated becuase I wanted to split up training from the inference, otherwise the entire network has to be run to predict each individual word. 

# this way the states are remembered so only the final portion need be run to predict each word.





original_lang_input = Input(shape=(None,))

translated_lang_input = Input(shape=(None,))



alignment_h_input = Input(shape=(LATENT_SIZE//2,))

alignment_c_input = Input(shape=(LATENT_SIZE//2,))

decoder_h_input = Input(shape=(LATENT_SIZE,))

decoder_c_input = Input(shape=(LATENT_SIZE,))



alignment_input = Input(shape=(None, LATENT_SIZE//2))

decoder_input = Input(shape=(None, LATENT_SIZE))



# layers 



encoder = Encoder_layer(VOCAB_SIZE, EMBEDDING_SIZE, LATENT_SIZE)

decoder = Decoder_layer(VOCAB_SIZE, EMBEDDING_SIZE, LATENT_SIZE)



#connecting the encoder_model

alignment_seq, alignment_h, alignment_c, encoder_seq, encoder_h, encoder_c = encoder(original_lang_input)



encoder_model = Model([original_lang_input], [alignment_seq, alignment_h, alignment_c, encoder_seq, encoder_h, encoder_c], name='encoder')

encoder_model.summary()









#connecting the decoder_model



decoder_prediction, decoder_alignment_h, decoder_alignment_c, decoder_h, decoder_c = decoder([translated_lang_input, 

                                                                              alignment_input,

                                                                              alignment_h_input, 

                                                                              alignment_c_input, 

                                                                              decoder_input, 

                                                                              decoder_h_input, 

                                                                              decoder_c_input])



decoder_model = Model(inputs=[translated_lang_input, alignment_input,alignment_h_input, alignment_c_input, decoder_input,decoder_h_input, decoder_c_input],

                     outputs=[decoder_prediction, decoder_alignment_h, decoder_alignment_c, decoder_h, decoder_c], name='decoder')



decoder_model.summary()



training_pred,_,_,_,_ = decoder_model([translated_lang_input, alignment_seq, alignment_h, alignment_c, encoder_seq, encoder_h, encoder_c])

training_model = Model(inputs=[original_lang_input, translated_lang_input], outputs=[training_pred])

training_model.summary()

# train the model

training_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')

history = training_model.fit_generator(train_generator, epochs=5, validation_data=test_generator)

import matplotlib.pyplot as plt

%matplotlib inline



try:

    plt.plot(history.history['loss'], c='r')

    plt.plot(history.history['val_loss'], c ='g')

    plt.show()

except:

    pass
def predict_sentence(dataframe, index):

    print('Original french: ' + dataframe['French'].iloc[index])

    print()

    print('Original english: '+ dataframe['English'].iloc[index])

    print()

    

    french_input = dataframe['French_sequences'].iloc[index]

    word = 'ssss'

    token = english_tokenizer.word_index[word]

    sentence = [word]

    

    count = 0

    

    alignment_seq, alignment_h, alignment_c, encoder_seq, encoder_h, encoder_c = encoder_model.predict(np.expand_dims(french_input,axis=0))

    while word != 'eeee' and count < 50:

        decoder_pred, alignment_h, alignment_c, encoder_h, encoder_c = decoder_model.predict([np.expand_dims(token, axis=0), alignment_seq, alignment_h, alignment_c, encoder_seq, encoder_h, encoder_c])

        

        token = np.argmax(decoder_pred)

        word = english_tokenizer.index_word[token]

        sentence.append(word)

        count += 1    

    print('Predicted english: ' + ' '.join(sentence[1:-1]))   

    print()

    print('-------------------------')  
# some predictions on the test data, The model has not been traind on this.



for i in range(50):

    try:

        predict_sentence(test,i)

    except:

        pass

# Some of the predictions on the training data, the model has been trained on this, therfore expect these to be better? 

for i in range(30):

    try:

        predict_sentence(train,i)

    except:

        pass