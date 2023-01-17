# os.mkdir('/kaggle/working/Model_Backup/')
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
import numpy as np  
import pandas as pd 
# %tensorflow_version 1.15
import tensorflow as tf
import re           
import os
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
import nltk
import os
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from attention import AttentionLayer
import matplotlib.pyplot as plt
import seaborn as sns
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# import tensorflow as tf
# import os
# from tensorflow.python.keras.layers import Layer
# from tensorflow.python.keras import backend as K


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
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            if verbose:
                print('Ws+Uh>', Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            assert_msg = "States must be an iterable. Got {} of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

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
raw_dataset = pd.read_csv("/kaggle/input/sample/cnn_raw.csv")
raw_dataset.head()
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}
def contraction(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''
    text = text.lower()
    text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) 
    return text
# nltk.download('stopwords')
# STOP_WORDS = set(stopwords.words('english'))
# from nltk.corpus import stopwords

# stop = stopwords.words('english')
# print(stop)
# def contraction(text):
#     '''Clean text by removing unnecessary characters and altering the format of words.'''
#     text = text.lower()
#     text = re.sub(r"i'm", "i am", text)
#     text = re.sub(r"he's", "he is", text)
#     text = re.sub(r"she's", "she is", text)
#     text = re.sub(r"it's", "it is", text)
#     text = re.sub(r"that's", "that is", text)
#     text = re.sub(r"what's", "that is", text)
#     text = re.sub(r"where's", "where is", text)
#     text = re.sub(r"how's", "how is", text)
#     text = re.sub(r"\'ll", " will", text)
#     text = re.sub(r"\'ve", " have", text)
#     text = re.sub(r"\'re", " are", text)
#     text = re.sub(r"\'d", " would", text)
#     text = re.sub(r"\'re", " are", text)
#     text = re.sub(r"won't", "will not", text)
#     text = re.sub(r"can't", "cannot", text)
#     text = re.sub(r"n't", " not", text)
#     text = re.sub(r"n'", "ng", text)
#     text = re.sub(r"'bout", "about", text)
#     text = re.sub(r"'til", "until", text)
#     text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text) 
#     return text
import re
from tqdm import tqdm
Cleaned_Stories = []
# tqdm is for printing the status bar
for sentence in tqdm(raw_dataset['Stories'].astype(str)):
    sent = contraction(sentence)
#     sent = sent.apply(lambda x: [item for item in x if item not in STOP_WORDS])
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", sent)
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent=re.sub('<[^>]*>', '', sent)
    Cleaned_Stories.append(sent.lower().strip())
Cleaned_Stories1 = Cleaned_Stories
# Cleaned_Stories1[0]
# from nltk.tokenize import word_tokenize
# tokenized_article = []

# text = "Nick likes to play football, however he is not too fond of tennis."
# for article in Cleaned_Stories1:
#     text_tokens = word_tokenize(article)
#     tokenized_article.append(text_tokens)

# text_tokens

# tokenized_article[0]
# tokenized_article_without_sw = []
# for ta in tokenized_article:
#     tokens_without_sw = [mot for mot in ta if not mot in stop]
#     tokenized_article_without_sw.append(tokens_without_sw)
# print(tokens_without_sw)

# Cleaned_Stories2 = []
# for taws in tokenized_article_without_sw:
#     filtered_sentence = (" ").join(taws)
#     Cleaned_Stories2.append(filtered_sentence)
# print(filtered_sentence)
Cleaned_Summary = []
for sentence in tqdm(raw_dataset['Summary'].astype(str)):
    sent = contraction(sentence)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", sent)
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent=re.sub('<[^>]*>', '', sent)
    Cleaned_Summary.append(sent.lower().strip())
Cleaned_Summary1 = Cleaned_Summary
# tokenized_summary = []

# text = "Nick likes to play football, however he is not too fond of tennis."
# for summary in Cleaned_Summary1:
#     text_tokens = word_tokenize(summary)
#     tokenized_summary.append(text_tokens)
# tokenized_summary_without_sw = []
# for ts in tokenized_summary:
#     tokens_summary_without_sw = [mot for mot in ts if not mot in stop]
#     tokenized_summary_without_sw.append(tokens_summary_without_sw)
# Cleaned_Summary2 = []
# for tsws in tokenized_summary_without_sw:
#     filtered_sentence = (" ").join(tsws)
#     Cleaned_Summary2.append(filtered_sentence)
raw_dataset['Cleaned_Stories'] = Cleaned_Stories1
raw_dataset['Cleaned_Summary'] = Cleaned_Summary1
raw_dataset_new = raw_dataset
raw_dataset_new['Cleaned_Summary'] = raw_dataset_new['Cleaned_Summary'].apply(lambda x : 'sostok '+ x + ' eostok')
# for i in range(3):
#     print("Story:",raw_dataset_new['Cleaned_Stories'][i])
#     print("Summary:",raw_dataset_new['Cleaned_Summary'][i])
#     print("\n")
#Calculating word count of text and getting percentile values
# raw_dataset_new['word_count_text'] = raw_dataset_new['Cleaned_Stories'].apply(lambda x: len(str(x).split()))
# for i in range(0,100,10):
#     var = raw_dataset_new['word_count_text'].values
#     var = np.sort(var, axis = None)
#     print("{} percentile value is {}".format(i, var[int(len(var)*(float(i)/100))]))
# print("100 percentile value is ", var[-1])
#Looking further till 99th percentile
# for i in range(90,100):
#     var = raw_dataset_new['word_count_text'].values
#     var = np.sort(var, axis = None)
#     print("{} percentile value is {}".format(i, var[int(len(var)*(float(i)/100))]))
# print("100 percentile value is ", var[-1]) 
#Calculating word count of text and getting percentile values
# raw_dataset_new['word_count_text'] = raw_dataset_new['Cleaned_Summary'].apply(lambda x: len(str(x).split()))
# for i in range(0,100,10):
#     var = raw_dataset_new['word_count_text'].values
#     var = np.sort(var, axis = None)
#     print("{} percentile value is {}".format(i, var[int(len(var)*(float(i)/100))]))
# print("100 percentile value is ", var[-1])
#Looking further till 99th percentile
# for i in range(90,100):
#     var = raw_dataset_new['word_count_text'].values
#     var = np.sort(var, axis = None)
#     print("{} percentile value is {}".format(i, var[int(len(var)*(float(i)/100))]))
# print("100 percentile value is ", var[-1])
max_story_len = 150
max_summary_len = 50
from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(np.array(raw_dataset_new['Cleaned_Stories']), np.array(raw_dataset_new['Cleaned_Summary']),
                                            test_size = 0.1,
                                            random_state = 0,
                                            shuffle = True)
#Loading our Glove Model 
embeddings_index = dict()
f = open('/kaggle/input/glove6b100d/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

#Calculating Percentage of words from train text present in Word2vec model
words_source_train = []
for i in x_tr :
    words_source_train.extend(i.split(' '))
## Find the total number of words in the Train data of Essays.
print("all the words in the corpus", len(words_source_train))
## Find the unique words in this set of words
words_source_train = set(words_source_train)
print("the unique words in the corpus", len(words_source_train))
## Find the words present in both Glove Vectors as well as our corpus.
inter_words = set(embeddings_index.keys()).intersection(words_source_train)
print("The number of words that are present in both glove vectors and our corpus are {} which \
is nearly {}% ".format(len(inter_words), np.round((float(len(inter_words))/len(words_source_train))
*100)))
words_corpus_source_train = {}
words_glove = set(embeddings_index.keys())
for i in words_source_train:
    if i in words_glove:
        words_corpus_source_train[i] = embeddings_index[i]
print("word 2 vec length", len(words_corpus_source_train))
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
import pickle
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))
#convert text sequences into integer sequences
x_tr    =   x_tokenizer.texts_to_sequences(x_tr )
x_val   =   x_tokenizer.texts_to_sequences(x_val)
#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr,  maxlen=max_story_len, padding='post') 
x_val   =   pad_sequences(x_val, maxlen=max_story_len, padding='post')
x_voc_size   =  len(x_tokenizer.word_index) +1
print(x_voc_size)

#preparing a tokenizer for summary on training data 
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))
#convert summary sequences into integer sequences
y_tr    =   y_tokenizer.texts_to_sequences(y_tr) 
y_val   =   y_tokenizer.texts_to_sequences(y_val) 
#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr, maxlen=max_summary_len, padding='post')
y_val   =   pad_sequences(y_val, maxlen=max_summary_len, padding='post')
y_voc_size  =   len(y_tokenizer.word_index) +1
print(y_voc_size)
word_index = x_tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
import pickle

x_tok = open('/kaggle/working/x_tok.pickle', 'wb')
pickle.dump(x_tokenizer, x_tok)
x_tok.close()

y_tok = open('/kaggle/working/y_tok.pickle', 'wb')
pickle.dump(y_tokenizer, y_tok)
y_tok.close()
# from tensorflow.python.keras import backend as K
# from keras import backend as K 
K.clear_session() 
latent_dim = 300 

# Encoder 
encoder_inputs = Input(shape=(max_story_len,)) 
enc_emb = Embedding(x_voc_size,100,input_length=max_story_len,weights=[embedding_matrix],trainable=False)(encoder_inputs) 

#LSTM 1 
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4) 
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb) 

#LSTM 2 
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

#LSTM 3 
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4) 
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

# Set up the decoder. 
decoder_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(x_voc_size,100,input_length=max_story_len,weights=[embedding_matrix],trainable=False,) 

dec_emb = dec_emb_layer(decoder_inputs) 

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.25) 
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c]) 

#Attention Layer
attn_layer = AttentionLayer(name="attention_layer")
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 

# Concat attention output and decoder LSTM output 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary()
# model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss='sparse_categorical_crossentropy')
model.compile(optimizer="rmsprop", loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)
checkpoint = ModelCheckpoint('/kaggle/working/bestmodel_weights_cnn_new.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
history = model.fit([x_tr, y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:,1:],
                    epochs = 50,
                    callbacks = [es, checkpoint],
                    batch_size = 128,
                    validation_data = ([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:,1:]))
model.save('/kaggle/working/bestmodel_weights_cnn_news_1.h5')
from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# import pickle

x_tok1 = open('/kaggle/working/x_tok1.pickle', 'wb')
pickle.dump(x_tokenizer, x_tok1)
x_tok1.close()

y_tok1 = open('/kaggle/working/y_tok1.pickle', 'wb')
pickle.dump(y_tokenizer, y_tok1)
y_tok1.close()
reverse_target_word_index=y_tokenizer.index_word 
reverse_source_word_index=x_tokenizer.index_word 
target_word_index=y_tokenizer.word_index
x_tok2 = open('/kaggle/working/x_tok2.pickle', 'wb')
pickle.dump(x_tokenizer, x_tok2)
x_tok2.close()

y_tok2 = open('/kaggle/working/y_tok2.pickle', 'wb')
pickle.dump(y_tokenizer, y_tok2)
y_tok2.close()
# encoder inference
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])
encoder_model.save('/kaggle/working/encoder_model.h5')


# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_story_len,latent_dim))

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
decoder_model.save('/kaggle/working/decoder_model.h5')
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
for i in range(100):
    print("Story:",seq2text(x_val[i]))
    print("Original summary:",seq2summary(y_val[i]))
    print("Predicted summary:",decode_sequence(x_val[i].reshape(1,max_story_len)))
    print("\n")
#liste des résumés de références
goldstandard = []
#liste des résumés générés
summ = []

#on en met 100 dans 2 listes
for i in range(0, 100):
  goldstandard.append(seq2summary(y_tr[i]))

for i in range(0, 100):
  summ.append(decode_sequence(x_tr[i].reshape(1, max_story_len)))
#on split les phrases pour créer une liste de liste de mots
GOLD = []
for i in goldstandard:
  GOLD.append(i.split())

SUMM = []
for i in summ:
  SUMM.append(i.split())

# on évalue la moyenne des scores
nltk.translate.bleu_score.corpus_bleu(GOLD, SUMM)
!pip install rouge
from rouge import Rouge 
rouge = Rouge()
# print("Story:",seq2text(x_tr[i]))
for j, k in zip(goldstandard, summ):
    print("Original summary:",j)
    print("Predicted summary:",k)
    scores = rouge.get_scores(j, k)
    print(scores)






