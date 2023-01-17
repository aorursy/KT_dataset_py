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

import tensorflow as tf

from tensorflow import keras



import re

from nltk.corpus import stopwords

from tqdm import tqdm
data = pd.read_json('../input/arxiv-papers-2010-2020/arXiv_title_abstract_20200809_2011_2020.json',encoding='utf-8')

data.head()
data.shape
#val_df = df.sample(frac=0.1, random_state=1007)

#train_df = df.drop(val_df.index)

#test_df = train_df.sample(frac=0.1, random_state=1007)

#train_df.drop(test_df.index, inplace=True)
# your code here 
#Drop rows with duplicate values in the text column

data.drop_duplicates(subset=["abstract"],inplace=True)

#Drop rows with null values in the text variable

data.dropna(inplace=True)

data.reset_index(drop=True,inplace=True)

# we are using the text variable as the summary and the ctext as the source text

print('Drop null and duplicates, Total rows:', len(data))

# Rename the columns

#data.columns = ['title','abstract']

data.head()
data.shape
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
data['abstract'][3]
data['title'][3]




stop_words = stopwords.words('english')



def preprocess(text):

    ''' Function to clean the input text: convert to lowercase, expand the contractions, remove the stopwords,

        remove punctuations

    '''



    text = text.lower() # lowercase

    text = text.split() # convert have'nt -> have not

    

    for i in range(len(text)): # For every token or word in the text

        word = text[i]

        if word in contraction_mapping:

            text[i] = contraction_mapping[word] # Expand the contractions

            

    text = " ".join(text) # Rejoin the word to a sentence

    text = text.split() # Split the text into words

    newtext = []

    for word in text: # For every token or word in the text

        if word not in stop_words:

            newtext.append(word) #Include only the non stopwords

    text = " ".join(newtext)

    text = text.replace("'s",'') # Expand contractions, convert your's -> your

    #text = re.sub(r'\(.*\)','',text) # remove (words)

    text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations

    text = re.sub(r'\.',' . ',text)

    text= re.sub(" +", " ", text)

    return text
data['abstract'] = data['abstract'].apply(lambda x:preprocess(x))

data['title'] = data['title'].apply(lambda x:preprocess(x))
data['abstract'][3]

data.title='<start> '+data.title+' <end>'
data.title.head()
title_max_len=0

for i in data.title:

    if title_max_len<len(i):

        title_max_len=len(i)

print(title_max_len)  
abstract_lengths=[]

title_lengths=[]

for i in tqdm(range(len(data.abstract))):

    abstract_lengths.append(len(data.abstract[i]))

    title_lengths.append(len(data.title[i]))
import seaborn as sns

sns.distplot(abstract_lengths)
sns.distplot(title_lengths)
train = data.sample(frac=0.8, random_state=1007)

test = data.drop(train.index)

train.shape
test.shape
train_x = train['abstract']

train_y = train['title']



test_x = test['abstract']

test_y = test['title']
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token="<OOV>", filters='')

tokenizer1 = Tokenizer(oov_token="<OOV>", filters='')

tokenizer.fit_on_texts(train_x)

tokenizer1.fit_on_texts(train_y)

word_index = tokenizer.word_index

print(len(word_index))
word_index1 = tokenizer1.word_index

print(len(word_index1))
x_sequences_train = tokenizer.texts_to_sequences(train_x)

x_padded_train = pad_sequences(x_sequences_train,maxlen=150,padding = 'post')



print(x_padded_train.shape)
y_sequences_train = tokenizer1.texts_to_sequences(train_y)

y_padded_train = pad_sequences(y_sequences_train,maxlen=10,padding = 'post')



print(y_padded_train.shape)
x_sequences_test = tokenizer1.texts_to_sequences(test_x)

x_padded_test = pad_sequences(x_sequences_test,maxlen=150,padding = 'post')



print(x_padded_test.shape)
y_sequences_test = tokenizer.texts_to_sequences(test_y)

y_padded_test = pad_sequences(y_sequences_test,maxlen=10,padding = 'post')



print(y_padded_test.shape)
x_train=x_padded_train

y_train=y_padded_train

x_test=x_padded_test

y_test=y_padded_test
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed,GRU

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
BUFFER_SIZE = len(x_train)

BATCH_SIZE = 32

steps_per_epoch = len(x_train)//BATCH_SIZE

embedding_dim = 64

units = 128

vocab_inp_size = len(tokenizer.word_index)+1

vocab_tar_size = len(tokenizer1.word_index)+1



dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, example_target_batch = next(iter(dataset))

example_input_batch.shape, example_target_batch.shape
import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):

        super(Encoder, self).__init__()

        self.batch_sz = batch_sz

        self.enc_units = enc_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.enc_units,

                                   return_sequences=True,

                                   return_state=True,

                                   recurrent_initializer='glorot_uniform')



    def call(self, x, hidden):

        x = self.embedding(x)

        output, state = self.gru(x, initial_state = hidden)

        return output, state



    def initialize_hidden_state(self):

        return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)



# sample input

sample_hidden = encoder.initialize_hidden_state()

sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))

print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):

        super(BahdanauAttention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)

        self.W2 = tf.keras.layers.Dense(units)

        self.V = tf.keras.layers.Dense(1)



    def call(self, query, values):

    

        query_with_time_axis = tf.expand_dims(query, 1)



        score = self.V(tf.nn.tanh(

        self.W1(query_with_time_axis) + self.W2(values)))



        attention_weights = tf.nn.softmax(score, axis=1)



        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)



        return context_vector, attention_weights
attention_layer = BahdanauAttention(10)

attention_result, attention_weights = attention_layer(sample_hidden, sample_output)



print("Attention result shape: (batch size, units) {}".format(attention_result.shape))

print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):

        super(Decoder, self).__init__()

        self.batch_sz = batch_sz

        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.GRU(self.dec_units,

                                   return_sequences=True,

                                   return_state=True,

                                   recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)



    def call(self, x, hidden, enc_output):

    

        context_vector, attention_weights = self.attention(hidden, enc_output)



    

        x = self.embedding(x)



   

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)



    

        output, state = self.gru(x)



    

        output = tf.reshape(output, (-1, output.shape[2]))



    

        x = self.fc(output)



        return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)



sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),

                                      sample_hidden, sample_output)



print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(

    from_logits=True, reduction='none')



def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))

    loss_ = loss_object(real, pred)



    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask



    return tf.reduce_mean(loss_)
checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer,

                                 encoder=encoder,

                                 decoder=decoder)
@tf.function

def train_step(inp, targ, enc_hidden):

    loss = 0

    with tf.GradientTape() as tape:

        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden



        dec_input = tf.expand_dims([tokenizer1.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)



    batch_loss = (loss / int(targ.shape[1]))



    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

   

    optimizer.apply_gradients(zip(gradients, variables))



    return batch_loss
EPOCHS = 35

import time

for epoch in range(EPOCHS):

    start = time.time()



    enc_hidden = encoder.initialize_hidden_state()

    total_loss = 0



    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):

        batch_loss = train_step(inp, targ, enc_hidden)

        total_loss += batch_loss

        



    if batch % 100 == 0:

        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,

                                                   batch,

                                                   batch_loss.numpy()))

    if (epoch + 1) % 2 == 0:

        checkpoint.save(file_prefix = checkpoint_prefix)



    print('Epoch {} Loss {:.4f}'.format(epoch + 1,

                                      total_loss / steps_per_epoch))

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
def evaluate(sentence):

    #attention_plot = np.zeros((max_length_targ, max_length_inp))



    #sentence = preprocess_sentence(sentence)



    inputs = [tokenizer.word_index[i] for i in sentence.split()]

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],

                                                         maxlen=150,

                                                         padding='post')

    inputs = tf.convert_to_tensor(inputs)



    result = ''



    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)



    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([tokenizer1.word_index['<start>']], 0)



    for t in range(10):

        predictions, dec_hidden , attention_weights = decoder(dec_input,

                                                         dec_hidden,

                                                         enc_out)



    

        #attention_weights = tf.reshape(attention_weights, (-1, ))

        #attention_plot[t] = attention_weights.numpy()



        predicted_id = tf.argmax(predictions[0]).numpy()



        result += tokenizer1.index_word[predicted_id] + ' '



        if tokenizer1.index_word[predicted_id] == '<end>':

            return result, sentence

        #, attention_plot



    

        dec_input = tf.expand_dims([predicted_id], 0)



    return result, sentence

#, attention_plot
from nltk.translate.bleu_score import corpus_bleu



actual, predicted = list(), list()

def pred(sentence,title):

    result, sentence= evaluate(sentence)



    actual.append(title)

    predicted.append(result)

    

    print('Input: %s' % (sentence),'\n')

    print('Actual Title: {}'.format(title),'\n')

    print('Predicted translation: {}'.format(result),'\n')

    

    BLEU_1=corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))

    BLEU_2=corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))

    BLEU_3=corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))

    BLEU_4=corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    BLUE_AVG= (BLEU_1+ BLEU_2+ BLEU_3+ BLEU_4)/4

    

    print('BLEU-1: %f' % BLEU_1)

    print('BLEU-2: %f' % BLEU_2)

    print('BLEU-3: %f' % BLEU_3)

    print('BLEU-4: %f' % BLEU_4)

    print('Average of BLUE Scor: %f' % BLEU_4)

    

    

    print('-----------------------------------------------')
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train_x

pred(train_x[17266],train_y[17266])
test.head()
pred(test_x[7],test_y[7])
test.sample(5)
for i in [26294,1869,25556,3242,18005]:

    

      pred(test_x[i],test_y[i])