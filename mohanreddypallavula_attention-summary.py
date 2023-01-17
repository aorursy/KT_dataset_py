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
data=pd.read_csv('/kaggle/input/news-summary/news_summary_more.csv')
data.head()
data1=pd.read_csv('/kaggle/input/news-summary/news_summary.csv',encoding='latin1')
data1.head()
summary=pd.DataFrame()
summary['text']=pd.concat([data1['text'],data['text']],ignore_index=True)
summary['summary']=pd.concat([data1['headlines'],data['headlines']],ignore_index=True)

summary.head()
summary.shape
data.shape
data1.shape
import re
from tqdm import tqdm
#Removes non-alphabetic characters:
def text_strip(column):
    s=[]
    for row in tqdm(column):
        
        #ORDER OF REGEX IS VERY VERY IMPORTANT!!!!!!
        
        row=re.sub("(\\t)", ' ', str(row)).lower() #remove escape charecters
        row=re.sub("(\\r)", ' ', str(row)).lower() 
        row=re.sub("(\\n)", ' ', str(row)).lower()
        
        row=re.sub("(__+)", ' ', str(row)).lower()   #remove _ if it occors more than one time consecutively
        row=re.sub("(--+)", ' ', str(row)).lower()   #remove - if it occors more than one time consecutively
        row=re.sub("(~~+)", ' ', str(row)).lower()   #remove ~ if it occors more than one time consecutively
        row=re.sub("(\+\++)", ' ', str(row)).lower()   #remove + if it occors more than one time consecutively
        row=re.sub("(\.\.+)", ' ', str(row)).lower()   #remove . if it occors more than one time consecutively
        
        row=re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(row)).lower() #remove <>()|&©ø"',;?~*!
        
        row=re.sub("(mailto:)", ' ', str(row)).lower() #remove mailto:
        row=re.sub(r"(\\x9\d)", ' ', str(row)).lower() #remove \x9* in text
        row=re.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(row)).lower() #replace INC nums to INC_NUM
        row=re.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM', str(row)).lower() #replace CM# and CHG# to CM_NUM
        
        
        row=re.sub("(\.\s+)", ' ', str(row)).lower() #remove full stop at end of words(not between)
        row=re.sub("(\-\s+)", ' ', str(row)).lower() #remove - at end of words(not between)
        row=re.sub("(\:\s+)", ' ', str(row)).lower() #remove : at end of words(not between)
        
        row=re.sub("(\s+.\s+)", ' ', str(row)).lower() #remove any single charecters hanging between 2 spaces
        
        #Replace any url as such https://abc.xyz.net/browse/sdf-5327 ====> abc.xyz.net
        try:
            url = re.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(row))
            repl_url = url.group(3)
            row = re.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)',repl_url, str(row))
        except:
            pass 
        

        
        row = re.sub("(\s+)",' ',str(row)).lower() 
        
    
        row=re.sub("(\s+.\s+)", ' ', str(row)).lower() 

        
        
        s.append(row)
    return s   
text=text_strip(summary['text'].values)
summary=text_strip(summary['summary'].values)
text=text[:2000]
summary=summary[:2000]
for i in range(len(summary)):
    summary[i]='<start> '+summary[i]+' <end>'
train_t=text[:int(len(text)*0.8)]
test_t=text[int(len(text)*0.8):]
train_s=summary[:int(len(text)*0.8)]
test_s=summary[int(len(text)*0.8):]
summary_max_len=0
for i in summary:
    if summary_max_len<len(i):
        summary_max_len=len(i)
print(summary_max_len)        
text_lengths=[]
summary_lengths=[]
for i in tqdm(range(len(text))):
    text_lengths.append(len(text[i]))
    summary_lengths.append(len(summary[i]))

import seaborn as sns
sns.distplot(text_lengths)
sns.distplot(summary_lengths)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token="<OOV>", filters='')
tokenizer1 = Tokenizer(oov_token="<OOV>", filters='')
tokenizer.fit_on_texts(train_t)
tokenizer1.fit_on_texts(train_s)
word_index = tokenizer.word_index
print(len(word_index))
word_index1 = tokenizer1.word_index
print(len(word_index1))
text_sequences_tr = tokenizer.texts_to_sequences(train_t)
text_padded_tr = pad_sequences(text_sequences_tr,maxlen=150,padding = 'post')

print(text_padded_tr.shape)
summary_sequences_tr = tokenizer1.texts_to_sequences(train_s)
summary_padded_tr = pad_sequences(summary_sequences_tr,maxlen=10,padding = 'post')

print(summary_padded_tr.shape)
text_sequences_te = tokenizer1.texts_to_sequences(test_t)
text_padded_te = pad_sequences(text_sequences_te,maxlen=150,padding = 'post')

print(text_padded_te.shape)
summary_sequences_te = tokenizer.texts_to_sequences(test_s)
summary_padded_te = pad_sequences(summary_sequences_te,maxlen=10,padding = 'post')

print(summary_padded_te.shape)
x_train=text_padded_tr
y_train=summary_padded_tr
x_test=text_padded_te
y_test=summary_padded_te
x_train.shape
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
EPOCHS = 10
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

    inputs = [tokenizer.word_index[i] for i in sentence.split(' ')]
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
def translate(sentence):
    result, sentence= evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
translate(text[0])
text[0]
summary[0]