import numpy as np 

import pandas as pd
from sklearn.model_selection import train_test_split

import os
import tensorflow as tf

import time
data=pd.read_csv('../input/english-to-hindi-parallel-dataset/newdata.csv')
data.head()
data=data.drop('Unnamed: 0',axis=1)
data.head()
data.describe()
data.info()
data=data.dropna()
print(data.shape)
n=int(input())

en=data['english_sentence'].values[n]

hi=data['hindi_sentence'].values[n]

print(en)

print(hi)
import string

sc = list(set(string.punctuation))
hi
#removing special charcaters

data['english_sentence']=data['english_sentence'].apply(lambda x: x.lower())

data.columns
data['english_sentence']=data['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in sc))
data['hindi_sentence']=data['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in sc))
data['english_sentence']=data['english_sentence'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
data['hindi_sentence']=data['hindi_sentence'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
data['english_sentence']=data['english_sentence'].apply(lambda x: '<start> '+x+' <end>')
data['hindi_sentence']=data['hindi_sentence'].apply(lambda x: '<start> '+x+' <end>')
data['length_eng_sentence']=data['english_sentence'].apply(lambda x:len(x.split(" ")))
data['length_hin_sentence']=data['hindi_sentence'].apply(lambda x:len(x.split(" ")))
data.head()
data=data[data['length_eng_sentence']<=20]
data=data[data['length_hin_sentence']<=20]
data.shape
n=int(input())

en=data['english_sentence'].values[n]

hi=data['hindi_sentence'].values[n]

print(en)

print(hi)
from collections import Counter 
def tokenize(lang):
    words=[]
    for i in lang:
        words.extend(i.split())
    s=Counter(words)
    a=list(s.keys())
    b=list(s.values())
    ind=np.argsort(np.array(b))
    word_to_ind={}
    for i in range(len(ind)):
        word_to_ind[a[ind[-(i+1)]]]=i+1
    sequences=[]
    for i in lang:
        sen=[]
        for j in i.split():
            sen.append(word_to_ind[j])
        sequences.append(sen)
    pad_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,padding='post')
    
    return word_to_ind,pad_sequences
    
en_word_to_ind,en_sequences=tokenize(data['english_sentence'].values)
hin_word_to_ind,hin_sequences=tokenize(data['hindi_sentence'].values)
len(en_word_to_ind),len(hin_word_to_ind)
en_sequences.shape,hin_sequences.shape
en_sequences[0].shape
x_train, x_val, y_train, y_val = train_test_split(en_sequences,hin_sequences, test_size=0.2)


print(len(x_train), len(y_train), len(x_val), len(y_val))
BUFFER_SIZE = len(x_train)
BATCH_SIZE = 128
steps_per_epoch = len(x_train)//BATCH_SIZE
embedding_dim = 256
units = 512
vocab_inp_size = len(en_word_to_ind)+1
vocab_tar_size = len(hin_word_to_ind)+1

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
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

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):

        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))


        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
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

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
checkpoint_dir = '/kaggle/working/training_checkpoints'
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

        dec_input = tf.expand_dims([hin_word_to_ind['<start>']] * BATCH_SIZE, 1)

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

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy()))

    if (epoch + 1) % 10 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
hin_ind_to_word={}

for i in hin_word_to_ind:
    hin_ind_to_word[hin_word_to_ind[i]]=i
    
en_ind_to_word={}

for i in en_word_to_ind:
    en_ind_to_word[en_word_to_ind[i]]=i
def preprocess_sentence(sentence):
    x=sentence.lower()
    x=''.join(ch for ch in x if ch not in sc)
    x=''.join([i for i in x if not i.isdigit()])
    x='<start> '+x+' <end>'
    return x
def evaluate(sentence):
    attention_plot = np.zeros((20, 20))

    sentence = preprocess_sentence(sentence)
   
    inputs = [en_word_to_ind[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=20,padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, 512))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([hin_word_to_ind['<start>']], 0)

    for t in range(20):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        

        if hin_ind_to_word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        result += hin_ind_to_word[predicted_id] + ' '

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot
for i in range(5):
    k=int(input())
    sentence=''
    for j in range(1,len(x_val[k])-1):
        if  x_val[k][j+1]==0:
            continue
        sentence+=en_ind_to_word[x_val[k][j]]+' '
    
    pred,x,atten_plot=evaluate(sentence.strip())
    actual=''
    for j in range(1,len(y_val[k])-1):
        if  x_val[k][j+1]==0:
            continue
        
        actual+=' '+hin_ind_to_word[y_val[k][j]]
    x=' '.join([j for j in x.split()[1:-1]])       
    print("english sentence---> "+x)
    print('\n')
    print('predicted sentence--->'+pred)
    print('\n')
    print('actual sentence-->'+actual)
    print('\n')
    print('--------------------------------------')