# %tensorflow_version 2.x

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = "retina"



import unicodedata

import re

import numpy as np

import pandas as pd

import io

import os

import time



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

sample = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')



train = train.dropna()
def unicode_to_ascii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s)

      if unicodedata.category(c) != 'Mn')





def preprocess_sentence(w):

    w = '<start> ' + w + ' <end>'

    return w
def create_dataset(data):

    text = train.text.apply(lambda x: preprocess_sentence(x))

    summary = train.selected_text.apply(lambda x: preprocess_sentence(x))

    test_text = test.text.apply(lambda x: preprocess_sentence(x))

    return text, summary, test_text
def max_length(tensor):

    return max(len(t) for t in tensor)
def tokenize(lang):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(

      filters=''

    )

    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer
def load_dataset(data, num_examples=None):

    text, summary, test = create_dataset(data)

    new_text = pd.concat([text,test],axis=0)

    new_text_tensor, text_tokenizer = tokenize(new_text)

    text_tensor, test_tensor = new_text_tensor[:len(text)],new_text_tensor[len(text):]

    summary_tensor, summary_tokenizer = tokenize(summary)

    return text_tensor, text_tokenizer, summary_tensor, summary_tokenizer, test_tensor
text_tensor, text_tokenizer, summary_tensor, summary_tokenizer, test_tensor = load_dataset(train)

max_length_text, max_length_summary = max_length(text_tensor), max_length(summary_tensor)
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(text_tensor, summary_tensor, test_size=0.1)
def convert(lang, tensor):

    for t in tensor:

        if t!=0:

            print ("%d ----> %s" % (t, lang.index_word[t]))



print ("Input Text; index to word mapping")

convert(text_tokenizer, input_tensor_train[1])

print ()

print ("Target Summary; index to word mapping")

convert(summary_tokenizer, target_tensor_train[1])
BUFFER_SIZE = len(input_tensor_train)

BATCH_SIZE = 64

steps_per_epoch = len(input_tensor_train)//BATCH_SIZE

embedding_dim = 256

units = 1024

vocab_inp_size = len(text_tokenizer.word_index)+1

vocab_tar_size = len(summary_tokenizer.word_index)+1



dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).cache().shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
example_input_batch, example_target_batch = next(iter(dataset))

example_input_batch.shape, example_target_batch.shape
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
attention_layer = BahdanauAttention(1)

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
optimizer = tf.keras.optimizers.RMSprop()

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



        dec_input = tf.expand_dims([summary_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)



        for t in range(1, targ.shape[1]):

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)



    batch_loss = (loss / int(targ.shape[1]))



    variables = encoder.trainable_variables + decoder.trainable_variables



    gradients = tape.gradient(loss, variables)



    optimizer.apply_gradients(zip(gradients, variables))



    return batch_loss
EPOCHS = 15



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

    attention_plot = np.zeros((max_length_summary, max_length_text))



    sentence = preprocess_sentence(sentence)

    inputs = [text_tokenizer.word_index[i] for i in sentence.lower().split()]

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],

                                                         maxlen=max_length_text,

                                                         padding='post')

    inputs = tf.convert_to_tensor(inputs)



    result = ''



    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)



    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([summary_tokenizer.word_index['<start>']], 0)



    for t in range(max_length_summary):

        predictions, dec_hidden, attention_weights = decoder(dec_input,

                                                             dec_hidden,

                                                             enc_out)



        # storing the attention weights to plot later on

        attention_weights = tf.reshape(attention_weights, (-1, ))

        attention_plot[t] = attention_weights.numpy()



        predicted_id = tf.argmax(predictions[0]).numpy()



        result += summary_tokenizer.index_word[predicted_id] + ' '



        if summary_tokenizer.index_word[predicted_id] == '<end>':

            return result, sentence, attention_plot



        # the predicted ID is fed back into the model

        dec_input = tf.expand_dims([predicted_id], 0)



    return result, sentence, attention_plot
# function for plotting the attention weights

import matplotlib.ticker as ticker

def plot_attention(attention, sentence, predicted_sentence):

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(1, 1, 1)

    ax.matshow(attention, cmap='viridis')



    fontdict = {'fontsize': 14}



    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)

    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)



    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))



    plt.show()
def summarize(sentence):

    result, sentence, attention_plot = evaluate(sentence)

    return result
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
sample['selected_text']  = test.text.apply(lambda x: summarize(x))
sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace("<end>",""))

sample.to_csv('submission.csv',index=False)

sample