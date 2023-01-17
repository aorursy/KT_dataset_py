from __future__ import absolute_import, division, print_function



# Import TensorFlow >= 1.10 and enable eager execution

import tensorflow as tf



#tf.enable_eager_execution()



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import unicodedata

import re

import numpy as np

import os

import time

import math



print(tf.__version__)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Path for the dataset file

path_to_file = os.path.join(dirname, filename)
def unicode_to_ascii(s):

    '''

    Args:

        s : UniCode file

    Returns:

        ASCII file

    

    Converts the unicode file to ascii

    '''

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')





def preprocess_sentence(w):

    '''

    Args:

        w : A single word

    Returns:

        w : Single normalize word 

    

    Convert Unicode to ASCII

    Creating a space between a word and the punctuation following it

    eg: "he is a boy." => "he is a boy ." 

    Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation

    

    Replacing everything with space except (a-z, A-Z, ا-ي ".", "?", "!", ",")

    

    Adding a start and an end token to the sentence

    

    '''

    w = unicode_to_ascii(w.lower().strip())

    

    w = re.sub(r"([?.!,¿])", r" \1 ", w)

    w = re.sub(r'[" "]+', " ", w)

    

    w = re.sub(r"[^a-zA-Z؀-ۿ?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    

    w = '<start> %s <end>'%w

    return w
def create_dataset(path, num_examples):

    '''

    Args:

        path: Path of the dataset file

        num_examples: Threshold to read a range from dataset

    Returns:

        word_pairs : All readed words from dataset as a pairs 

        

    1. Remove the accents

    2. Clean the sentences

    3. Return word pairs in the format: [ENGLISH, ARABIC]

    '''

    lines = open(path, encoding='utf-8-sig').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

    print(len(lines))

    print(len(lines[:num_examples]))

    return word_pairs
class LanguageIndex():

    ''' Creates a word -> index mapping



    This class creates a word -> index mapping 

    (e.g., "dad" -> 5) and vice-versa 

    (e.g., 5 -> "dad") for each language,

    

    Attributes:

        lang: A langauage to map to its index.

    '''

    def __init__(self, lang):

        self.lang = lang

        self.word2idx = {}

        self.idx2word = {}

        self.vocab = set()



        self.create_index()

    

    def create_index(self):

        for phrase in self.lang:

            self.vocab.update(phrase.split(' '))

    

        self.vocab = sorted(self.vocab)

    

        self.word2idx['<pad>'] = 0

        

        for index, word in enumerate(self.vocab):

            self.word2idx[word] = index + 1

    

        for word, index in self.word2idx.items():

            self.idx2word[index] = word
def max_length(tensor):

    '''

    Args: 

        tensor: Tensor of indexed words

    Return:

        The maximum size for the longest tensor

    '''

    return max(len(t) for t in tensor)





def load_dataset(path, num_examples):

    '''

    Args:

        path: Path of the dataset

        num_examples: Threshold to read a range from dataset

    Return:

        input_tensor: Tensor of the input language 

        target_tensor: Tensor for the traget to translate language

        inp_lang: Indexed input language 

        targ_lang:  Indexed traget to translate language

        max_length_inp: Maximum length of the input language 

        max_length_tar: Maximum length of the traget language

        

    Creating cleaned input, output pairs,

    Index language using the class defined above,

    Vectorize the input and target languages,

    Calculate max_length of input and output tensor, then

    Padding the input and output tensor to the maximum length

    

    '''

    pairs = create_dataset(path, num_examples)



    inp_lang = LanguageIndex(sp for en, sp in pairs)

    targ_lang = LanguageIndex(en for en, sp in pairs)

    

    

    input_tensor = [[inp_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]



    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]

    

    # Here, we'll set those to the longest sentence in the dataset

    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 

                                                                 maxlen=max_length_inp,

                                                                 padding='post')

    

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 

                                                                  maxlen=max_length_tar, 

                                                                  padding='post')

    

    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar
# Try experimenting with the size of that dataset

num_examples = 30000

# Load the dataset and return Tensor of the input, Tensor for the traget, Indexed input, Indexed traget, Max length input, Max length target

input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)
# Creating training and validation sets using an 80-20 split

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)



# Show length

len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)
BUFFER_SIZE = len(input_tensor_train)

BATCH_SIZE = 24

N_BATCH = BUFFER_SIZE//BATCH_SIZE

embedding_dim = 256

units = 300

vocab_inp_size = len(inp_lang.word2idx)

vocab_tar_size = len(targ_lang.word2idx)



dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
def gru(units):

    

    '''

    Args: 

        units: number of units

    Return:

        GRU units

    

    If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)

    the code automatically does that.

    '''

    return tf.keras.layers.GRU(units, 

                                return_sequences=True, 

                                return_state=True, 

                                recurrent_activation='sigmoid', 

                                recurrent_initializer='glorot_uniform')
class Encoder(tf.keras.Model):

    ''' Create the word embeddings, then fed as input into the main network,

    

    This class create encoder which consists of two multi-layer RNNs – an encoder 

    for the source language and a decoder for the target language.

    

    Attributes:

        vocab_size: Size of vocabulary

        embedding_dim: Embedding vector length

        enc_units:  Number of GRUs units

        batch_sz: batch size ** minimize it for low RAM

    '''

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):

        super(Encoder, self).__init__()

        self.batch_sz = batch_sz

        self.enc_units = enc_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = gru(self.enc_units)

        

    def call(self, x, hidden):

        x = self.embedding(x)

        output, state = self.gru(x, initial_state = hidden)        

        return output, state

    

    def initialize_hidden_state(self):

        return tf.zeros((self.batch_sz, self.enc_units))
class Decoder(tf.keras.Model):

    ''' Create the decoder, also needs to have access to the source information.

    

    This class create decoder which consists of decoder_cell (similar to encoder_cell), 

    a helper, and the previous encoder_state as inputs.

    

    Attributes:

        vocab_size: Size of vocabulary

        embedding_dim: Embedding vector length

        enc_units:  Number of GRUs units

        batch_sz: batch size ** minimize it for low RAM

    '''

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):

        super(Decoder, self).__init__()

        self.batch_sz = batch_sz

        self.dec_units = dec_units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = gru(self.dec_units)

        self.fc = tf.keras.layers.Dense(vocab_size)

        

        # used for attention

        self.W1 = tf.keras.layers.Dense(self.dec_units)

        self.W2 = tf.keras.layers.Dense(self.dec_units)

        self.V = tf.keras.layers.Dense(1)

        

    def call(self, x, hidden, enc_output):

        # enc_output shape == (batch_size, max_length, hidden_size)

        

        # hidden shape == (batch_size, hidden size)

        # hidden_with_time_axis shape == (batch_size, 1, hidden size)

        # we are doing this to perform addition to calculate the score

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        

        # score shape == (batch_size, max_length, 1)

        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V

        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        

        # attention_weights shape == (batch_size, max_length, 1)

        attention_weights = tf.nn.softmax(score, axis=1)

        

        # context_vector shape after sum == (batch_size, hidden_size)

        context_vector = attention_weights * enc_output

        context_vector = tf.reduce_sum(context_vector, axis=1)



        # x shape after passing through embedding == (batch_size, 1, embedding_dim)

        x = self.embedding(x)

        

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        

        # passing the concatenated vector to the GRU

        output, state = self.gru(x)

        

        # output shape == (batch_size * 1, hidden_size)

        output = tf.reshape(output, (-1, output.shape[2]))

        

        # output shape == (batch_size * 1, vocab)

        x = self.fc(output)

            

        return x, state, attention_weights

        

    def initialize_hidden_state(self):

        return tf.zeros((self.batch_sz, self.dec_units))

    

    def saved_hidden_state(self):

        return self.gru.states
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()





def loss_function(real, pred):

    mask = 1 - np.equal(real, 0)

    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask

    return tf.reduce_mean(loss_)
save_dir = '/kaggle/model_save/'

encoder_prefix = os.path.join(save_dir, "encoder_weights")

decoder_prefix = os.path.join(save_dir, "decoder_weights")
save_dir = '/kaggle/model_save/'

encoder_prefix = os.path.join(save_dir, "encoder_weights")

decoder_prefix = os.path.join(save_dir, "decoder_weights")
!pip install googledrivedownloader
from google_drive_downloader import GoogleDriveDownloader as gdd



gdd.download_file_from_google_drive(file_id='1MrID4-om-Z2wjsMuNX8MyRbLSKKT7Ajy',dest_path='/kaggle/model_save_1.zip',unzip=True)
!unzip /kaggle/model_save_1
!mkdir -p /kaggle/model_save/

!mv content/model_save/* /kaggle/model_save/

!ls -l --block-size=M /kaggle/model_save
encoder.load_weights(encoder_prefix)

decoder.load_weights(decoder_prefix)
EPOCHS = 0

best_total_loss = math.inf



for epoch in range(EPOCHS):

    start = time.time()

    

    hidden = encoder.initialize_hidden_state()

    total_loss = 0

    

    for (batch, (inp, targ)) in enumerate(dataset):

        loss = 0

        

        with tf.GradientTape() as tape:

            enc_output, enc_hidden = encoder(inp, hidden)

            

            dec_hidden = enc_hidden

            

            dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       

            

            # Teacher forcing - feeding the target as the next input

            for t in range(1, targ.shape[1]):

                # passing enc_output to the decoder

                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                

                loss += loss_function(targ[:, t], predictions)

                

                # using teacher forcing

                dec_input = tf.expand_dims(targ[:, t], 1)

        

        batch_loss = (loss / int(targ.shape[1]))

        

        total_loss += batch_loss

        

        variables = encoder.variables + decoder.variables

        

        gradients = tape.gradient(loss, variables)

        

        optimizer.apply_gradients(zip(gradients, variables))

        

        if batch % 100 == 0:

            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,

                                                         batch,

                                                         batch_loss.numpy()))

    

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))

    

    if best_total_loss > total_loss:

        

        best_total_loss = total_loss

        

        encoder.save_weights(encoder_prefix)

        decoder.save_weights(decoder_prefix)

        

        !zip -r /content/model_save.zip /content/model_save/

        !cp -r '/content/model_save.zip'  'drive/My Drive/Colab Notebooks/NMT With Attention For English to Arabic/'

        

        print('Saving weights at epoch {} with Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))

        

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):

    attention_plot = np.zeros((max_length_targ, max_length_inp))

    

    sentence = preprocess_sentence(sentence)



    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')

    inputs = tf.convert_to_tensor(inputs)

    

    result = ''



    hidden = [tf.zeros((1, units))]

    enc_out, enc_hidden = encoder(inputs, hidden)



    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)



    for t in range(max_length_targ):

        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        

        # storing the attention weigths to plot later on

        attention_weights = tf.reshape(attention_weights, (-1, ))

        attention_plot[t] = attention_weights.numpy()



        predicted_id = tf.argmax(predictions[0]).numpy()



        result += targ_lang.idx2word[predicted_id] + ' '



        if targ_lang.idx2word[predicted_id] == '<end>':

            return result, sentence, attention_plot

        

        # the predicted ID is fed back into the model

        dec_input = tf.expand_dims([predicted_id], 0)



    return result, sentence, attention_plot
# function for plotting the attention weights

def plot_attention(attention, sentence, predicted_sentence):

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(1, 1, 1)

    ax.matshow(attention, cmap='viridis')

    

    fontdict = {'fontsize': 14}

    

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)

    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)



    plt.show()
def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):

    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

        

    print('Input: {}'.format(sentence))

    print('Predicted translation: {}'.format(result))

    

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]

    plot_attention(attention_plot, sentence.split(' '), result.split(' '))
translate(u'جرب', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
translate(u'كيف تفعل ذلك بشكل افضل', encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)