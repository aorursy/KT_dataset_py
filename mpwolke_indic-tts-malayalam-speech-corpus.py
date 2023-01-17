#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/png;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tf.__version__
path_to_file = '../input/indic-tts-malayalam-speech-corpus/mono_female_1/mono_female/txt.done.data'
text = open(path_to_file, 'r',encoding='utf-8',

                 errors='ignore').read()
print(text[:1000])
# The unique characters in the file

vocab = sorted(set(text))

print(vocab)

len(vocab)
char_to_ind = {u:i for i, u in enumerate(vocab)}

ind_to_char = np.array(vocab)

encoded_text = np.array([char_to_ind[c] for c in text])

seq_len = 250

total_num_seq = len(text)//(seq_len+1)

total_num_seq
# Create Training Sequences

char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)



sequences = char_dataset.batch(seq_len+1, drop_remainder=True)



def create_seq_targets(seq):

    input_txt = seq[:-1]

    target_txt = seq[1:]

    return input_txt, target_txt



dataset = sequences.map(create_seq_targets)
# Batch size

batch_size = 128



# Buffer size to shuffle the dataset so it doesn't attempt to shuffle

# the entire sequence in memory. Instead, it maintains a buffer in which it shuffles elements

buffer_size = 10000



dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)



# Length of the vocabulary in chars

vocab_size = len(vocab)



# The embedding dimension

embed_dim = 64



# Number of RNN units

rnn_neurons = 2052
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU

from tensorflow.keras.losses import sparse_categorical_crossentropy
def sparse_cat_loss(y_true,y_pred):

    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):

    model = Sequential()

    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))

    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))

    # Final Dense Layer to Predict

    model.add(Dense(vocab_size))

    model.compile(optimizer='adam', loss=sparse_cat_loss) 

    return model
model = create_model(

  vocab_size = vocab_size,

  embed_dim=embed_dim,

  rnn_neurons=rnn_neurons,

  batch_size=batch_size)
model.summary()
epochs = 3 
model.fit(dataset,epochs=epochs)
model.save('txt.done.data')
from tensorflow.keras.models import load_model
model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)



model.load_weights('../input/indic-tts-malayalam-speech-corpus/mono_female_1/mono_female/txt.done.data')



model.build(tf.TensorShape([1, None]))
model.summary()
def generate_text(model, start_seed,gen_size=100,temp=1.1):

  num_generate = gen_size

  input_eval = [char_to_ind[s] for s in start_seed]

  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  temperature = temp

  model.reset_states()

  for i in range(num_generate):

      predictions = model(input_eval)

      predictions = tf.squeeze(predictions, 0)

      predictions = predictions / temperature

      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(ind_to_char[predicted_id])

  return (start_seed + ''.join(text_generated))
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRS6vNi_aBNWzmYKO31o4Ve0yBTNIvrzFJWo1K8AxkzlXYGoBWO&usqp=CAU',width=400,height=400)