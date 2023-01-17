# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/working/checkPointFolder'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
song_lyrics = open('../input//poetry/Kanye_West.txt','rb').read().decode(encoding='utf-8')

#adding all vocabularies

vocab = sorted(set(song_lyrics))

char2idx = {u: i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)


text_as_integer = np.array([char2idx[text] for text in song_lyrics])

char_data_set = tf.data.Dataset.from_tensor_slices(text_as_integer)
# sequence_length is the size of the training sequence that is fed into Neural network at an instance

sequence_length = 256

examples_per_epoch = len(song_lyrics) // sequence_length
sequence_text = char_data_set.batch(sequence_length+1, drop_remainder=True)



for item in sequence_text.take(5):

    print(repr(''.join(idx2char[item.numpy()])))
# divide the sequence into input and output sampel

def split_input(chunk):

    inp = chunk[:-1]

    out = chunk[1:]

    return inp, out

dataset = sequence_text.map(split_input)


for j, k in dataset.take(1):

    print(repr(''.join(idx2char[j.numpy()])))

    print(repr(''.join(idx2char[k.numpy()])))

    
BATCH_SIZE= 64

dataset= dataset.shuffle(100000).batch(BATCH_SIZE,drop_remainder=True)
def loss(labels, logits):

    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
#create the checkpoint for callback function

filepath = os.path.join('checkPointFolder/', "ckpt{epoch}")

checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True)
vocabulary_size = len(vocab)

embedding_dim = 512

rnn_units = 1024



def createModel(batchSize):    

    model = keras.Sequential()

    model.add(keras.layers.Embedding(vocabulary_size, embedding_dim, batch_input_shape=[batchSize, None]))

    model.add(keras.layers.GRU(rnn_units, return_sequences=True,stateful=True, recurrent_initializer='glorot_uniform'))

    model.add(keras.layers.Dense(vocabulary_size))

    return model

model = createModel(BATCH_SIZE)

model.compile(optimizer="adam", loss=loss)

model.fit(dataset,epochs=40,callbacks=[checkpoint_callback])
#load the model and build model

predictionModel = createModel(1)

predictionModel.load_weights(tf.train.latest_checkpoint("/kaggle/working/checkPointFolder"))

predictionModel.build(tf.TensorShape([1,None]))

predictionModel.summary()
def generate_text(model_hat, start_string):

    # Evaluation step (generating text using the learned model)



    # Number of characters to generate

    num_generate = 1000



    # Converting our start string to numbers (vectorizing)

    input_eval = [char2idx[s] for s in start_string]

    input_eval = tf.expand_dims(input_eval, 0)



    # Empty string to store our results

    text_generated = []



    # Low temperatures results in more predictable text.

    # Higher temperatures results in more surprising text.

    # Experiment to find the best setting.

    temperature = 1.0



    # Here batch size == 1

    model_hat.reset_states()

    for i in range(num_generate):

        predictions = model_hat(input_eval)

        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)



        # using a categorical distribution to predict the word returned by the model

        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()



        # We pass the predicted word as the next input to the model

        # along with the previous hidden state

        input_eval = tf.expand_dims([predicted_id], 0)



        text_generated.append(idx2char[predicted_id])



    return (start_string + ''.join(text_generated))

out = generate_text(predictionModel, "Lamborgini")

print(out)