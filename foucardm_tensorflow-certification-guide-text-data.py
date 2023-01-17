from IPython.display import Image

Image("../input/text-gen-image/text_gen_image.jpeg")
import io

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from tensorflow import keras

import tensorflow_datasets as tfds
(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',

                                         split=(tfds.Split.TRAIN, tfds.Split.TEST),

                                         with_info=True, as_supervised=True)
info
info.features
encoder = info.features['text'].encoder

encoder
string_exemple = "Marry has a little lamb"

print("The string exemple is: {}".format(string_exemple))



encoded_string = encoder.encode(string_exemple)

print("The encoded string is: {}".format(encoded_string))



original_string = encoder.decode(encoded_string)

print("The decoded string is: {}".format(original_string))



print("\n")

for index in encoded_string:

    print("{} ----> {}".format(index, encoder.decode([index])))
train_data = train_data.shuffle(10000)



val_data = train_data.take(5000) 

train_data = train_data.skip(5000)
BUFFER_SIZE = 10000

BATCH_SIZE = 32

train_batches = train_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE) #The shuffle is used here to shuffle the data in the dataset

val_batches = val_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE).repeat()

test_batches = test_data.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
for i in train_batches.take(1):

    print(i)
for batch in train_batches:

    print(batch[0])

    print("\n")

    print(batch[1])

    break
iterator = train_batches.__iter__()

next_element = iterator.get_next()

one_batch_of_reviews = next_element[0]

one_batch_of_labels = next_element[1]



print(one_batch_of_reviews)

print('\n')

print(one_batch_of_labels)
decoded_review = encoder.decode(one_batch_of_reviews[0])

print(decoded_review)

print("\n")

if one_batch_of_labels[0] == 0:

    print("this person didn't liked the movie")

else:

    print("this person liked this movie")
def plot_graph(history):

    history_dict = history.history

    

    acc = history_dict['acc'] # We won't display the 3 first epochs in order to have a more precise view of the last points.

    val_acc = history_dict['val_acc']

    loss = history_dict['loss']

    val_loss = history_dict['val_loss']



    epochs = range(1, len(acc) + 1)

    

    fig = plt.figure(figsize = (18,8))

    plt.subplot2grid((1,2), (0,0))

    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()



    plt.subplot2grid((1,2), (0,1))

    plt.plot(epochs, loss, 'bo', label='Training loss')

    plt.plot(epochs, val_loss, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
from keras.models import Sequential

from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout, Flatten, Bidirectional
import tensorflow as tf



# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
from keras.models import Sequential

from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout, Flatten, Bidirectional, GlobalAveragePooling1D



embedding_dim = 16



with strategy.scope():

    model_pool = keras.Sequential([

    Embedding(encoder.vocab_size, embedding_dim),

    GlobalAveragePooling1D(),

    Dense(16, activation='relu'),

    Dropout(0.2),

    Dense(1)

])

    

model_pool.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model_pool.summary()
history_pool = model_pool.fit(train_batches, epochs=10, validation_data = test_batches)
plot_graph(history_pool)
from keras.models import Sequential

from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout, Flatten, Bidirectional



embedding_dim = 16



with strategy.scope():

    model_LSTM_stacked = Sequential([

    Embedding(input_dim = encoder.vocab_size, output_dim = embedding_dim ,mask_zero=True),

    Bidirectional(LSTM(64, return_sequences=True)),

    Bidirectional(LSTM(32)),

    Dense(32, activation='relu'),

    Dropout(0.2),

    Dense(1)

])

    

model_LSTM_stacked.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model_LSTM_stacked.summary()
history_LSTM_stacked = model_LSTM_stacked.fit(train_batches, epochs=5, validation_data = test_batches)
plot_graph(history_LSTM_stacked)
from keras.layers import GlobalMaxPool1D



embed_size = 128



with strategy.scope():

    model = Sequential()

    model.add(Embedding(input_dim = encoder.vocab_size, output_dim = embedding_dim ,mask_zero=True))

    model.add(Bidirectional(LSTM(64, return_sequences = True)))

    model.add(Bidirectional(LSTM(64, return_sequences = True)))

    model.add(GlobalMaxPool1D())

    model.add(Dense(64, activation="relu"))

    model.add(Dropout(0.1))

    model.add(Dense(1, activation="sigmoid"))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])



model.fit(train_batches, epochs=5, validation_data = test_batches)
import tensorflow as tf



import numpy as np

import os

import time
print('Loading the data...')

path_to_file = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

print('...Data loaded.')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# *open(file_path, open_mode)* if the function used to open any kind of file. You have to specify the directory and the opening mode. Here 'rb' means 'read binary'

# *read()* returns one big string containing all the characters

# You then have to decode it. We use the *utf-8* encoding since it is the most common and the data is binary and written in 8bits.
print(text[0:250]) #It looks great !
text[0:250]
print('The text object is {} of length {}'.format(type(text),len(text))) #so one really big string
vocab = sorted(set(text)) #set selects the unique parameters

print("There are {} unique characters in the whole text".format(len(vocab)))
char2idx = {u:i for i, u in enumerate(vocab)} # *enumerate* is used to count the number of loops, accessible with the variable *i*

idx2char = np.array(vocab) #Since we sorted the vocabulary and won't touch it anymore,

# the index that corresponds to the character is just the position of the character in the array, easily accessible.
print('{')

for char,_ in zip(char2idx, range(20)):

    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))

print('  ...\n}')
text_as_int = np.array([char2idx[c] for c in text]) # here we create a list that contains all the characters as numbers, and then turns it to a numpy array
print("the type of text_as_int is {} of shape {}".format(type(text_as_int),text_as_int.shape))
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))
seq_length = 100 # Number of character in one sequence of word (numbers here). For now it contains the input and the target since the size of 100.

examples_per_epoch = text_as_int.shape[0]//(seq_length+1) # The number of sequence we have, hence, the number of sample per epoch for later
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for element in char_dataset.take(13):

    print(element)
tiny_dataset = char_dataset.take(5) # *take(x)* creates a new TF dataset that contains at most x elements

for element in tiny_dataset.as_numpy_iterator():

    print(element)
# or in a fancier way:

print(list(tiny_dataset.as_numpy_iterator()))
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
five_sequences = sequences.take(5) # *take(x)* creates a new TF dataset that contains at most x elements

for element in five_sequences:

    print(repr(''.join(idx2char[element.numpy()])))

    print('\n')
def split_input_target(chunk):

    input_text = chunk[:-1]

    target_text = chunk[1:]

    return(input_text, target_text)
dataset = sequences.map(split_input_target)
for element in dataset.take(1):

    print("Each element is now a {} containing {} Tf.Tensors:\n".format(type(element),np.ndim(element)))

    print(element[0])

    print(element[1])
for input_example, target_example in  dataset.take(1):

    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))

    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):

    print("Step {:4d}".format(i))

    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))

    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
# Batch size

BATCH_SIZE = 64

BUFFER_SIZE = 10000 # This value is used to shuffle the dataset. The value has to be greater than the size of the dataset

# in order to make a good shuffle. If too low, the dataset won't be shuffled completely.



dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



dataset
# Number of different character in the text

vocab_size = len(vocab)



# The embedding dimension

# (you already know this part, as usual, we want vectors that represents well words with the same meaning)

# Here, each character will be encoded in a vector of dimension embedding_size

embedding_dim = 256



# Number of RNN units

rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

    model = tf.keras.Sequential([

        tf.keras.layers.Embedding(vocab_size, embedding_dim,

                                  batch_input_shape=[batch_size, None]),

        tf.keras.layers.GRU(rnn_units,

                            return_sequences=True,

                            stateful=True,

                            recurrent_initializer='glorot_uniform'),

        tf.keras.layers.Dense(vocab_size)

      ])

    return model
model = build_model(

    vocab_size = len(vocab),

    embedding_dim=embedding_dim,

    rnn_units=rnn_units,

    batch_size=BATCH_SIZE)
for input_example_batch, target_example_batch in dataset.take(1): #Taking the first batch of input and target of the dataset

    example_batch_predictions = model(input_example_batch) #Let's predict the input_example_batch

    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
model.summary()
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices # This is a tf.Tensor of shape=(100,1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
sampled_indices # Now it's just an array of 100 elements. This is the prediction for the first sequence of the batch.
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))

print("\n")

print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
def loss(labels, logits):# Since the model gives log of probabilities, we have to flag the from_logits

  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
example_batch_loss  = loss(target_example_batch, example_batch_predictions)

print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")

print("scalar_loss:      ", example_batch_loss.numpy().mean())
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved

checkpoint_dir = './training_checkpoints'

# Name of the checkpoint files

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")



checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(

    filepath=checkpoint_prefix,

    save_weights_only=True)
EPOCHS = 20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir) # Find the checkpoint specified with the *checkpoint_dir* path.
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1) #Let's build another model but with batches of 3 elements now (to get results of 3 sentences)



model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)) # Load the weight to the model



model.build(tf.TensorShape([1, None]))
model.summary()
def generate_text(model, start_string):

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

    temperature = 0.5



    # Here batch size == 1

    model.reset_states()

    for i in range(num_generate):

        predictions = model(input_eval)

        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)



        # using a categorical distribution to predict the character returned by the model

        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()



        # We pass the predicted character as the next input to the model

        # along with the previous hidden state

        input_eval = tf.expand_dims([predicted_id], 0)



        text_generated.append(idx2char[predicted_id])



    return (start_string + ''.join(text_generated))
print(generate_text(model, start_string=u"ROMEO: "))