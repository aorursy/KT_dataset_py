import tensorflow as tf

import numpy as np

import os

import time
path_to_file = tf.keras.utils.get_file('The Jungle Book.txt', 'https://www.gutenberg.org/files/236/236-0.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
start_index = text.find("START OF THIS PROJECT GUTENBERG")

end_index = text.find('End of the Project Gutenberg')

text = text[start_index : end_index]
text[:1000]
# The unique characters in the file

vocab = sorted(set(text))

print('Length of text: {} characters.'.format(len(text)))

print('Unique characters: {}'.format(len(vocab)))
char2idx = {char:i for i, char in enumerate(vocab)}

idx2char = np.array(vocab)
text_encoded = [char2idx[c] for c in text]

text_encoded = np.array(text_encoded)
# Show how the first 31 characters from the text are mapped to integers

print('Text: {} \n==> Encoded as : {}'.format(text[:31], text_encoded[:31]))
# The maximum length sentence we want for a single input in characters

seq_length = 100

example_per_epoch = len(text)//seq_length # as we have 1 example of seq_length characters.
# Create training examples / targets

char_dataset = tf.data.Dataset.from_tensor_slices(text_encoded)
for i in char_dataset.take(5):

    print(idx2char[i.numpy()])
sequences = char_dataset.batch(batch_size=seq_length+1, drop_remainder=True)
for item in sequences.take(5):

    print(repr(''.join(idx2char[item.numpy()]))) # repr function print string representation of an object.
# Always try to make preprocessing function for one input and after apply it on whole list by using map or apply.

def split_input_target(chunk):

    input_text = chunk[:-1]

    target_text = chunk[1:]

    return input_text, target_text
dataset = sequences.map(split_input_target)
for input_ex, target_ex in dataset.take(1):

    print('Input data: ', repr(''.join(idx2char[input_ex.numpy()])))

    print('Output data:', repr(''.join(idx2char[target_ex.numpy()])))
for i, (input_idx, target_idx) in enumerate(zip(input_ex[:5], target_ex[:5])):

    print("Step {:4d}".format(i))

    print("Input : {} ({:s})".format(input_idx, repr(idx2char[input_idx])))

    print("Expected output : {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
# Batch size 

BATCH_SIZE = 64



# Buffer size to shuffle the dataset

# (TF data is designed to work with possibly infinite sequences,

# so it doesn't attempt to shuffle the entire sequence in memory. Instead,

# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
# Length of vocabulary in chars

vocab_size = len(vocab)



# The embedding dimension

embedding_dim = 256



# Number of RNN units

rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

    model = tf.keras.Sequential([

        tf.keras.layers.Embedding(input_dim=vocab_size,

                                  output_dim = embedding_dim,

                                  batch_input_shape = [batch_size, None]),

        tf.keras.layers.GRU(units = rnn_units,

                            return_sequences= True,

                            stateful=True,

                            recurrent_initializer='glorot_uniform'),

        tf.keras.layers.Dense(vocab_size)

    ])

    

    return model
model = build_model(vocab_size = len(vocab),

                    embedding_dim = embedding_dim,

                    rnn_units = rnn_units,

                    batch_size = BATCH_SIZE)
model.summary()
# First check the shape of the output:

for input_ex_batch, target_ex_batch in dataset.take(1):

    ex_batch_prediction = model(input_ex_batch) # simply it takes input and calculate output with initial weights.

    print(ex_batch_prediction.shape, "# (batch_size, sequence_length, vocab_size)")
sampled_indices = tf.random.categorical(ex_batch_prediction[0], num_samples=1)

sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
sampled_indices
print('Input: \n', repr("".join(idx2char[input_ex_batch[0].numpy()])))

print('\nPredicted sequence for next characters is: \n', repr("".join(idx2char[sampled_indices])))
def loss(labels, logits):

    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)



ex_batch_loss = loss(target_ex_batch, ex_batch_prediction)

print("Prediction shape: ", ex_batch_prediction.shape, " # (batch_size, sequence_length, vocab_size)")

print("Scaler loss: ", ex_batch_loss.numpy().mean())
model.compile(optimizer='adam', loss=loss)
# Directory where the checkpoints will be saved

checkpoint_dir = './training_checkpoints'



# Name of the checkpoint files

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_prefix, save_weights_only=True)
EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None])) # Builds the model based on input shapes received.
model.summary()
def generate_text(model, start_string, num_generate, temperature):

    

    # Converting our start string to numbers (vectorizing)

    input_eval = [char2idx[s] for s in start_string]

    # convert (x,y) shaped matrix to (1,x,y).

    input_eval = tf.expand_dims(input_eval, axis=0) 

    

    # Empty string to store our results

    text_generated = []

    

    # Here batch size == 1

    model.reset_states()

    for i in range(num_generate):

        predictions = model(input_eval)

        

        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        

        # using a categorical distribution to predict the 

        # character returned by the model

        predictions = predictions / temperature

        

        # We got the predictions for every timestep but we 

        # want only last so first we take [-1] to consider on last 

        # predictions distribution only and after we try to get id 

        # from 1D array. Ex. we got '2' from a=['2'] by a[0].

        predicted_id = tf.random.categorical(predictions, 

                                             num_samples=1

                                            )[-1,0].numpy()

        

        # We pass the predicted character as the next input to the 

        # model along with the previous hidden state

        input_eval = tf.expand_dims([predicted_id], 0)

        

        text_generated.append(idx2char[predicted_id])

        

    return (start_string + ''.join(text_generated))
print(generate_text(model, 

                    start_string=u"The moon was sinking behind the", 

                    num_generate=500, temperature=1.0))
print(generate_text(model, 

                    start_string=u"The moon was sinking behind the", 

                    num_generate=500, temperature=0.5))