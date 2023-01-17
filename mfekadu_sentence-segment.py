from keras.models import Model

from keras.layers import Input, LSTM, Dense

import pandas as pd

from os.path import join

from os import listdir

from sklearn.utils import shuffle
COURSE_NUMBER = 482  # the seed



# Set the `PYTHONHASHSEED` environment variable at a fixed value

from os import environ

environ['PYTHONHASHSEED'] = str(COURSE_NUMBER)



# Set the `python` built-in pseudo-random generator at a fixed value



from random import seed

seed(COURSE_NUMBER)



# Set the `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(COURSE_NUMBER)  # for numpy, scikit-learn



# Set the `tensorflow` pseudo-random generator at a fixed value

import tensorflow as tf

tf.set_random_seed(COURSE_NUMBER)





# Configure a new global `tensorflow` session

from keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)
INPUT_DIR = "../input/sentences"

files = listdir(INPUT_DIR)

print(files)
def read_utf8_csv(filepath):

    return pd.read_csv(filepath, encoding="utf-8")



input1_df = read_utf8_csv(join(INPUT_DIR, "cv-unique-no-end-punct-sentences.csv"))

input1_df.columns = ["num", "sent_in"]

input2_df = read_utf8_csv(join(INPUT_DIR, "simple-wiki-unique-no-end-punct-sentences.csv"))

input2_df.columns = ["num", "sent_in"]

output1_df = read_utf8_csv(join(INPUT_DIR, "cv-unique-has-end-punct-sentences.csv"))

output1_df.columns = ["num", "sent_out"]

output2_df = read_utf8_csv(join(INPUT_DIR, "simple-wiki-unique-has-end-punct-sentences.csv"))

output2_df.columns = ["num", "sent_out"]
# pd.concat([input_df, output_df])

df = pd.DataFrame()

df['sent_in'] = pd.concat([input1_df['sent_in'], input2_df['sent_in']])

df['sent_out'] = pd.concat([output1_df['sent_out'], output2_df['sent_out']])

df = df.reset_index()

print("\nhere's one sample of the {} sentences...\n".format(len(df)))

df.head().to_numpy()[0]
# Convert to lowercase

df['sent_in'] = df['sent_in'].str.lower()

df['sent_out'] = df['sent_out'].str.lower()
# convert digits to 0's while retaining number length

for digit in "0123456789":

    df['sent_in'] = df['sent_in'].str.replace(digit, "0")
# check what it looks like

# df['sent_in'][df['sent_in'].str.find("0") > -1][:5]

df['sent_in'][66]
# shuffle so that the order of the data does not matter

df = shuffle(df)

indices = df.index.values

# save the indices for reproducability

pd.DataFrame(indices).to_csv("shuffled_indices.csv")
# Make random pairs of sentences

arr1, arr2 = np.split(df['sent_in'].drop(1), 2)  # drop 1 for even split

arr3, arr4 = np.split(df['sent_out'].drop(1), 2)  # drop 1 for even split

pair_df = pd.DataFrame()

# also append a space to delimit the sentences

pair_df['pair_in'] = arr1.str.cat(" " + arr2.values).to_list()

pair_df['pair_out'] = arr3.str.cat(" " + arr4.values).to_list()
pair_df['pair_in'][0]
pair_df['pair_out'][0]
batch_size = 64  # Batch size for training.

epochs = 100  # Number of epochs to train for.

latent_dim = 256  # Latent dimensionality of the encoding space.

num_samples = 10000  # Number of samples to train on.

# Path to the data txt file on disk.

data_path = 'fra-eng/fra.txt'

START_CHAR = '\t'  # the "start sequence" character

END_CHAR = '\n'  # the "end sequence" character.



# Vectorize the data.

input_texts = []

target_texts = []

input_characters = set()

target_characters = set()
# let's use the dataframe of pairs

df = pair_df
for pair in df[: min(num_samples, len(df) - 1)].values:

    input_text, target_text = pair

    target_text = START_CHAR + target_text + END_CHAR

    input_texts.append(input_text)

    target_texts.append(target_text)

    for char in input_text:

        if char not in input_characters:

            input_characters.add(char)

    for char in target_text:

        if char not in target_characters:

            target_characters.add(char)
input_characters = sorted(list(input_characters))

target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)

num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])

max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('Number of samples:', len(input_texts))

print('Number of unique input tokens:', num_encoder_tokens)

print('Number of unique output tokens:', num_decoder_tokens)

print('Max sequence length for inputs:', max_encoder_seq_length)

print('Max sequence length for outputs:', max_decoder_seq_length)
input_token_index = dict(

    [(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict(

    [(char, i) for i, char in enumerate(target_characters)])
encoder_input_data = np.zeros(

    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),

    dtype='float32')

decoder_input_data = np.zeros(

    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

    dtype='float32')

decoder_target_data = np.zeros(

    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),

    dtype='float32')
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

    for t, char in enumerate(input_text):

        encoder_input_data[i, t, input_token_index[char]] = 1.

    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.

    for t, char in enumerate(target_text):

        # decoder_target_data is ahead of decoder_input_data by one timestep

        decoder_input_data[i, t, target_token_index[char]] = 1.

        if t > 0:

            # decoder_target_data will be ahead by one timestep

            # and will not include the start character.

            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.

    decoder_target_data[i, t:, target_token_index[' ']] = 1.
# Define an input sequence and process it.

encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder = LSTM(latent_dim, return_state=True)

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.

encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.

decoder_inputs = Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,

# and to return internal states as well. We don't use the

# return states in the training model, but we will use them in inference.

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_inputs,

                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Run training

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',

              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,

          batch_size=batch_size,

          epochs=epochs,

          validation_split=0.2)
# Save model

model.save('s2s.h5')
# Next: inference mode (sampling).

# Here's the drill:

# 1) encode input and retrieve initial decoder state

# 2) run one step of decoder with this initial state

# and a "start of sequence" token as target.

# Output will be the next target token

# 3) Repeat with the current target token and current states



# Define sampling models

encoder_model = Model(encoder_inputs, encoder_states)



decoder_state_input_h = Input(shape=(latent_dim,))

decoder_state_input_c = Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(

    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(

    [decoder_inputs] + decoder_states_inputs,

    [decoder_outputs] + decoder_states)



# Reverse-lookup token index to decode sequences back to

# something readable.

reverse_input_char_index = dict(

    (i, char) for char, i in input_token_index.items())

reverse_target_char_index = dict(

    (i, char) for char, i in target_token_index.items())
def decode_sequence(input_seq):

    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)



    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Populate the first character of target sequence with the start character.

    target_seq[0, 0, target_token_index['\t']] = 1.



    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).

    stop_condition = False

    decoded_sentence = ''

    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(

            [target_seq] + states_value)



        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = reverse_target_char_index[sampled_token_index]

        decoded_sentence += sampled_char



        # Exit condition: either hit max length

        # or find stop character.

        if (sampled_char == '\n' or

           len(decoded_sentence) > max_decoder_seq_length):

            stop_condition = True



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1, 1, num_decoder_tokens))

        target_seq[0, 0, sampled_token_index] = 1.



        # Update states

        states_value = [h, c]



    return decoded_sentence
for seq_index in range(100):

    # Take one sequence (part of the training set)

    # for trying out decoding.

    input_seq = encoder_input_data[seq_index: seq_index + 1]

    decoded_sentence = decode_sequence(input_seq)

    print('-')

    print('Input sentence:', input_texts[seq_index])

    print('Decoded sentence:', decoded_sentence)