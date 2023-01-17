!ls
import pandas as pd

pd.set_option('display.max_colwidth', -1)

import ast

import seaborn as sns

import gc





%pylab inline
from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer



import keras

from keras import optimizers

from keras import backend as K

from keras import regularizers

from keras.models import Sequential, Model

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 

from keras.layers import Input, LSTM, Bidirectional

from keras.layers.merge import concatenate

from keras.utils import plot_model

from keras.callbacks import EarlyStopping

from keras import callbacks



from keras.utils import to_categorical
data = pd.read_csv("../input/fra-eng/fra.txt", delimiter='\t', header=None, names=["English", "French"])

data[:5]
data.shape
!ls -lh
!wget http://embeddings.net/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin

!pip install word2vec
import word2vec

french_word2vec = word2vec.load('frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin')
# simple english

!wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec
!ls
import io

import tqdm



EN_EMBEDDINGS_DIM = 300

FR_EMBEDDINGS_DIM = 200



def load_vectors(fname):

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')

    n, d = map(int, fin.readline().split())

    data = {}

    for line in tqdm.tqdm_notebook(fin):

        tokens = line.rstrip().split(' ')

        data[tokens[0]] = list(map(float, tokens[1:]))

    return data



english_embedding_dict = load_vectors("wiki.simple.vec")

french_embedding_dict = dict(zip(french_word2vec.vocab, french_word2vec.vectors))

def clean_text(text):

    '''Clean text by removing unnecessary characters and altering the format of words.'''



    text = text.lower()

    

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"it's", "it is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"what's", "that is", text)

    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"how's", "how is", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r"n't", " not", text)

    text = re.sub(r"n'", "ng", text)

    text = re.sub(r"'bout", "about", text)

    text = re.sub(r"'til", "until", text)

    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    

    return text
# TODO: add named entity recognition for PERSON, PLACE, etc (use spacy)
text1 = data["English"].apply(clean_text)

text2 = data["French"].apply(clean_text)
def tagger(sentence):

    bos = "<BOS> "

    eos = " <EOS>"

    return bos + sentence + eos
text1 = text1.apply(tagger)

text2 = text2.apply(tagger)
VOCAB_SIZE = 5000

def vocab_creater(text_lists, VOCAB_SIZE):

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<NEW>")

    tokenizer.fit_on_texts(text_lists)

    dictionary = tokenizer.word_index



    word2idx = {}

    idx2word = {}

    for k, v in dictionary.items():

        if v < VOCAB_SIZE:

            word2idx[k] = v

            idx2word[v] = k

            if v >= VOCAB_SIZE - 1:

                continue



    return word2idx, idx2word, tokenizer



en_word2idx, en_idx2word, en_tokenizer = vocab_creater(text_lists=text1, VOCAB_SIZE=VOCAB_SIZE)

fr_word2idx, fr_idx2word, fr_tokenizer = vocab_creater(text_lists=text2, VOCAB_SIZE=VOCAB_SIZE)
text1_sequence = en_tokenizer.texts_to_sequences(text1)

text2_sequence = fr_tokenizer.texts_to_sequences(text2)
series_text1_sequence = pd.Series([len(v) for v in text1_sequence])

series_text2_sequence = pd.Series([len(v) for v in text2_sequence])

print("95th pct text 1: {}".format(series_text1_sequence.quantile(0.95)))

print("95th pct text 2: {}".format(series_text2_sequence.quantile(0.95)))

sns.distplot(series_text1_sequence.dropna(), kde=False)
from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 10



text1_sequence = pad_sequences(text1_sequence, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

text2_sequence = pad_sequences(text2_sequence, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
en_embedding_matrix = np.zeros((len(en_word2idx) + 1, EN_EMBEDDINGS_DIM))

fr_embedding_matrix = np.zeros((len(fr_word2idx) + 1, FR_EMBEDDINGS_DIM))



for word, i in tqdm.tqdm_notebook(en_word2idx.items()):

    embedding_vector = english_embedding_dict.get(word)

    if embedding_vector is not None and len(embedding_vector) > 0:

        en_embedding_matrix[i] = embedding_vector

        

for word, i in tqdm.tqdm_notebook(fr_word2idx.items()):

    embedding_vector = french_embedding_dict.get(word)

    if embedding_vector is not None and len(embedding_vector) > 0:

        fr_embedding_matrix[i] = embedding_vector
LSTM_DIMS = 256

LEARNING_RATE = 1e-3

MOMENTUM = 0.9
K.clear_session()



# optimizer = optimizers.RMSprop()

# optimizer = optimizers.RMSprop(lr=LEARNING_RATE)

optimizer = optimizers.SGD(lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)



# Define an input sequence and process it.

encoder_inputs = Input(shape=(None, ))

english_embeddings = Embedding(VOCAB_SIZE, EN_EMBEDDINGS_DIM, weights=[en_embedding_matrix], trainable=False)

encoder = LSTM(LSTM_DIMS, return_state=True)



encoder_outputs, state_h, state_c = encoder(english_embeddings(encoder_inputs))

# We discard `encoder_outputs` and only keep the states.

encoder_states = [state_h, state_c]



# Set up the decoder, using `encoder_states` as initial state.

decoder_inputs = Input(shape=(None, ))

french_embeddings = Embedding(VOCAB_SIZE, FR_EMBEDDINGS_DIM, weights=[fr_embedding_matrix], trainable=False)

# We set up our decoder to return full output sequences,

# and to return internal states as well. We don't use the

# return states in the training model, but we will use them in inference.

decoder_lstm = LSTM(LSTM_DIMS, return_sequences=True, return_state=True)

decoder_embedded = french_embeddings(decoder_inputs)

decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)

decoder_dense = Dense(VOCAB_SIZE, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)



# Define the model that will turn

# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)



# Run training

model.compile(optimizer=optimizer, loss='categorical_crossentropy')
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model, show_shapes=True, rankdir="TB").create(prog='dot', format='svg'))
# Define sampling models

encoder_model = Model(encoder_inputs, encoder_states)



decoder_state_input_h = Input(shape=(LSTM_DIMS,))

decoder_state_input_c = Input(shape=(LSTM_DIMS,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(

    [decoder_inputs] + decoder_states_inputs,

    [decoder_outputs] + decoder_states)
SVG(model_to_dot(encoder_model, show_shapes=True).create(prog='dot', format='svg'))
text1.shape, text1_sequence.shape
from sklearn.model_selection import train_test_split

test_size = 0.2



# reverse the first sequence

chat1_train, chat1_test, text1_train, text1_test, chat2_train, chat2_test, text2_train, text2_test = train_test_split(

    text1_sequence[:, ::-1], text1, text2_sequence, text2, test_size=test_size)

text1[20:30]
text2[20:30]
BATCH_SIZE = 256

NUM_EPOCHS = 30



PATIENCE = 2

DROPOUT = .25



TRAIN_SAMPLES, _ = chat1_train.shape

NUM_SUBSETS = 8



Step = int(np.around(TRAIN_SAMPLES / NUM_SUBSETS))

SAMPLE_ROUNDS = Step * NUM_SUBSETS

Step, SAMPLE_ROUNDS



weights_file="model.h5"
def decode_sequence(input_seq):

    # Encode the input as state vectors.

    states_value = encoder_model.predict(input_seq)



    # Generate empty target sequence of length 1.

    # Populate the first character of target sequence with the start character.

    target_seq = np.zeros((1, 1))

    target_seq[0,0] = en_word2idx['bos']



    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).

    stop_condition = False

    decoded_sentence = ''

    i = 0

    total_score = 0

    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        

        # Sample a token

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        max_score = np.max(output_tokens[0, -1, :])

        total_score += -log(max_score)

        if sampled_token_index == 0:

            break

        else:

            sampled_char = fr_idx2word[sampled_token_index]

#         print(sampled_token_index, end=', ')

        decoded_sentence += sampled_char + " "



        # Exit condition: either hit max length

        # or find stop character.

        if (sampled_char == 'eos' or i > MAX_LEN):

            stop_condition = True



        # Update the target sequence (of length 1).

        target_seq = np.zeros((1, 1))

        target_seq[0,0] = sampled_token_index



        # Update states

        states_value = [h, c]

        i+=1



    return decoded_sentence.strip(), total_score
# beam search

def beam_search_decoder(input_seq, k):

    states_value = encoder_model.predict(input_seq)

    

    # Generate empty target sequence of length 1.

    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character.

    target_seq[0,0] = en_word2idx['bos']

    prev_best_tokens = [target_seq]*k

    

    # flag if the sequence has already 'ended'

    sequence_ended = [False]*k

    

    prev_states = [None]*k

    prev_states[0] = states_value



    # Sampling loop for a batch of sequences

    # (to simplify, here we assume a batch of size 1).

    stop_condition = False

    decoded_sentence = ''

    idx = 0

    

    sequences = [[list(), 0.0]]

    # for row in data:

    while not stop_condition:

        all_candidates = list()

        

        # Exit condition: either hit max length or all sequences have ended

        if (idx >= MAX_LEN or np.all(sequence_ended)):

            stop_condition = True

            break

        

        # beam search - model predict every candidate target sequence

        for i in range(len(sequences)):

            

            # if sequence has ended, proceed to next 'alive' sequence

            if sequence_ended[i]:

                continue



            target_seq = np.zeros((1, 1))

            target_seq[0, 0]  = prev_best_tokens[i]

            

            if prev_best_tokens[i] == fr_word2idx['eos']:

                sequence_ended[i] = True

                break

            

#             for seq_idx, element in enumerate(sequences[i][0]):

#                 target_seq[0, 0]  = element

#                 print(idx,i, sequences[i])

                

#                 # if element is EOS, go next

#                 if element == fr_word2idx['eos']:

#                     sequence_ended[i] = True

#                     break

            

#             print("Decoding")

            output_tokens, h, c = decoder_model.predict([target_seq] + prev_states[i])

            states_value = [h, c]

            

            # update states

            # first iteration

            if idx == 0:

                prev_states = [states_value]*k

            else:

                prev_states[i] = states_value

            

            seq, score = sequences[i]

            for j in range(VOCAB_SIZE):

                # sum of log probabilities

                # TODO: consider changing to output_tokens[0][idx][j]

                candidate = [seq + [j], score + -log(output_tokens[0][0][j])]

                all_candidates.append(candidate)

                

        # order all candidates by score

        ordered = sorted(all_candidates, key=lambda tup:tup[1])

        # select k best

        if len(sequences) == 1:

            sequences = ordered[:k]

        else:

            for i, candidate_max in enumerate(ordered[:k]):

                word, score = candidate_max

#                 print(idx, i, sequences[i], word)

                sequences[i][0].append(word[-1])

                sequences[i][1] = score

#         print("K best", sequences)

        

        # append to last candidate

        prev_best_tokens = [v[0][-1] for v in sequences]



        idx+=1

    

    sum_probabilities = [v[1] for v in sequences]

    word_sequences = [v[0] for v in sequences]

    

    list_word_sequences = []

    for index_list in word_sequences:

        list_words = []

        for word_index in index_list:

            if word_index == fr_word2idx['eos']:

                list_words.append('eos')

                break

            if word_index > 0:

                list_words.append(fr_idx2word[word_index])

        list_word_sequences.append(list_words)

    

    return list_word_sequences, sum_probabilities
def encode_sequence_one_hot(input_sequence):

    encoded = np.zeros((len(input_sequence), MAX_LEN, VOCAB_SIZE), dtype='float32')



    for i, sequence in enumerate(input_sequence):

        for t, token in enumerate(sequence):

            encoded[i, t, token] = 1

            

    return encoded
sample_input_sentence = ["Hello how are you doing?"]

processed = [tagger(clean_text(v[::-1])) for v in sample_input_sentence]

processed = en_tokenizer.texts_to_sequences(processed)

processed = pad_sequences(processed, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

print("Text: ", sample_input_sentence)

print("Response:", decode_sequence(processed))

print("Response:", beam_search_decoder(processed,3))
text2_train[:10]
chat2_train[:10]
decoder_input_data = chat2_train[:Step]

decoder_target_data = chat2_train[:Step, 1:]

decoder_target_data = np.hstack( (decoder_target_data, np.zeros((len(decoder_target_data), 1) ))).astype(np.int32)

decoder_target_data = encode_sequence_one_hot(decoder_target_data)
np.argmax(decoder_target_data[:10], axis=2)
x = range(0, NUM_EPOCHS)

VALID_LOSS = np.zeros(NUM_EPOCHS)

TRAIN_LOSS = np.zeros(NUM_EPOCHS)

histories = []



for n_epoch in range(NUM_EPOCHS):

    # Loop over training batches due to memory constraints

    for n_batch in range(0, SAMPLE_ROUNDS, Step):

        

        # convert to one hot encoded (3D tensor)

        encoder_input_data = chat1_train[n_batch:n_batch+Step]

        decoder_input_data = chat2_train[n_batch:n_batch+Step]

        # teacher forcing - one time step later

        decoder_target_data = chat2_train[n_batch:n_batch+Step, 1:]

#         decoder_target_data = np.hstack( (decoder_target_data, np.zeros((len(decoder_target_data), 1) ))).astype(np.int32)

        decoder_target_data = encode_sequence_one_hot(decoder_target_data)

        

        print('Training epoch: %d, Training examples: %d - %d'%(n_epoch, n_batch, n_batch + Step))

        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data, 

                            batch_size=BATCH_SIZE, epochs=1, validation_split=0.1, callbacks=[callbacks.ReduceLROnPlateau()])

        histories.append((history.history['loss'][0], history.history['val_loss'][0], history.history['lr'][0]))

        

        decoder_target_data = None

        gc.collect()

        print("Greedy Response:", decode_sequence(processed))

        print("Beam Response:", beam_search_decoder(processed, 3))

        



    model.save_weights(weights_file, overwrite=True)
df_history = pd.DataFrame(histories, columns=["Loss", "Val Loss", "LR"])



ax = plt.figure(figsize=(15,7)).add_subplot(111)

df_history["Loss"].plot(color='dodgerblue', ax=ax)

df_history["Val Loss"].plot(color='salmon', ax=ax)



df_history["LR"].plot(style='--', color='lightgray', ax=ax.twinx())
model.save_weights(weights_file, overwrite=True)
# sample_input_sentences = ["Hello", "Go", "Thanks", "You're welcome", "Girl", "Let's go"]

sample_input_sentences = ["Hello", "Go"]



for sample_input_sentence in sample_input_sentences:

    processed = [tagger(clean_text(v[::-1])) for v in sample_input_sentence]

    processed = en_tokenizer.texts_to_sequences(processed)

    processed = pad_sequences(processed, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')



    print("Line:", sample_input_sentence)

    print("Greedy Response:", decode_sequence(processed))

    print("Beam Response:", beam_search_decoder(processed, 3))
# def decode_sequence(input_seq):

#     # Encode the input as state vectors.

#     states_value = encoder_model.predict(input_seq)



#     # Generate empty target sequence of length 1.

#     # Populate the first character of target sequence with the start character.

#     target_seq = np.zeros((1, MAX_LEN))

#     target_seq[0] = en_word2idx['bos']



#     # Sampling loop for a batch of sequences

#     # (to simplify, here we assume a batch of size 1).

#     stop_condition = False

#     decoded_sentence = ''

#     i = 0

#     total_score = 0

#     while not stop_condition:

#         output_tokens, h, c = decoder_model.predict(

#             [target_seq] + states_value)

        

#         # Sample a token

#         sampled_token_index = np.argmax(output_tokens[0, -1, :])

#         max_score = np.max(output_tokens[0, -1, :])

#         total_score += -log(max_score)

#         if sampled_token_index == 0:

#             sampled_char = ""

#         else:

#             sampled_char = fr_idx2word[sampled_token_index]

# #         print(sampled_token_index, end=', ')

#         decoded_sentence += sampled_char + " "



#         # Exit condition: either hit max length

#         # or find stop character.

#         if (sampled_char == 'eos' or i > MAX_LEN):

#             stop_condition = True



#         # Update the target sequence (of length 1).

#         target_seq = np.zeros((1, 1))

#         target_seq[0, 0] = sampled_token_index



#         # Update states

#         states_value = [h, c]

#         i+=1



#     return decoded_sentence.strip(), total_score
# # beam search

# def beam_search_decoder(input_seq, k):

#     states_value = encoder_model.predict(input_seq)

    

#     # Generate empty target sequence of length 1.

#     target_seq = np.zeros((1, MAX_LEN))

#     # Populate the first character of target sequence with the start character.

#     target_seq[0] = en_word2idx['bos']

#     prev_best_tokens = [target_seq]*k

    

#     # flag if the sequence has already 'ended'

#     sequence_ended = [False]*k

    

#     prev_states = [None]*k

#     prev_states[0] = states_value



#     # Sampling loop for a batch of sequences

#     # (to simplify, here we assume a batch of size 1).

#     stop_condition = False

#     decoded_sentence = ''

#     idx = 0

    

#     sequences = []

#     for _ in range(k):

#         sequences.append((list(), 0.0))

#     # for row in data:

#     while not stop_condition:

#         all_candidates = list()

        

#         # Exit condition: either hit max length or all sequences have ended

#         if (idx >= MAX_LEN or np.all(sequence_ended)):

#             stop_condition = True

        

#         # beam search - model predict every candidate target sequence

#         for i in range(len(sequences)):

                

#             if idx > 0:

# #               element = prev_best_tokens[i]

# #               target_seq = np.zeros((1, MAX_LEN))

# #               target_seq[0] = element



#                 target_seq = np.zeros((1, MAX_LEN))

#                 for seq_idx, element in enumerate(sequences[i][0]):

#                     target_seq[0, seq_idx]  = element



#                     # if element is EOS, go next

#                     if element == fr_word2idx['eos']:

#                         sequence_ended[i] = True

#                         break



#                 print(idx,i, sequences[i][0])

            

#             # if sequence has ended, proceed to next 'alive' sequence

#             if sequence_ended[i]:

#                 continue

            

#             print("Decoding")

#             output_tokens, h, c = decoder_model.predict([target_seq] + prev_states[i])

#             states_value = [h, c]

            

#             # update states

#             # first iteration

#             if idx == 0:

#                 prev_states = [states_value]*k

#             else:

#                 prev_states[i] = states_value

            

#             seq, score = sequences[i]

#             for j in range(VOCAB_SIZE):

#                 # sum of log probabilities

#                 # TODO: consider changing to output_tokens[0][idx][j]

#                 candidate = [seq + [j], score + -log(output_tokens[0][0][j])]

#                 all_candidates.append(candidate)

                

#         # order all candidates by score

#         ordered = sorted(all_candidates, key=lambda tup:tup[1])

#         # select k best

#         sequences = ordered[:k]

#         print("K best", sequences)

        

#         # append to last candidate

#         prev_best_tokens = [v[0][0] for v in sequences]



#         idx+=1

    

#     sum_probabilities = [v[1] for v in sequences]

#     word_sequences = [v[0] for v in sequences]

    

#     list_word_sequences = []

#     for index_list in word_sequences:

#         list_words = []

#         for word_index in index_list:

#             if word_index == fr_word2idx['eos']:

#                 break

#             if word_index > 0:

#                 list_words.append(fr_idx2word[word_index])

#         list_word_sequences.append(list_words)

    

#     return list_word_sequences, sum_probabilities