import re
import os
import random
import numpy as np

# To make sure our kernel runs all the way through and gets saved,
# we'll trim some things back and skip training
IS_KAGGLE = True 

CMU_DICT_PATH = os.path.join(
    '../input', 'cmu-pronunciation-dictionary-unmodified-07b', 'cmudict-0.7b')
CMU_SYMBOLS_PATH = os.path.join(
    '../input', 'cmu-pronouncing-dictionary', 'cmudict.symbols')

# Skip words with numbers or symbols
ILLEGAL_CHAR_REGEX = "[^A-Z-'.]"

# Only 3 words are longer than 20 chars
# Setting a limit now simplifies training our model later
MAX_DICT_WORD_LEN = 20
MIN_DICT_WORD_LEN = 2


def load_clean_phonetic_dictionary():

    def is_alternate_pho_spelling(word):
        # No word has > 9 alternate pronounciations so this is safe
        return word[-1] == ')' and word[-3] == '(' and word[-2].isdigit() 

    def should_skip(word):
        if not word[0].isalpha():  # skip symbols
            return True
        if word[-1] == '.':  # skip abbreviations
            return True
        if re.search(ILLEGAL_CHAR_REGEX, word):
            return True
        if len(word) > MAX_DICT_WORD_LEN:
            return True
        if len(word) < MIN_DICT_WORD_LEN:
            return True
        return False

    phonetic_dict = {}
    with open(CMU_DICT_PATH, encoding="ISO-8859-1") as cmu_dict:
        for line in cmu_dict:

            # Skip commented lines
            if line[0:3] == ';;;':
                continue

            word, phonetic = line.strip().split('  ')

            # Alternate pronounciations are formatted: "WORD(#)  F AH0 N EH1 T IH0 K"
            # We don't want to the "(#)" considered as part of the word
            if is_alternate_pho_spelling(word):
                word = word[:word.find('(')]

            if should_skip(word):
                continue

            if word not in phonetic_dict:
                phonetic_dict[word] = []
            phonetic_dict[word].append(phonetic)

    if IS_KAGGLE: # limit dataset to 5,000 words
        phonetic_dict = {key:phonetic_dict[key] 
                         for key in random.sample(list(phonetic_dict.keys()), 5000)}
    return phonetic_dict

phonetic_dict = load_clean_phonetic_dictionary()
example_count = np.sum([len(prons) for _, prons in phonetic_dict.items()])
print("\n".join([k+' --> '+phonetic_dict[k][0] for k in random.sample(list(phonetic_dict.keys()), 10)]))
print('\nAfter cleaning, the dictionary contains %s words and %s pronunciations (%s are alternate pronunciations).' % 
      (len(phonetic_dict), example_count, (example_count-len(phonetic_dict))))
import string

START_PHONE_SYM = '\t'
END_PHONE_SYM = '\n'


def char_list():
    allowed_symbols = [".", "-", "'"]
    uppercase_letters = list(string.ascii_uppercase)
    return [''] + allowed_symbols + uppercase_letters


def phone_list():
    phone_list = [START_PHONE_SYM, END_PHONE_SYM]
    with open(CMU_SYMBOLS_PATH) as file:
        for line in file: 
            phone_list.append(line.strip())
    return [''] + phone_list


def id_mappings_from_list(str_list):
    str_to_id = {s: i for i, s in enumerate(str_list)} 
    id_to_str = {i: s for i, s in enumerate(str_list)}
    return str_to_id, id_to_str


# Create character to ID mappings
char_to_id, id_to_char = id_mappings_from_list(char_list())

# Load phonetic symbols and create ID mappings
phone_to_id, id_to_phone = id_mappings_from_list(phone_list())

# Example:
print('Char to id mapping: \n', char_to_id)
CHAR_TOKEN_COUNT = len(char_to_id)
PHONE_TOKEN_COUNT = len(phone_to_id)


def char_to_1_hot(char):
    char_id = char_to_id[char]
    hot_vec = np.zeros((CHAR_TOKEN_COUNT))
    hot_vec[char_id] = 1.
    return hot_vec


def phone_to_1_hot(phone):
    phone_id = phone_to_id[phone]
    hot_vec = np.zeros((PHONE_TOKEN_COUNT))
    hot_vec[phone_id] = 1.
    return hot_vec

# Example:
print('"A" is represented by:\n', char_to_1_hot('A'), '\n-----')
print('"AH0" is represented by:\n', phone_to_1_hot('AH0'))
MAX_CHAR_SEQ_LEN = max([len(word) for word, _ in phonetic_dict.items()])
MAX_PHONE_SEQ_LEN = max([max([len(pron.split()) for pron in pronuns]) 
                         for _, pronuns in phonetic_dict.items()]
                       ) + 2  # + 2 to account for the start & end tokens we need to add


def dataset_to_1_hot_tensors():
    char_seqs = []
    phone_seqs = []
    
    for word, pronuns in phonetic_dict.items():
        word_matrix = np.zeros((MAX_CHAR_SEQ_LEN, CHAR_TOKEN_COUNT))
        for t, char in enumerate(word):
            word_matrix[t, :] = char_to_1_hot(char)
        for pronun in pronuns:
            pronun_matrix = np.zeros((MAX_PHONE_SEQ_LEN, PHONE_TOKEN_COUNT))
            phones = [START_PHONE_SYM] + pronun.split() + [END_PHONE_SYM]
            for t, phone in enumerate(phones):
                pronun_matrix[t,:] = phone_to_1_hot(phone)
                
            char_seqs.append(word_matrix)
            phone_seqs.append(pronun_matrix)
    
    return np.array(char_seqs), np.array(phone_seqs)
            

char_seq_matrix, phone_seq_matrix = dataset_to_1_hot_tensors()        
print('Word Matrix Shape: ', char_seq_matrix.shape)
print('Pronunciation Matrix Shape: ', phone_seq_matrix.shape)
phone_seq_matrix_decoder_output = np.pad(phone_seq_matrix,((0,0),(0,1),(0,0)), mode='constant')[:,1:,:]
from keras.models import Model
from keras.layers import Input, LSTM, Dense

def baseline_model(hidden_nodes = 256):
    
    # Shared Components - Encoder
    char_inputs = Input(shape=(None, CHAR_TOKEN_COUNT))
    encoder = LSTM(hidden_nodes, return_state=True)
    
    # Shared Components - Decoder
    phone_inputs = Input(shape=(None, PHONE_TOKEN_COUNT))
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True)
    decoder_dense = Dense(PHONE_TOKEN_COUNT, activation='softmax')
    
    # Training Model
    _, state_h, state_c = encoder(char_inputs) # notice encoder outputs are ignored
    encoder_states = [state_h, state_c]
    decoder_outputs, _, _ = decoder(phone_inputs, initial_state=encoder_states)
    phone_prediction = decoder_dense(decoder_outputs)

    training_model = Model([char_inputs, phone_inputs], phone_prediction)
    
    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)
    
    # Testing Model - Decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_inputs, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    phone_prediction = decoder_dense(decoder_outputs)
    
    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [phone_prediction] + decoder_states)
    
    return training_model, testing_encoder_model, testing_decoder_model
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2
    
(char_input_train, char_input_test, 
 phone_input_train, phone_input_test, 
 phone_output_train, phone_output_test) = train_test_split(
    char_seq_matrix, phone_seq_matrix, phone_seq_matrix_decoder_output, 
    test_size=TEST_SIZE, random_state=42)

TEST_EXAMPLE_COUNT = char_input_test.shape[0]
from keras.callbacks import ModelCheckpoint, EarlyStopping

def train(model, weights_path, encoder_input, decoder_input, decoder_output):
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss',patience=3)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([encoder_input, decoder_input], decoder_output,
          batch_size=256,
          epochs=100,
          validation_split=0.2, # Keras will automatically create a validation set for us
          callbacks=[checkpointer, stopper])
BASELINE_MODEL_WEIGHTS = os.path.join(
    '../input', 'predicting-english-pronunciations-model-weights', 'baseline_model_weights.hdf5')
training_model, testing_encoder_model, testing_decoder_model = baseline_model()
if not IS_KAGGLE:
    train(training_model, BASELINE_MODEL_WEIGHTS, char_input_train, phone_input_train, phone_output_train)
def predict_baseline(input_char_seq, encoder, decoder):
    state_vectors = encoder.predict(input_char_seq) 
    
    prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
    prev_phone[0, 0, phone_to_id[START_PHONE_SYM]] = 1.
    
    end_found = False 
    pronunciation = '' 
    while not end_found:
        decoder_output, h, c = decoder.predict([prev_phone] + state_vectors)
        
        # Predict the phoneme with the highest probability
        predicted_phone_idx = np.argmax(decoder_output[0, -1, :])
        predicted_phone = id_to_phone[predicted_phone_idx]
        
        pronunciation += predicted_phone + ' '
        
        if predicted_phone == END_PHONE_SYM or len(pronunciation.split()) > MAX_PHONE_SEQ_LEN: 
            end_found = True
        
        # Setup inputs for next time step
        prev_phone = np.zeros((1, 1, PHONE_TOKEN_COUNT))
        prev_phone[0, 0, predicted_phone_idx] = 1.
        state_vectors = [h, c]
        
    return pronunciation.strip()
# Helper method for converting vector representations back into words
def one_hot_matrix_to_word(char_seq):
    word = ''
    for char_vec in char_seq[0]:
        if np.count_nonzero(char_vec) == 0:
            break
        hot_bit_idx = np.argmax(char_vec)
        char = id_to_char[hot_bit_idx]
        word += char
    return word


# Some words have multiple correct pronunciations
# If a prediction matches any correct pronunciation, consider it correct.
def is_correct(word,test_pronunciation):
    correct_pronuns = phonetic_dict[word]
    for correct_pronun in correct_pronuns:
        if test_pronunciation == correct_pronun:
            return True
    return False


def sample_baseline_predictions(sample_count, word_decoder):
    sample_indices = random.sample(range(TEST_EXAMPLE_COUNT), sample_count)
    for example_idx in sample_indices:
        example_char_seq = char_input_test[example_idx:example_idx+1]
        predicted_pronun = predict_baseline(example_char_seq, testing_encoder_model, testing_decoder_model)
        example_word = word_decoder(example_char_seq)
        pred_is_correct = is_correct(example_word, predicted_pronun)
        print('✅ ' if pred_is_correct else '❌ ', example_word,'-->', predicted_pronun)
training_model.load_weights(BASELINE_MODEL_WEIGHTS)  # also loads weights for testing models
sample_baseline_predictions(10, one_hot_matrix_to_word)
def syllable_count(phonetic_sp): 
    count = 0
    for phone in phonetic_sp.split(): 
        if phone[-1].isdigit():
            count += 1 
    return count

# Examples:
for ex_word in list(phonetic_dict.keys())[:3]:
    print(ex_word, '--', syllable_count(phonetic_dict[ex_word][0]), 'syllables')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def is_syllable_count_correct(word, test_pronunciation):
    correct_pronuns = phonetic_dict[word]
    for correct_pronun in correct_pronuns:
        if syllable_count(test_pronunciation) == syllable_count(correct_pronun):
            return True
    return False
    
    
def bleu_score(word,test_pronunciation):
    references = [pronun.split() for pronun in phonetic_dict[word]]
    smooth = SmoothingFunction().method1
    return sentence_bleu(references, test_pronunciation.split(), smoothing_function=smooth)


def evaluate(test_examples, encoder, decoder, word_decoder, predictor):
    correct_syllable_counts = 0
    perfect_predictions = 0
    bleu_scores = []
    
    for example_idx in range(TEST_EXAMPLE_COUNT):
        example_char_seq = test_examples[example_idx:example_idx+1]
        predicted_pronun = predictor(example_char_seq, encoder, decoder)
        example_word = word_decoder(example_char_seq)
        
        perfect_predictions += is_correct(example_word,predicted_pronun)
        correct_syllable_counts += is_syllable_count_correct(example_word,predicted_pronun)

        bleu = bleu_score(example_word,predicted_pronun)
        bleu_scores.append(bleu)
        
    syllable_acc = correct_syllable_counts / TEST_EXAMPLE_COUNT
    perfect_acc = perfect_predictions / TEST_EXAMPLE_COUNT
    avg_bleu_score = np.mean(bleu_scores)
    
    return syllable_acc, perfect_acc, avg_bleu_score


def print_results(model_name, syllable_acc, perfect_acc, avg_bleu_score):
    print(model_name)
    print('-'*20)
    print('Syllable Accuracy: %s%%' % round(syllable_acc*100, 1))
    print('Perfect Accuracy: %s%%' % round(perfect_acc*100, 1))
    print('Bleu Score: %s' % round(avg_bleu_score, 4))
syllable_acc, perfect_acc, avg_bleu_score = evaluate(
    char_input_test, testing_encoder_model, testing_decoder_model, one_hot_matrix_to_word, predict_baseline)
print_results('Baseline Model',syllable_acc, perfect_acc, avg_bleu_score)
from keras import backend as K
K.clear_session()
def dataset_for_embeddings():
    char_seqs = []
    phone_seqs = []
    
    for word,pronuns in phonetic_dict.items():
        word_matrix = np.zeros((MAX_CHAR_SEQ_LEN))
        for t,char in enumerate(word):
            word_matrix[t] = char_to_id[char]
        for pronun in pronuns:
            pronun_matrix = np.zeros((MAX_PHONE_SEQ_LEN))
            phones = [START_PHONE_SYM] + pronun.split() + [END_PHONE_SYM]
            for t, phone in enumerate(phones):
                pronun_matrix[t] = phone_to_id[phone]
                
            char_seqs.append(word_matrix)
            phone_seqs.append(pronun_matrix)
    
    return np.array(char_seqs), np.array(phone_seqs)

            
char_emb_matrix, phone_emb_matrix = dataset_for_embeddings()        

print('Embedding Word Matrix Shape: ', char_emb_matrix.shape)
print('Embedding Phoneme Matrix Shape: ', phone_emb_matrix.shape)
(emb_char_input_train, emb_char_input_test, 
 emb_phone_input_train, emb_phone_input_test) = train_test_split(
    char_emb_matrix, phone_emb_matrix, test_size=TEST_SIZE, random_state=42)
from keras.layers import Embedding, Dropout, Activation

def embedding_model(hidden_nodes = 256, emb_size = 256):
    
    # Shared Components - Encoder
    char_inputs = Input(shape=(None,))
    char_embedding_layer = Embedding(CHAR_TOKEN_COUNT, emb_size, input_length=MAX_CHAR_SEQ_LEN, mask_zero=True)
    encoder = LSTM(hidden_nodes, return_state=True, recurrent_dropout=0.1)
    
    # Shared Components - Decoder
    phone_inputs = Input(shape=(None,))
    phone_embedding_layer = Embedding(PHONE_TOKEN_COUNT, emb_size, mask_zero=True)
    decoder = LSTM(hidden_nodes, return_sequences=True, return_state=True, recurrent_dropout=0.1)
    decoder_dense = Dense(PHONE_TOKEN_COUNT, activation='softmax')
    
    # Training Model
    char_embeddings = char_embedding_layer(char_inputs)
    char_embeddings = Activation('relu')(char_embeddings)
    char_embeddings = Dropout(0.5)(char_embeddings)
    _, state_h, state_c = encoder(char_embeddings)
    encoder_states = [state_h, state_c]
    
    phone_embeddings = phone_embedding_layer(phone_inputs)
    phone_embeddings = Activation('relu')(phone_embeddings)
    phone_embeddings = Dropout(0.5)(phone_embeddings)
    decoder_outputs, _, _ = decoder(phone_embeddings, initial_state=encoder_states)
    decoder_outputs = Dropout(0.5)(decoder_outputs)
    phone_outputs = decoder_dense(decoder_outputs)

    training_model = Model([char_inputs, phone_inputs], phone_outputs)
    
    # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_states)
    
    # Testing Model - Decoder
    decoder_state_input_h = Input(shape=(hidden_nodes,))
    decoder_state_input_c = Input(shape=(hidden_nodes,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    test_decoder_outputs, decoder_state_h, decoder_state_c = decoder(phone_embeddings, initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    test_phone_outputs = decoder_dense(test_decoder_outputs)
    
    testing_decoder_model = Model([phone_inputs] + decoder_state_inputs, [test_phone_outputs] + decoder_states)
    
    return training_model, testing_encoder_model, testing_decoder_model
EMBEDDING_MODEL_WEIGHTS = os.path.join(
    '../input', 'predicting-english-pronunciations-model-weights', 'embedding_model_weights.hdf5')
emb_training_model, emb_testing_encoder_model, emb_testing_decoder_model = embedding_model()
if not IS_KAGGLE:
    train(emb_training_model, EMBEDDING_MODEL_WEIGHTS, emb_char_input_train, emb_phone_input_train, phone_output_train)
def id_vec_to_word(emb_char_seq):
    word = ''
    for char_id in emb_char_seq[0]:
        char = id_to_char[char_id]
        word += char
    return word.strip()
def predict_emb(input_char_seq, encoder, decoder):
    state_vectors = encoder.predict(input_char_seq) 
    output_phone_seq = np.array([[phone_to_id[START_PHONE_SYM]]])
    
    end_found = False 
    pronunciation = '' 
    while not end_found:
        decoder_output, h, c = decoder.predict([output_phone_seq] + state_vectors)
        
        # Predict the phoneme with the highest probability
        predicted_phone_idx = np.argmax(decoder_output[0, -1, :])
        predicted_phone = id_to_phone[predicted_phone_idx]
        
        pronunciation += predicted_phone + ' '
        
        if predicted_phone == END_PHONE_SYM or len(pronunciation.split()) > MAX_PHONE_SEQ_LEN: 
            end_found = True
        
        # Setup inputs for next time step
        output_phone_seq = np.array([[predicted_phone_idx]])
        state_vectors = [h, c]
        
    return pronunciation.strip()
emb_training_model.load_weights(EMBEDDING_MODEL_WEIGHTS) # also loads weights for testing models
syllable_acc, perfect_acc, avg_bleu_score = evaluate(
    emb_char_input_test, emb_testing_encoder_model, emb_testing_decoder_model, id_vec_to_word, predict_emb)
print_results('Embedding Model', syllable_acc, perfect_acc, avg_bleu_score)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_embeddings(embeddings, symbols, perplexity):
    embeddings_in_2D = TSNE(n_components=2,perplexity=perplexity).fit_transform(embeddings)
    embeddings_in_2D[:,0] = embeddings_in_2D[:,0] / np.max(np.abs(embeddings_in_2D[:,0]))
    embeddings_in_2D[:,1] = embeddings_in_2D[:,1] / np.max(np.abs(embeddings_in_2D[:,1]))

    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)
    ax.scatter(embeddings_in_2D[:,0], embeddings_in_2D[:,1],c='w')

    for i, letter in enumerate(symbols):
        ax.annotate(letter, (embeddings_in_2D[i,0],embeddings_in_2D[i,1]), fontsize=12, fontweight='bold')
        
        
char_embedding = emb_training_model.layers[2].get_weights()[0]
plot_embeddings(char_embedding, char_to_id.keys(), 5)

phone_embedding = emb_training_model.layers[3].get_weights()[0]
plot_embeddings(phone_embedding, phone_to_id.keys(), 18)
K.clear_session()
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Multiply, Reshape, RepeatVector, Lambda, Flatten
from keras.activations import softmax

def attention_model(hidden_nodes = 256, emb_size = 256):
    # Attention Mechanism Layers
    attn_repeat = RepeatVector(MAX_CHAR_SEQ_LEN)
    attn_concat = Concatenate(axis=-1)
    attn_dense1 = Dense(128, activation="tanh")
    attn_dense2 = Dense(1, activation="relu")
    attn_softmax = Lambda(lambda x: softmax(x,axis=1))
    attn_dot = Dot(axes = 1)
    
    def get_context(encoder_outputs, h_prev):
        h_prev = attn_repeat(h_prev)
        concat = attn_concat([encoder_outputs, h_prev])
        e = attn_dense1(concat)
        e = attn_dense2(e)
        attention_weights = attn_softmax(e)
        context = attn_dot([attention_weights, encoder_outputs])
        return context
    
    # Shared Components - Encoder
    char_inputs = Input(shape=(None,))
    char_embedding_layer = Embedding(CHAR_TOKEN_COUNT, emb_size, input_length=MAX_CHAR_SEQ_LEN)
    encoder = Bidirectional(LSTM(hidden_nodes, return_sequences=True, recurrent_dropout=0.2))
    
    # Shared Components - Decoder
    decoder = LSTM(hidden_nodes, return_state=True, recurrent_dropout=0.2)
    phone_embedding_layer = Embedding(PHONE_TOKEN_COUNT, emb_size)
    embedding_reshaper = Reshape((1,emb_size,))
    context_phone_concat = Concatenate(axis=-1)
    context_phone_dense = Dense(hidden_nodes*3, activation="relu")
    output_layer = Dense(PHONE_TOKEN_COUNT, activation='softmax')
    
    # Training Model - Encoder
    char_embeddings = char_embedding_layer(char_inputs)
    char_embeddings = Activation('relu')(char_embeddings)
    char_embeddings = Dropout(0.5)(char_embeddings)
    encoder_outputs = encoder(char_embeddings)
    
    # Training Model - Attention Decoder
    h0 = Input(shape=(hidden_nodes,))
    c0 = Input(shape=(hidden_nodes,))
    h = h0 # hidden state
    c = c0 # cell state
    
    phone_inputs = []
    phone_outputs = []
    
    for t in range(MAX_PHONE_SEQ_LEN):
        phone_input = Input(shape=(None,))
        phone_embeddings = phone_embedding_layer(phone_input)
        phone_embeddings = Dropout(0.5)(phone_embeddings)
        phone_embeddings = embedding_reshaper(phone_embeddings)
        
        context = get_context(encoder_outputs, h)
        phone_and_context = context_phone_concat([context, phone_embeddings])
        phone_and_context = context_phone_dense(phone_and_context)
        
        decoder_output, h, c = decoder(phone_and_context, initial_state = [h, c])
        decoder_output = Dropout(0.5)(decoder_output)
        phone_output = output_layer(decoder_output)
        
        phone_inputs.append(phone_input)
        phone_outputs.append(phone_output)
    
    training_model = Model(inputs=[char_inputs, h0, c0] + phone_inputs, outputs=phone_outputs)
    
   # Testing Model - Encoder
    testing_encoder_model = Model(char_inputs, encoder_outputs)

    # Testing Model - Decoder
    test_prev_phone_input = Input(shape=(None,))
    test_phone_embeddings = phone_embedding_layer(test_prev_phone_input)
    test_phone_embeddings = embedding_reshaper(test_phone_embeddings)
    
    test_h = Input(shape=(hidden_nodes,), name='test_h')
    test_c = Input(shape=(hidden_nodes,), name='test_c')
    
    test_encoding_input = Input(shape=(MAX_CHAR_SEQ_LEN, hidden_nodes*2,))
    test_context = get_context(test_encoding_input, test_h)
    test_phone_and_context = Concatenate(axis=-1)([test_context, test_phone_embeddings])
    test_phone_and_context = context_phone_dense(test_phone_and_context)
        
    test_seq, out_h, out_c = decoder(test_phone_and_context, initial_state = [test_h, test_c])
    test_out = output_layer(test_seq)
    
    testing_decoder_model = Model([test_prev_phone_input, test_h, test_c, test_encoding_input], [test_out,out_h,out_c])
    
    return training_model, testing_encoder_model, testing_decoder_model
def train_attention(model, weights_path, validation_size=0.2, epochs=100):    
    h0 = np.zeros((emb_char_input_train.shape[0], 256))
    c0 = np.zeros((emb_char_input_train.shape[0], 256))
    inputs = list(emb_phone_input_train.swapaxes(0,1))
    outputs = list(phone_output_train.swapaxes(0,1))
    
    callbacks = []
    if validation_size > 0:
        checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
        stopper = EarlyStopping(monitor='val_loss',patience=3)
        callbacks = [checkpointer, stopper]

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit([emb_char_input_train, h0, c0] + inputs, outputs,
              batch_size=256,
              epochs=epochs,
              validation_split=validation_size,
              callbacks=callbacks)
    
    if validation_size == 0:
        model.save_weights(weights_path)
ATTENTION_MODEL_WEIGHTS = os.path.join(
    '../input', 'predicting-english-pronunciations-model-weights', 'attention_model_weights.hdf5')
attn_training_model, attn_testing_encoder_model, attn_testing_decoder_model = attention_model()
if not IS_KAGGLE:
    train_attention(attn_training_model, ATTENTION_MODEL_WEIGHTS)
def predict_attention(input_char_seq, encoder, decoder):
    encoder_outputs = encoder.predict(input_char_seq) 

    output_phone_seq = np.array([[phone_to_id[START_PHONE_SYM]]])
    
    h = np.zeros((emb_char_input_train.shape[0], 256))
    c = np.zeros((emb_char_input_train.shape[0], 256))
    
    end_found = False 
    pronunciation = '' 
    while not end_found:
        decoder_output, h, c = decoder.predict([output_phone_seq, h, c, encoder_outputs])
        
        # Predict the phoneme with the highest probability
        predicted_phone_idx = np.argmax(decoder_output[0,:])
        predicted_phone = id_to_phone[predicted_phone_idx]
        
        pronunciation += predicted_phone + ' '
        
        if predicted_phone == END_PHONE_SYM or len(pronunciation.split()) > MAX_PHONE_SEQ_LEN: 
            end_found = True
        
        # Setup inputs for next time step
        output_phone_seq = np.array([[predicted_phone_idx]])
        
    return pronunciation.strip()
attn_training_model.load_weights(ATTENTION_MODEL_WEIGHTS) # also loads weights for testing models
syllable_acc, perfect_acc, avg_bleu_score = evaluate(
    emb_char_input_test, attn_testing_encoder_model, attn_testing_decoder_model, id_vec_to_word, predict_attention)
print_results('Attention Model', syllable_acc, perfect_acc, avg_bleu_score)
K.clear_session()
FINAL_ATTENTION_MODEL_WEIGHTS = os.path.join(
    '../input', 'predicting-english-pronunciations-model-weights', 'final_attention_model_weights.hdf5')
attn_training_model, attn_testing_encoder_model, attn_testing_decoder_model = attention_model()
if not IS_KAGGLE:
    train_attention(attn_training_model, FINAL_ATTENTION_MODEL_WEIGHTS, validation_size=0.0, epochs=29)
attn_training_model.load_weights(FINAL_ATTENTION_MODEL_WEIGHTS) # also loads weights for testing models
syllable_acc, perfect_acc, avg_bleu_score = evaluate(
    emb_char_input_test, attn_testing_encoder_model, attn_testing_decoder_model, id_vec_to_word, predict_attention)
print_results('Final Attention Model', syllable_acc, perfect_acc, avg_bleu_score)
def predict_beamsearch(input_char_seq, encoder, decoder, k=3):
    a = encoder.predict(input_char_seq) 
    
    s = np.zeros((emb_char_input_train.shape[0], 256))
    c = np.zeros((emb_char_input_train.shape[0], 256))
    
    all_seqs = []
    all_seq_scores = []
    
    live_seqs = [[phone_to_id[START_PHONE_SYM]]]
    live_scores = [0]
    live_states = [[s,c]]

    while len(live_seqs) > 0: 
        new_live_seqs = [] 
        new_live_scores = [] 
        new_live_states = []
        
        for sidx,seq in enumerate(live_seqs):
            target_seq = np.array([[seq[-1]]])
            output_token_probs, s, c = decoder.predict([target_seq] + live_states[sidx] + [a])
            
            best_token_indicies = output_token_probs[0,:].argsort()[-k:]

            for token_index in best_token_indicies:
                new_seq = seq + [token_index]
                prob = output_token_probs[0,:][token_index]
                new_seq_score = live_scores[sidx] - np.log(prob)
                if id_to_phone[token_index] == END_PHONE_SYM or len(new_seq) > MAX_PHONE_SEQ_LEN:
                    all_seqs.append(new_seq) 
                    all_seq_scores.append(new_seq_score) 
                    continue
                new_live_seqs.append(new_seq)
                new_live_scores.append(new_seq_score)
                new_live_states.append([s, c])
                
        while len(new_live_scores) > k:
            worst_seq_score_idx = np.array(new_live_scores).argsort()[-1] 
            del new_live_seqs[worst_seq_score_idx]
            del new_live_scores[worst_seq_score_idx]
            del new_live_states[worst_seq_score_idx]
            
        live_seqs = new_live_seqs
        live_scores = new_live_scores
        live_states = new_live_states
        
    best_idx = np.argmin(all_seq_scores)
    score = all_seq_scores[best_idx]
    
    pronunciation = ''
    for i in all_seqs[best_idx]:
        pronunciation += id_to_phone[i] + ' '
    
    return pronunciation.strip()
syllable_acc, perfect_acc, avg_bleu_score = evaluate(
    emb_char_input_test, attn_testing_encoder_model, attn_testing_decoder_model, id_vec_to_word, predict_beamsearch)
print_results('Final Attention Model + Beamsearch', syllable_acc, perfect_acc, avg_bleu_score)
def display_wrong_predictions(sample_count, word_decoder, encoder, decoder):
    found = 0
    while found < sample_count:
        sample_index = random.sample(range(TEST_EXAMPLE_COUNT), 1)[0]
        example_char_seq = emb_char_input_test[sample_index:sample_index+1]
        predicted_pronun = predict_attention(example_char_seq, encoder, decoder)
        example_word = word_decoder(example_char_seq)
        pred_is_correct = is_correct(example_word,predicted_pronun)
        if not pred_is_correct:
            found += 1
            print('❌ ', example_word,'-->',predicted_pronun)
            
display_wrong_predictions(10, id_vec_to_word, attn_testing_encoder_model, attn_testing_decoder_model)
