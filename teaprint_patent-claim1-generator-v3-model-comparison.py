%load_ext autoreload
%autoreload 2
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import HTML

InteractiveShell.ast_node_interactivity = 'all'

import warnings
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)

import pandas as pd
import numpy as np
#from utils import get_data, generate_output, guess_human, seed_sequence, get_embeddings, find_closest

from keras.models import load_model
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer

from sklearn.utils import shuffle

from IPython.display import HTML

from itertools import chain
from keras.utils import plot_model
import numpy as np
import pandas as pd
import random
import json
import re
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#data = pd.read_csv('../input/the-first-claim-of-patents/semiconductor_patents_claim1.csv')
#data.head()
def format_sequence(s):
    """Add spaces around punctuation."""
    
    # Add spaces around punctuation
    s =  re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', s)
    
    # Remove references to figures
    s = re.sub(r'\((\d+)\)', r'', s)
    
    # Remove double spaces
    s = re.sub(r'\s\s', ' ', s)
    return s

def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction=0.7):
    """Create training and validation features and labels."""
    
    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory management
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid
def cleanser(txts):
    ##input 'txts': a list of text
    cleansed1 = [' '.join(i.split(' ')[1:-1]) if i[-3:]=='...' else ' '.join(i.split(' ')[1:]) for i in txts]
    
    ### Add spaces to the two sides of punctuations
    cleansed2 = [re.sub(r'(?<=[^\s0-9])(?=[.,;:?])', r' ', i) for i in cleansed1]
    #print(cleansed2)
    cleansed3 = [re.sub(r'(?<=[.,;:?])(?=[^\s0-9])', r' ', i) for i in cleansed2]
    #print(cleansed3)
    return cleansed3

def make_sequences(texts,
                   training_length=50,
                   lower=True,
                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    """Turn a set of texts into sequences of integers"""
    # Data cleaning before tokenizaion
    texts = cleanser(texts)
    
    # Create the tokenizer object and train on texts
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    print(f'There are {num_words} unique words.')

    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)

    # Limit to sequences with more than training length tokens
    seq_lengths = [len(x) for x in sequences]
    over_idx = [
        i for i, l in enumerate(seq_lengths) if l > (training_length + 20)
    ]

    new_texts = []
    new_sequences = []

    # Only keep sequences with more than training length tokens
    for i in over_idx:
        new_texts.append(texts[i])
        new_sequences.append(sequences[i])

    training_seq = []
    labels = []

    # Iterate through the sequences of tokens
    for seq in new_sequences:

        # Create multiple training examples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label
            extract = seq[i - training_length:i + 1]

            # Set the features and label
            training_seq.append(extract[:-1])
            labels.append(extract[-1])

    print(f'There are {len(training_seq)} training sequences.')

    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, training_seq, labels

def get_data2(file, filters='"#$%&*+/:<=>?@[\\]^_`{|}~\t\n', training_len=50,
             lower=False):
    """Retrieve formatted training and validation data from a file"""
    
    data = pd.read_csv(file, parse_dates=['Application No.']).dropna(subset = ['First Claim'])
    #claims = [format_sequence(a) for a in list(data['First Claim'])]
    claims = [a for a in list(data['First Claim'])]
    word_idx, idx_word, num_words, word_counts, texts, sequences, features, labels = make_sequences(
        claims, training_len, lower, filters)
    X_train, X_valid, y_train, y_valid = create_train_valid(features, labels, num_words)
    training_dict = {'X_train': X_train, 'X_valid': X_valid, 
                     'y_train': y_train, 'y_valid': y_valid}
    return training_dict, word_idx, idx_word, sequences


filters='"#$%&*+/<=>?@[\\]^_`{|}~\t\n' # ,;.: are recoginized valid tokens
training_len=50
lower=False

### read dataset 
data = pd.read_csv('../input/semiconductor_patents_claim1_cleansed2.csv')
#data = pd.read_csv('../input/patent-abstract/neural_network_patent_query.csv')
training_dict, word_idx, idx_word, sequences = get_data2('../input/semiconductor_patents_claim1_cleansed2.csv', filters=filters, training_len = 50)

data.head()
clms = [i for i in data['First Claim']]
word_idx, idx_word, num_words, word_counts, texts, sequences, features, labels = make_sequences(
        clms, training_len, lower, filters)
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam

from keras.utils import plot_model
def rnn_construction(model, word_idx):
    # Embedding layer
    model.add(
            Embedding(
                input_dim=len(word_idx) + 1,
                output_dim=100,
                weights=None,
                trainable=True))

    # Recurrent layer
    model.add(
        LSTM(
            64, return_sequences=False, dropout=0.1,
            recurrent_dropout=0.1))

    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(len(word_idx) + 1, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()
    
    return model

model = Sequential()
model = rnn_construction(model,word_idx)
h = model.fit(training_dict['X_train'], training_dict['y_train'], epochs = 50, batch_size = 2048, 
          validation_data = (training_dict['X_valid'], training_dict['y_valid']), 
          verbose = 1)
# Check the model performance
print('Model Performance: Log Loss and Accuracy on training data')
model.evaluate(training_dict['X_train'], training_dict['y_train'], batch_size = 2048)

print('\nModel Performance: Log Loss and Accuracy on validation data')
model.evaluate(training_dict['X_valid'], training_dict['y_valid'], batch_size = 2048)
# Save the model
model.save('model_rnn.h5')
### source: https://github.com/WillKoehrsen/recurrent-neural-networks/blob/master/notebooks/utils.py
from keras.models import load_model
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Flatten
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer

from sklearn.utils import shuffle

from IPython.display import HTML

from itertools import chain
from keras.utils import plot_model
import numpy as np
import pandas as pd
import random
import json
import re

def remove_spaces(s):
    """Remove spaces around punctuation"""
    s = re.sub(r'\s+([.,;:?])', r'\1', s)
    return s

def generate_output(model,
                    sequences,
                    idx_word,
                    seed_length=50,
                    new_words_min=50,
                    new_words_max = 1000,
                    diversity=1,
                    n_gen=1):
    """Generate `new_words` words of output from a trained model and format into HTML."""

    # Choose a random sequence
    seq = random.choice(sequences)
    #print([idx_word[i] for i in seq])
    
    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - seed_length - 10)
    seed_idx = 0
    
    # Ending index for seed
    end_idx = seed_idx + seed_length
    
    dot_idx = word_idx['.']
    
    gen_list = []

    # Extract the seed sequence
    seed = seq[seed_idx:end_idx]
    #print(' '.join([idx_word[i] for i in seed]))
    
    generated = list()

    next_idx = -1
    window = seed.copy()
    # Keep adding new words
    for i in range(new_words_max):
        # Check the termination condition:
        if next_idx == dot_idx and i >= new_words_min:
            break

        # Make a prediction from the seed
        preds = model.predict(np.array(window).reshape(1, -1))[0].astype(
            np.float64)

        # Diversify
        preds = np.log(preds) / diversity
        exp_preds = np.exp(preds)

        # Softmax
        preds = exp_preds / sum(exp_preds)

        # Choose the next word
        probas = np.random.multinomial(1, preds, 1)[0]

        next_idx = np.argmax(probas)

        # New seed adds on old word
        window = window[1:] + [next_idx]
        #seed += [next_idx]
        #print(next_idx)
        generated.append(next_idx)
            
    # Find the actual entire sequence
    #print(i, len(generated))
    actual_seq = seq[end_idx:end_idx + len(generated)]

    # Decode sequences into words
    a_text = [idx_word[j] for j in actual_seq]
    gen_text = [idx_word[j] for j in generated]
    #print(len(actual_seq), len(generated))
    
    original_txt = remove_spaces(' '.join([idx_word[i] for i in seed]))
    #gen_txt = remove_spaces(' '.join(gen_list[0]))
    gen_txt = '< --- >' + remove_spaces(' '.join(gen_text))
    actual_txt = '< --- >' + remove_spaces(' '.join(a_text))
    return [original_txt, gen_txt, actual_txt]
    #return original_sequence, gen_list, a


original, gen, actual = generate_output(model, sequences, idx_word, seed_length = 50, new_words_min = 30, new_words_max = 200,diversity = 1.50)
print('Original seed text:')
print(original)
print('RNN generated text:')
print(gen)
print('Actual text:')
print(actual)
def mlp_construction(model, word_idx, training_len):
    # Embedding layer
    model.add(
            Embedding(
                input_dim=len(word_idx) + 1,
                output_dim=100,
                weights=None,
                trainable=True,
                input_length = training_len))
    
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(len(word_idx) + 1, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()
    
    return model

len(word_idx)
model_mlp = Sequential()
model_mlp = mlp_construction(model_mlp, word_idx, training_len)

### Model fitting
import time
start_time = time.time()
model_mlp.fit(training_dict['X_train'], training_dict['y_train'], epochs = 50, batch_size = 2048, 
          validation_data = (training_dict['X_valid'], training_dict['y_valid']), 
          verbose = 1)
print('training time = ', (time.time()-start_time)/60.,'mins')

### Check the model performance
print('Model Performance: Log Loss and Accuracy on training data')
model_mlp.evaluate(training_dict['X_train'], training_dict['y_train'], batch_size = 2048)

print('\nModel Performance: Log Loss and Accuracy on validation data')
model_mlp.evaluate(training_dict['X_valid'], training_dict['y_valid'], batch_size = 2048)

# Save the model
model_mlp.save('model_mlp.h5')
### Check the text generated by RNN and MLP
original, gen, actual = generate_output(model, sequences, idx_word, seed_length = 50, new_words_min = 30, new_words_max = 200,diversity = 1.50)
print('Original seed text:')
print(original)
print('RNN generated text:')
print(gen)
print('Actual text:')
print(actual)
original, gen, actual = generate_output(model_mlp, sequences, idx_word, seed_length = 50, new_words_min = 30, new_words_max = 200,diversity = 1.50)
print('Original seed text:')
print(original)
print('RNN generated text:')
print(gen)
print('Actual text:')
print(actual)
def bi_construction(model, word_idx,lstm_cells=64):
    # Embedding layer
    model.add(
            Embedding(
                input_dim=len(word_idx) + 1,
                output_dim=100,
                weights=None,
                trainable=True))
    '''
    # Recurrent layer
    model.add(
        LSTM(
            lstm_cells, return_sequences=False, dropout=0.1,
            recurrent_dropout=0.1))
    '''
    # Bi-directional LSTM layer
    model.add(
    Bidirectional(
        LSTM(
            lstm_cells,
            return_sequences=False,
            dropout=0.1,
            recurrent_dropout=0.1)))
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(len(word_idx) + 1, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.summary()
    
    return model

model_bi = Sequential()
model_bi = bi_construction(model_bi, word_idx)
### Model fitting
import time
start_time = time.time()
model_bi.fit(training_dict['X_train'], training_dict['y_train'], epochs = 50, batch_size = 2048, 
          validation_data = (training_dict['X_valid'], training_dict['y_valid']), 
          verbose = 1)
print('training time = ', (time.time()-start_time)/60.,'mins')

### Check the model performance
print('Model Performance: Log Loss and Accuracy on training data')
model_bi.evaluate(training_dict['X_train'], training_dict['y_train'], batch_size = 2048)

print('\nModel Performance: Log Loss and Accuracy on validation data')
model_bi.evaluate(training_dict['X_valid'], training_dict['y_valid'], batch_size = 2048)

# Save the model
model_bi.save('model_bi.h5')
def generate_output(model,
                    sequences,
                    idx_word,
                    seed_length=50,
                    new_words=50,
                    diversity=1,
                    return_output=False,
                    n_gen=1):
    """Generate `new_words` words of output from a trained model and format into HTML."""

    # Choose a random sequence
    seq = random.choice(sequences)

    # Choose a random starting point
    seed_idx = random.randint(0, len(seq) - seed_length - 10)
    # Ending index for seed
    end_idx = seed_idx + seed_length

    gen_list = []

    for n in range(n_gen):
        # Extract the seed sequence
        seed = seq[seed_idx:end_idx]
        original_sequence = [idx_word[i] for i in seed]
        generated = seed[:] + ['#']

        # Find the actual entire sequence
        actual = generated[:] + seq[end_idx:end_idx + new_words]

        # Keep adding new words
        for i in range(new_words):

            # Make a prediction from the seed
            preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
                np.float64)

            # Diversify
            preds = np.log(preds) / diversity
            exp_preds = np.exp(preds)

            # Softmax
            preds = exp_preds / sum(exp_preds)

            # Choose the next word
            probas = np.random.multinomial(1, preds, 1)[0]

            next_idx = np.argmax(probas)

            # New seed adds on old word
            seed = seed[1:] + [next_idx]
            #seed += [next_idx]
            # print(len(seed))
            generated.append(next_idx)

        # Showing generated and actual abstract
        n = []

        for i in generated:
            n.append(idx_word.get(i, '< --- >'))

        gen_list.append(n)

    a = []

    for i in actual:
        a.append(idx_word.get(i, '< --- >'))

    a = a[seed_length:]

    gen_list = [gen[seed_length:seed_length + len(a)] for gen in gen_list]

    if return_output:
        return original_sequence, gen_list, a

    # HTML formatting
    seed_html = ''
    seed_html = addContent(seed_html, header(
        'Seed Sequence', color='darkblue'))
    seed_html = addContent(seed_html,
                           box(remove_spaces(' '.join(original_sequence))))

    gen_html = ''
    gen_html = addContent(gen_html, header('RNN Generated', color='darkred'))
    gen_html = addContent(gen_html, box(remove_spaces(' '.join(gen_list[0]))))

    a_html = ''
    a_html = addContent(a_html, header('Actual', color='darkgreen'))
    a_html = addContent(a_html, box(remove_spaces(' '.join(a))))

    return seed_html, gen_html, a_html

def guess_human(model, sequences, idx_word, seed_length=50):
    """Produce 2 RNN sequences and play game to compare to actaul.
       Diversity is randomly set between 0.5 and 1.25"""
    
    new_words = np.random.randint(10, 50)
    diversity = np.random.uniform(0.5, 1.25)
    sequence, gen_list, actual = generate_output(model, sequences, idx_word, seed_length, new_words,
                                                 diversity=diversity, return_output=True, n_gen = 2)
    gen_0, gen_1 = gen_list
    
    output = {'sequence': remove_spaces(' '.join(sequence)),
              'computer0': remove_spaces(' '.join(gen_0)),
              'computer1': remove_spaces(' '.join(gen_1)),
              'human': remove_spaces(' '.join(actual))}
    
    print(f"Seed Sequence: {output['sequence']}\n")
    
    choices = ['human', 'computer0', 'computer1']
          
    selected = []
    i = 0
    while len(selected) < 3:
        choice = random.choice(choices)
        selected.append(choice)
        print(f'\nOption {i + 1} {output[choice]}')
        choices.remove(selected[-1])
        i += 1
    
    print('\n')
    guess = int(input('Enter option you think is human (1-3): ')) - 1
    print('\n')
    
    if guess == np.where(np.array(selected) == 'human')[0][0]:
        print('*' * 3 + 'Correct' + '*' * 3 + '\n')
        print('-' * 60)
        print('Ordering: ', selected)
    else:
        print('*' * 3 + 'Incorrect' + '*' * 3 + '\n')
        print('-' * 60)
        print('Correct Ordering: ', selected)
          
    print('Diversity', round(diversity, 2))
guess_human(model, sequences, idx_word)
