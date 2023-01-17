import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re # regex package

from keras.models import Sequential

from keras.layers import Dense, Activation, TimeDistributed, LSTM, Dropout

from keras import optimizers



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



def load_data(file, nrows = 100):

    df = pd.read_csv('../input/' + file + '.csv', nrows = nrows)

    return(df)



def shape_data(df):

    df = df.groupby('author', as_index=True).agg([lambda x: ' '.join(x)])

    return(df)



def to_char_list(string):

    char_list = []

    for char in string:

        char_list.append(char)

    return(char_list)



## replace multiple whitespaces with single ' ' character

## also trim the start and end of the string, because it will come from a list

## so has leading and trailing '"[ ' and ']"' characters respectively

def clean_text(s):

    replaced = re.sub('\\s+', ' ', s).strip() # replace multiple whitespaces with single ' ' character

    replaced = replaced[3:-2] # remove leading and trailing '"[ ' and ']"' characters respectively

    return(replaced)



## Model

HIDDEN_DIM = 256

LAYER_NUM = 2

def get_model(learning_rate = 0.009):

    model = Sequential()

    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))

    model.add(Dropout(0.3))

    for i in range(LAYER_NUM - 1):

        model.add(LSTM(HIDDEN_DIM, return_sequences=True))

        model.add(Dropout(0.3))

    model.add(TimeDistributed(Dense(VOCAB_SIZE)))

    model.add(Activation('softmax'))

    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return(model)



## Train model

BATCH_SIZE = 64

GENERATE_LENGTH = 150

def train_model(model, X, y, epochs, every=2):

    nb_epoch = 0

    while True:

        #print('\n\n')

        if not nb_epoch % every == 0:

            model.fit(X, y, batch_size=BATCH_SIZE, verbose=0, epochs=1)

            nb_epoch += 1

        if nb_epoch % every == 0:

            model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)

            print('nb_epoch = %i'%nb_epoch)

            print(generate_text(model, GENERATE_LENGTH))

            model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))

            if nb_epoch % epochs == 0 and not nb_epoch == 0:

                break

            nb_epoch += 1

            

## Text Generation function

def generate_text(model, length):

    ix = [np.random.randint(VOCAB_SIZE)]

    #ix = [char_to_ix['.']]

    y_char = [ix_to_char[ix[-1]]]

    X = np.zeros((1, length, VOCAB_SIZE))

    for i in range(length):

        X[0, i, :][ix[-1]] = 1

        #print(ix_to_char[ix[-1]], end="")

        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)

        y_char.append(ix_to_char[ix[-1]])

    return ('').join(y_char)
## Main ## 

train_sample = load_data('train', nrows=5000) # read first 5000 rows of the dataset

print(train_sample.shape)

#test = load_data('test')

train = shape_data(train_sample) # aggregate text column, groupped by author

print(train.shape)
eap_text = clean_text(str(train['text'].loc['EAP',:].values))

mws_text = clean_text(str(train['text'].loc['MWS',:].values))

hpl_text = clean_text(str(train['text'].loc['HPL',:].values))
eap_char_list = to_char_list(eap_text)

print('num characters in EAP text: %i'%len(eap_char_list))

mws_char_list = to_char_list(mws_text)

print('num characters in MWS text: %i'%len(mws_char_list))

hpl_char_list = to_char_list(hpl_text)

print('num characters in HPL text: %i'%len(hpl_char_list))
eap_char_to_ix = {x:y for x,y in zip(set(eap_char_list),range(len(set(eap_char_list))))}

mws_char_to_ix = {x:y for x,y in zip(set(mws_char_list),range(len(set(mws_char_list))))}

hpl_char_to_ix = {x:y for x,y in zip(set(hpl_char_list),range(len(set(hpl_char_list))))}

print('num unique chars eap: {}\nnum unique chars mws: {}\nnum unique chars hpl: {}'.format(len(eap_char_to_ix),len(mws_char_to_ix),len(hpl_char_to_ix)))
## Quick Question: which characters does Edgar Allen Poe use that Mary Shelley does not?

## (same question for H.P. Lovecraft and all 6 combinations).

## List Comprehensions

## Letters Edgar Allen Poe Uses

eap_not_mws = [x for x in list(eap_char_to_ix.keys()) if x not in list(mws_char_to_ix.keys())]

eap_not_hpl = [x for x in list(eap_char_to_ix.keys()) if x not in list(hpl_char_to_ix.keys())]

## Letters Mary Shelley Uses

mws_not_eap = [x for x in list(mws_char_to_ix.keys()) if x not in list(eap_char_to_ix.keys())]

mws_not_hpl = [x for x in list(mws_char_to_ix.keys()) if x not in list(hpl_char_to_ix.keys())]

## Letters Lovecraft Uses

hpl_not_eap = [x for x in list(hpl_char_to_ix.keys()) if x not in list(eap_char_to_ix.keys())]

hpl_not_mws = [x for x in list(hpl_char_to_ix.keys()) if x not in list(mws_char_to_ix.keys())]



# Even when loading the full training set, this didn't really lead to any obvious character

# abuses, except for some EAP and HPL text having non-english characters.
AUTHOR = 'EAP'

if AUTHOR == 'EAP':

    data = eap_char_list # our text data in 'by character' format

    char_to_ix = eap_char_to_ix # our dictionary



ix_to_char = dict (zip(char_to_ix.values(),char_to_ix.keys())) # reverse of our dictionary

VOCAB_SIZE = len(char_to_ix) # length of our dictionary

SEQ_LENGTH = 150

#SEQ_LENGTH = (len(eap_char_list) + len(mws_char_list) + len(hpl_char_list)) // train_sample.shape[0] # how long to make a sentence

NUM_SEQUENCES = len(data)//SEQ_LENGTH # number of sequences we're training on



## Making a dataset to train on from what we have so far

X = np.zeros((NUM_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))

y = np.zeros((NUM_SEQUENCES, SEQ_LENGTH, VOCAB_SIZE))

for i in range(0, NUM_SEQUENCES):

    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]

    X_sequence_ix = [char_to_ix[value] for value in X_sequence]

    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))

    for j in range(SEQ_LENGTH):

        input_sequence[j][X_sequence_ix[j]] = 1.

    X[i] = input_sequence



    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]

    y_sequence_ix = [char_to_ix[value] for value in y_sequence]

    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))

    for j in range(SEQ_LENGTH):

        target_sequence[j][y_sequence_ix[j]] = 1.

    y[i] = target_sequence

    

print('AUTHOR: {}'.format(AUTHOR))

print('SEQUENCE LENGTH: {}'.format(SEQ_LENGTH))

print('NUM SEQUENCES: {}'.format(NUM_SEQUENCES))

print('VOCAB SIZE: {}'.format(VOCAB_SIZE))
print('X: type-{}, shape-{}'.format(type(X), X.shape))

print('y: type-{}, shape-{}'.format(type(y), y.shape))
print("X shape - ", X[0,:,:].shape, 'getting col sums...')

print(X[0,:,:].sum(0), '\ngetting row sums...\n', X[0,:,:].sum(1))



print("y shape - ", y[0,:,:].shape, 'getting col sums...')

print(y[0,:,:].sum(0), '\ngetting row sums...\n', y[0,:,:].sum(1))
X_seq = []

for row in X[0,:,:]: # iterate over the 150 rows in this sequence

    ix = np.argmax(row) # gets the index of the max value in the row (will be 1 and the rest 0)

    char = ix_to_char[ix] # looks in our dictionary to switch from index to character

    X_seq.append(char) # appends the character to our list

X_seq = ''.join(X_seq) # turns our list into a string



y_seq = []

for row in y[0,:,:]: # iterate over the 150 rows in this sequence

    ix = np.argmax(row) # gets the index of the max value in the row (will be 1 and the rest 0)

    char = ix_to_char[ix] # looks in our dictionary to switch from index to character

    y_seq.append(char) # appends the character to our list

y_seq = ''.join(y_seq) # turns our list into a string

print('X_seq: %s'%X_seq)

print('\ny_seq: %s'%y_seq)

print('\nCapital T appears 1 time in X and 0 times in y')

print('Capital I appears 1 time in X and 2 times in y')
model = get_model(learning_rate = .003)

model.summary()
print("Building Edgar Allen Poe model, taking into account previous {} characters...".format(SEQ_LENGTH))

train_model(model, X, y, 30, 5) # train 30 epochs, will print/save every 5 epochs
generate_text(model, 150) # Generate random text of length 150
generate_text(model, 150) # Generate random text of length 150