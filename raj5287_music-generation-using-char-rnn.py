import os

import json

import argparse



import numpy as np

from keras.models import Sequential, load_model

from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
file1 = open("../input/input.txt","r")

text=file1.read()
#saving the model i.e. weights to use it for music generation

MODEL_DIR = '../model'

def save_weights(epoch, model):

    if not os.path.exists(MODEL_DIR):

        os.makedirs(MODEL_DIR)

    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))
#load the weights

def load_weights(epoch, model):

    model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))
BATCH_SIZE = 207

SEQ_LENGTH = 64



def read_batches(T, vocab_size):

    length = T.shape[0]; #1,684,914

    batch_chars = int(length / BATCH_SIZE); # 8104



    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH): # (0, 8040, 64)

        X = np.zeros((BATCH_SIZE, SEQ_LENGTH)) # 16X64

        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size)) # 16X64X95

        for batch_idx in range(0, BATCH_SIZE): # (0,16)

            for i in range(0, SEQ_LENGTH): #(0,64)

                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] # 

                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1

        yield X, Y
#building the model here with 4 LSTM layers and a Time distributed Layer

#A softmax layer @end to get probablities charater-wise

def build_model(batch_size, seq_len, vocab_size):

    model = Sequential()

    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))

    for i in range(4):

        model.add(LSTM(256, return_sequences=True, stateful=True))

        model.add(Dropout(0.2))



    model.add(TimeDistributed(Dense(vocab_size))) 

    model.add(Activation('softmax'))

    return model

def train(text, epochs, save_freq):



    # character to index and vice-versa mappings

    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }

    print("Number of unique characters: " + str(len(char_to_idx))) #95

    

    with open('char_to_idx.json', 'w') as f:

        json.dump(char_to_idx, f)



    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }

    vocab_size = len(char_to_idx)



    #model_architecture

    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])





    #Train data generation

    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32) #convert complete text into numerical indices



    print("Length of text:" + str(T.size)) #129,665



    steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH  





    for epoch in range(epochs):

        print('\nEpoch {}/{}'.format(epoch + 1, epochs))

        

        losses, accs = [], []



        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):

            

            #print(X);



            loss, acc = model.train_on_batch(X, Y)

            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))

            losses.append(loss)

            accs.append(acc)

    if (epoch + 1) % save_freq == 0:

            save_weights(epoch + 1, model)

            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))
epochs=200

save_freq=40

train(text,epochs,save_freq)
def build_sample_model(vocab_size):

    model = Sequential()

    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))

    for i in range(4):

        model.add(LSTM(256, return_sequences=(i != 3), stateful=True))

        model.add(Dropout(0.2))



    model.add(Dense(vocab_size))

    model.add(Activation('softmax'))

    return model
#generating new music by sampling

def sample(epoch, header, num_chars):

    with open('char_to_idx.json') as f:

        char_to_idx = json.load(f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }

    vocab_size = len(char_to_idx)



    model = build_sample_model(vocab_size)

    load_weights(epoch, model)

    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))



    sampled = [char_to_idx[c] for c in header]

    print(sampled)

    



    for i in range(num_chars):

        batch = np.zeros((1, 1))

        if sampled:

            batch[0, 0] = sampled[-1]

        else:

            batch[0, 0] = np.random.randint(vocab_size)

        result = model.predict_on_batch(batch).ravel()

        sample = np.random.choice(range(vocab_size), p=result)

        sampled.append(sample)



    return ''.join(idx_to_char[c] for c in sampled)
epoch=200 #'epoch checkpoint to sample from'

seed='' #'initial character to start for the generated text'

length=512 #'number of characters to sample'

print(sample(epoch,seed,length))