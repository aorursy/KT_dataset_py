from tensorflow.keras.utils import to_categorical

import numpy as np

# In this code we will try to recreate

# Encoder-Decoder architecture in "Encoder_decoder_reversing.png"

# To get the idea of ending-decoding without using machine learning
def words2onehot(word_list, word2index):

  

  # Convert words to word IDs..

  word_ids = [word2index[w] for w in word_list]

  # Convert word IDs to onehot vectors and return the onehot array

  onehot = to_categorical(word_ids, num_classes=3)

  return onehot



def encoder(onehot):

  # Get word IDs from onehot vectors and return the IDs

  word_ids = np.argmax(onehot, axis=1)

  return word_ids

# Define the onehot2words function that returns words for a set of onehot vectors

def onehot2words(onehot, index2word):

  ids = np.argmax(onehot, axis=1)

  res = [index2word[id] for id in ids]

  return res



# Define the decoder function that returns reversed onehot vectors

def decoder(context_vector):

  word_ids_rev = context_vector[::-1]

  onehot_rev = to_categorical(word_ids_rev, num_classes=3)

  return onehot_rev
if __name__ == '__main__':

    

    word2index = {'We': 0, 'dogs': 2, 'like': 1}



    # Define "We like dogs" as words

    words = ["We", "like", "dogs"]

    # Convert words to onehot vectors using words2onehot

    onehot = words2onehot(words, word2index)

    # Get the context vector by using the encoder function

    context = encoder(onehot)

    print(context)



    

    index2word = {0: 'We', 1: 'like', 2: 'dogs'}

    # Convert context to reversed onehot vectors using decoder

    onehot_rev = decoder(context)

    # Get the reversed words using the onehot2words function

    reversed_words = onehot2words(onehot_rev, index2word)

    print(reversed_words)
