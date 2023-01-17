# Example for https://www.kaggle.com/maxsod/fasttext-aligned-word-vectors

import os

import gensim



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading 200000 most common words for English (limit for loading time)

model = gensim.models.KeyedVectors.load_word2vec_format('/kaggle/input/wiki.en.align.vec', limit=200000)
# Word vector for the word "test"

model['test']
# Most similar words to "test"

model.most_similar('test')