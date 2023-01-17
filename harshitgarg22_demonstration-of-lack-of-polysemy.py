import numpy as np



# Get the interactive Tools for Matplotlib

%matplotlib notebook

import matplotlib.pyplot as plt

plt.style.use('ggplot')



from sklearn.decomposition import PCA



from gensim.test.utils import datapath, get_tmpfile

from gensim.models import KeyedVectors

from gensim.scripts.glove2word2vec import glove2word2vec
glove_file = datapath('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')

word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")

glove2word2vec(glove_file, word2vec_glove_file)
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
print(*model.most_similar('bank', topn = 20), sep = '\n')