# Using GloVe vectors in word2vec format in Kaggle kernels
from gensim.test.utils import get_tmpfile

from gensim.models import KeyedVectors
vectors = KeyedVectors.load_word2vec_format("../input/glove_w2v.txt") # import the data file
vectors.similarity('dog', 'princess')
result = vectors.similar_by_word("patriots")

print("{}: {:.4f}".format(*result[0]))