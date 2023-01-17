from gensim.models import KeyedVectors



# As of Gensim 3.7.3 it's using some deprecated function and we don't care about it

import warnings

warnings.filterwarnings("ignore")
%timeit -n 30 model = KeyedVectors.load("../input/glove.twitter.27B.200d.gensim")
%timeit -n 30 model = KeyedVectors.load("../input/glove.twitter.27B.200d.gensim", mmap="r")
model.most_similar("good")
model.most_similar(positive=["woman", "king"], negative=["man"])
model["good"]