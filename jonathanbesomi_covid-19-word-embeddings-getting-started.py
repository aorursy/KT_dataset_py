from gensim.models import Word2Vec

WV_PATH = '../input/covid19-word-embeddings/CORD19_word2vec_13032020_200.model'



model_wv = Word2Vec.load(WV_PATH)

model_wv.wv.most_similar("coronavirus")