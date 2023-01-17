import gensim
f = open("../input/karate.emb", "r")

embeddings = f.read()

f.close()
embeddings[1:200]
model = gensim.models.KeyedVectors.load_word2vec_format("../input/karate.emb")
model.get_vector("34")
model.most_similar("34")
model.similarity("1", "34")