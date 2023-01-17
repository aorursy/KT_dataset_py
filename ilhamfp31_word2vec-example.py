import gensim

DIR_DATA_MISC = "../input/word2vec-100-indonesian"

path = '{}/idwiki_word2vec_100.model'.format(DIR_DATA_MISC)

id_w2v = gensim.models.word2vec.Word2Vec.load(path)
id_w2v['makan']
word = "makan"  # for any word in model

index = id_w2v.wv.vocab.get(word).index

print(index)
id_w2v.wv.index2word[255]
id_w2v.wv.index2word[index] == word
print(id_w2v.most_similar('makan'))