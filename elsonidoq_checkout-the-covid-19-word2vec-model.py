from gensim.models import Word2Vec



model = Word2Vec.load('/kaggle/input/covid19-challenge-trained-w2v-model/covid.w2v')
model.wv.most_similar('coronavirus', topn=20)
model.wv.most_similar('transmission', topn=20)