import gensim

model = gensim.models.Word2Vec.load("/kaggle/input/urdu-word2vec/urdu-w2vec")



print(model)
model.most_similar('چاند')

model.most_similar('محبت')

model.most_similar('کراچی')

import gensim

model = gensim.models.Word2Vec.load("/kaggle/input/urdu-word2vec/urdu-w2vec-big")



print(model)
word_add = ['عورت', 'بادشاہ']

word_sub = ['مرد']

model.most_similar(positive=word_add, negative=word_sub)