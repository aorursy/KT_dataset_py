# 如果没有安装 gensim, sklearn 和 numpy 库

# 请使用 pip install gensim scikit-learn numpy 安装

import plotly

import sklearn.decomposition

import gensim

import numpy
# 读入语料

sentences = gensim.models.word2vec.Text8Corpus("../input/wikisent2.txt")

# 训练word2vec模型

# size --> dim of word2vec

model = gensim.models.word2vec.Word2Vec(sentences, size=300)

# 保存模型

model.save("/kaggle/working/wiki.w2v")
model = gensim.models.Word2Vec.load("/kaggle/working/wiki.w2v")

# 装载词向量

all_word_vector = model[model.wv.vocab]
model.most_similar('banana')
model.most_similar(positive=['woman', 'king'], negative=['man'])