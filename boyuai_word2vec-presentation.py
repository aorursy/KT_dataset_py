# 如果没有安装 gensim, sklearn 和 numpy 库

# 请使用 pip install gensim scikit-learn numpy 安装

import plotly

import sklearn.decomposition

import gensim

import numpy
model = gensim.models.Word2Vec.load("../input/wikiw2v/wiki.w2v")
model.most_similar('banana')
model.most_similar(positive=['prince', 'woman'], negative=['man'])
model.most_similar(positive=['French', 'China'], negative=['Chinese'])
model.most_similar(positive=['Japanese', 'China'], negative=['Chinese'])
model.most_similar(positive=['strong', 'worse'], negative=['bad'])
model.most_similar(positive=['strong', 'worst'], negative=['bad'])
model.most_similar(positive=['bird', 'sea'], negative=['sky'])
model.most_similar(positive=['boy', 'mother'], negative=['father'])