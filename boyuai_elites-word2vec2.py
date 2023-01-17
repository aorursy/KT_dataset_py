import plotly

import sklearn.decomposition

import gensim

import numpy

import jieba

import itertools

from collections import Counter
# 读入语料

sentences = gensim.models.word2vec.Text8Corpus('/kaggle/input/word2vec-data/text8/text8')

# 训练word2vec模型

# size --> dim of word2vec

model = gensim.models.word2vec.Word2Vec(sentences, size=300)

# 保存模型

model.save("text8.w2v")
model = gensim.models.Word2Vec.load("text8.w2v")

# 装载词向量

all_word_vector = model[model.wv.vocab]
start_word = 'apple'

topn = 50

pca = sklearn.decomposition.PCA(n_components=3)

pca.fit(all_word_vector)

# 收集与start_word最相似的词向量

similar_word_list = [start_word] + [pair[0] for pair in model.most_similar(start_word, topn=topn)]

similar_word_vector =  [model[word] for word in similar_word_list]

# 降维

decomposed_vector = pca.transform(similar_word_vector)
# 设置坐标图中画出的点的坐标，文本标注的位置和颜色

trace = plotly.graph_objs.Scatter3d(

    x=decomposed_vector[:, 0],

    y=decomposed_vector[:, 1],

    z=decomposed_vector[:, 2],

    mode="markers+text",

    text=similar_word_list,

    textposition="bottom center",

    marker=dict(

        color=[256 - int(numpy.linalg.norm(decomposed_vector[i] - decomposed_vector[0])) for i in range(len(similar_word_list))]

    )

)

layout = plotly.graph_objs.Layout(

    title="Top " + str(topn) + " Word Most Similar With \"" + start_word + "\""

)

data = [trace]

figure = plotly.graph_objs.Figure(data=data, layout=layout)

graph_name = "word2vec.html"

# 绘图

plotly.offline.plot(figure, filename=graph_name, auto_open=False)
model.most_similar(positive=['does','have'], negative=['do'])
model.most_similar(positive=['woman', 'king'], negative=['man'])