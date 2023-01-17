# 如果没有安装 gensim, sklearn 和 numpy 库

# 请使用 pip install gensim scikit-learn numpy 安装

import plotly

import sklearn.decomposition

import gensim

import numpy
model = gensim.models.Word2Vec.load("../input/wikiw2v/wiki.w2v")

# 装载词向量

all_word_vector = model[model.wv.vocab]
model.most_similar('banana')
model.most_similar(positive=['prince', 'woman'], negative=['man'])
model.most_similar(positive=['Japanese', 'China'], negative=['Chinese'])
model.most_similar(positive=['strong', 'worse'], negative=['bad'])
model.most_similar(positive=['tall', 'strongest'], negative=['strong'])
start_word = 'China'

topn = 49

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