# 如果没有安装 gensim, sklearn 和 numpy 库

# 请使用 pip install gensim scikit-learn numpy 安装

import plotly

import sklearn.decomposition

import gensim

import numpy

import jieba

import itertools

from collections import Counter
fp=open('../input/hhhlllmmm/HongLouMeng_word2vec_1.txt','r')

file=fp.readlines()

for line in file:

    line=line.strip('\n') 
text=' '.join(file)

split_mode='jieba'

if split_mode=='char':

    token_list=[char for char in text]

elif split_mode=='jieba':

    token_list=[word for word in jieba.cut(text)]

text_cut=jieba.cut(text)

text_cut=' '.join(text_cut)
print(text_cut[:-1])
# 读入语料

#sentences = gensim.models.word2vec.Text8Corpus("../input/tlmmmm/HongLouMeng_word2vec2.txt")

sentences=text_cut

# 训练word2vec模型

# size --> dim of word2vec

model = gensim.models.word2vec.Word2Vec(sentences, size=300)

# 保存模型

model.save("poetry.w2v")
print(type(model))
model = gensim.models.Word2Vec.load("./" + "poetry.w2v")

# 装载词向量

all_word_vector = model[model.wv.vocab]
'''

胭:https://www.kaggleusercontent.com/kf/18889669/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..NtenmeIibrCKRzf7QJPvTg.SwO0flI2xoDsXnLMwP4CneZGfbb9w0p6bwwXBqj19t9oTx7SQSCTrBxGVe28Asu9STbqT523ragVp7kDjCiY_CKJ9AKnkp-r7EXh-tZ-vB1Iy_dzuQQOXhWsdI-uGgazWO7QhyZmZWO0q9CZ6b2GZw.toyMLmTHojT4LnOq7gajLg/word2vec.html

亭:https://www.kaggleusercontent.com/kf/18889920/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..APQRiQjvatVf-4sevptaSg.RkHPSTXMQJghsYp_x0-E3BONOAwpdccju9hqpa7gbi3-ptEXBnsV1j9tx5eoVTbseBF6Sk3xQBw_7SPl3IwG3V2DyJgsbFzBrz2c1LW3ToI7MSVVB5vTfw8UPUltt1fkFgkFoF0OCKTsliR0Ty0Ncw.8xmXA8wjxBcrkz5nQeKJrw/word2vec.html

楼:https://www.kaggleusercontent.com/kf/18892167/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..odeRQDOPtGKd9Af-NlTU-g.Wy2vbaL1S0FV_mpCs01m94bsXmdhozeQ5JVj7_4DeB_ASENmSS3VmY9zqJk_9IHpSSIUgEEilre7e7H6UUGrUuhqU_4i4lmDDG6bN6KUyVEaluX_iQOcah49vNw2cTMGTGU7mXFb1ufLK2AwM0fUFA.EKB8sdgMHN2iK7gmPbzSyg/word2vec.html



'''
jieba.suggest_freq('胭脂',True)
start_word = '山'

topn = 50

pca = sklearn.decomposition.PCA(n_components=3)

pca.fit(all_word_vector)

# 收集与start_word最相似的词向量

similar_word_list = [start_word] + [pair[0] for pair in model.most_similar(start_word, topn=topn)]

similar_word_vector =  [model[word] for word in similar_word_list]

# 降维

decomposed_vector = pca.transform(similar_word_vector)
print(model.most_similar)
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
model.most_similar(positive=['水','火'], negative=['冰'])
