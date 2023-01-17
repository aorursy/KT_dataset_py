#导入数据处理包

import numpy as np

import pandas as pd

#导入绘图包

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline

#导入sklearn工具

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#读取数据集

train = pd.read_csv('../input/train.csv')

train.head()
#抽取真实值梭所在的列

target = train['label']

#将真实值所在的列删除后作为训练数据

train = train.drop('label',axis=1)
#标准化数据集

from sklearn.preprocessing import StandardScaler

X = train.values

X_std = StandardScaler().fit_transform(X)



#计算特征向量与特征值之间的协方差矩阵

mean_vec = np.mean(X_std,axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#创建一个（特征向量，特征值）的元组

eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

#对每对特征向量，特征值从高到低排序

eig_pairs.sort(key=lambda x:x[0],reverse=True)

#根据特征值计算解释性方差，方差值越大，说明特征越重要

tot = sum(eig_vals)

#单个解释性方差

var_exp = [(i/tot)*100 for i in sorted(eig_vals,reverse=True)]

#累计解释性方差

cum_vaar_exp = np.cumsum(var_exp)
trace1 = go.Scatter(

    x=list(range(784)),

    y = cum_vaar_exp,

    mode = 'lines+markers',

    name = "'Cumulative Explained Variance'",

    line = dict(

        shape='spline',

        color='goldenrod'))

trace2 = go.Scatter(

    x=list(range(784)),

    y = var_exp,

    mode='lines+markers',

    name="'Individual Explained Variance'",

    line=dict(

        shape='linear',

        color='black'))

fig = tls.make_subplots(insets=[{'cell':(1,1),'l':0.7,'b':0.5}],print_grid=True)

fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,1)

fig.layout.title='Explained Variance plots - Full and Zoomed-in'

fig.layout.xaxis=dict(range=[0,80],title='Feature columns')

fig.layout.yaxis=dict(range=[0,60],title='Explained Variance')

py.iplot(fig,filename='styled-scatter')
#sklearn的PCA方法

n_components = 30

pca = PCA(n_components=n_components).fit(train.values)

#抽取PCA的组件值-特征值

eigenvalues = pca.components_.reshape(n_components,28,28)

eigenvalues = pca.components_
n_row = 4

n_col = 7



plt.figure(figsize=(13,12))

for i in list(range(n_row*n_col)):

    offset = 0

    plt.subplot(n_row,n_col,i+1)

    plt.imshow(eigenvalues[i].reshape(28,28),cmap='jet')

    title_text = 'Eigenvalue'+str(i+1)

    plt.title(title_text,size=6.5)

    plt.xticks(())

    plt.yticks(())

plt.show()
plt.figure(figsize=(14,12))

for digit_num in range(0,70):

    plt.subplot(7,10,digit_num+1)

    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)

    plt.imshow(grid_data,interpolation='none',cmap='afmhot')

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()
#删除前面创建的变量

del X



X = train[:6000].values



#删除前面创建的变量

del train



#标准化数据

X_std = StandardScaler().fit_transform(X)

#使用5个组件进行PCA降维

pca = PCA(n_components=5)

pca.fit(X_std)

X_5d = pca.transform(X_std)



Target = target[:6000]
trace0 = go.Scatter(

    x = X_5d[:,0],

    y = X_5d[:,1],

    mode = 'markers',

    text = Target,

    showlegend=False,

    marker=dict(

        size = 8,

        color=Target,

        colorscale='jet',

        showscale=False,

        line = dict(

            width = 2,

            color='rgb(255,255,255)'),

        opacity = 0.8))



data = [trace0]



layout = go.Layout(

    title='Principal Component Analysize(PCA)',

    hovermode='closest',

    xaxis=dict(

        title='First Principal Component',

        ticklen=5,

        zeroline=False,

        gridwidth=2,),

    yaxis=dict(

        title='Second Principal Component',

        ticklen=5,

        gridwidth=2,),

    showlegend=True)



fig = dict(data=data,layout=layout)

py.iplot(fig,filename='styled-scatter')
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=9)

x_clustered = kmeans.fit_predict(X_5d)



trace_Kmeans = go.Scatter(x=X_5d[:,0],y=X_5d[:,1],mode='markers',

                         showlegend=False,

                         marker=dict(

                             size=8,

                             color=x_clustered,

                             colorscale='Portland',

                             showscale=False,

                             line=dict(

                                 width=2,

                                 color='rgb(255,255,255)'

                             )))

layout = go.Layout(

    title='KMeans Clustering',

    hovermode='closest',

    xaxis=dict(

        title='First Principal Compoonent',

        ticklen=5,

        zeroline=False,

        gridwidth=2,),

    yaxis=dict(

        title='Second Principal Component',

        ticklen=5,

        gridwidth=2,

    ),

    showlegend=True)

data = [trace_Kmeans]

fig1 = dict(data=data,layout=layout)

py.iplot(fig1,filename='svm')
from IPython.display import display,Math,Latex
lda = LDA(n_components=5)

X_LDA_2D = lda.fit_transform(X_std,Target.values)
traceLDA = go.Scatter(

    x = X_LDA_2D[:,0],

    y = X_LDA_2D[:,1],

    mode = 'markers',

    text = Target,

    showlegend=True,

    marker = dict(

        size = 8,

        color = Target,

        colorscale = 'Jet',

        showscale = False,

        line = dict(

            width=2,

            color='rgb(255,255,255)'),

    opacity=0.8))



data = [traceLDA]



layout = go.Layout(

    title="Linear Discrimiant Analysis(LDA)",

    hovermode = 'closest',

    xaxis=dict(

        title='First Linear Discriminant',

        ticklen=5,

        zeroline=False,

        gridwidth=2,),

    yaxis=dict(

        title='Second Linear Discrimiant',

        ticklen=5,

        gridwidth=2,),

    showlegend=False)



fig = dict(data=data,layout=layout)

py.iplot(fig,filename='style-scatter')
tsne = TSNE(n_components=2)

tsne_results = tsne.fit_transform(X_std)
traceTSNE = go.Scatter(

    x = tsne_results[:,0],

    y = tsne_results[:,1],

    mode='markers',

    showlegend=True,

    marker=dict(

        size = 8,

        color = Target,

        colorscale='Jet',

        line = dict(

            width = 2,

            color = 'rgb(255,255,255)'),

        opacity=0.8))



data = [traceTSNE]



layout = dict(title="TSNE(T-Distributed Stochastic Neighbour Embedding)",

             hovermode = 'closest',

             yaxis = dict(zeroline=False),

             xaxis = dict(zeroline=False),

             showlegend=False)



fig = dict(data=data,layout=layout)

py.iplot(fig,filename='style-scatter')