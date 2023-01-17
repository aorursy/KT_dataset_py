# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import plotly.offline as py
py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
%matplotlib inline


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
train=pd.read_csv('../input/train.csv')

train.shape

train.head()
# save the labels to a Pandas series target
target = train['label']
# Drop the label feature
train.drop("label",axis=1,inplace=True)
# Standardize the data
from sklearn.preprocessing import StandardScaler
X = train.values
X_std = StandardScaler().fit_transform(X)

# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(X_std, axis=0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
#PCA analysis

#standardizing the data

from sklearn.preprocessing import StandardScaler
X = train.values
X_std = StandardScaler().fit_transform(X)

mean_vec=np.mean(X_std,axis=0)
cov_mat=np.cov(X_std.T)
eigvalues ,eigvectors =np.linalg.eig(cov_mat)

eigpairs=[(np.abs(eigvalues[i]),eigvectors[:,i] )for i in range(len(eigvalues))]


eigpairs.sort(key=lambda x:x[0],reverse=True)
   
tot=sum(eigvalues)
var_exp=[(i/tot)*100 for i in sorted(eigvalues,reverse=True)]
cum_var_exp=np.cumsum(var_exp)

#Using plotly to visualise individual explained variance and cummulative explained variance


trace1 = go.Scatter(
    x=list(range(784)),
    y= cum_var_exp,
    mode='lines+markers',
    name="'Cumulative Explained Variance'",
   
    line = dict(
        shape='spline',
        color = 'goldenrod'
    )
)
trace2 = go.Scatter(
    x=list(range(784)),
    y= var_exp,
    mode='lines+markers',
    name="'Individual Explained Variance'",
 
     line = dict(
        shape='linear',
        color = 'black'
    )
)
fig = tls.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.5}],
                          print_grid=True)

fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,1)


fig.layout.title='explained Variance plots'
fig.layout.xaxis=dict(range=[0,800],title='Feature columns')
fig.layout.yaxis=dict(range=[0,100],title='explained variance')


py.iplot(fig,filename='inset example')

#We see that nearly 90% of the explained variance can be explained by 200 features
pca=PCA(30)
pca.fit(X_std)

X_pca=pca.transform(X_std)
X_pca.shape
X_std.shape
eigenvalues=pca.components_
eigenvalues.shape



#plotting eigen values
plt.figure(figsize=(13,12))

x_row=4
y_col=7

for i in list(range(x_row*y_col)):
    
    plt.subplot(x_row,y_col,i+1)
    plt.imshow(eigenvalues[i].reshape(28,28),cmap='jet')
    title_='Eigen value'+str(i+1)
    plt.title(title_)
    plt.xticks(())
    plt.yticks(())
plt.show()    
    
#plotting mnist data

plt.figure(figsize=(12,13))

for i in list(range(0,70)):
    plt.subplot(7,10,i+1)
    
    plt.imshow(train.iloc[i].values.reshape(28,28), interpolation = "none", cmap = "copper")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
               
plt.tight_layout
#standardising data and implementing pca

X_=train[:6000].values
X_std_=StandardScaler().fit_transform(X_)
pca_=PCA(5)
X_5d=pca_.fit_transform(X_std_)
Target=target[:6000]
X_5d.shape
eigenvalues_=pca_.components_
eigenvalues_.shape
#visualisation of pca representations

trace = go.Scatter(
    x = X_5d[:,0],
    y = X_5d[:,1],
    name = str(Target),
    
    mode = 'markers',
    text = Target,
    showlegend = False,
    marker = dict(
        size = 8,
        color = Target,
        colorscale ='Jet',
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
        
        opacity = 0.8
    )
)

data=[trace]

layout=go.Layout(title='PCA',
                hovermode='closest',
                xaxis=dict(
                    title='First principal direction',
                    ticklen=5,
                    zeroline=False),
                 yaxis=dict(
                 title='Second principal direction',
                 ticklen=5
            ),
                 showlegend=True
                
                    
                )
fig=dict(data=data,layout=layout)
py.iplot(fig,filename='pca')
#kmeans


from sklearn.cluster import KMeans
kmeans=KMeans(9)
X_clustered=kmeans.fit_predict(X_5d)



tracekmeans = go.Scatter(x=X_5d[:, 0], y= X_5d[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(
                            size=8,
                            color = X_clustered,
                            colorscale = 'Portland',
                            showscale=False, 
                            line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        )
                   ))


layout=go.Layout(title='Kmeans clustering',
                 hovermode='closest',
                 xaxis=dict(title='first principal direction',
                           ticklen=5,
                           zeroline=False,
                           gridwidth=2),
                 yaxis=dict(title='second principal component',
                           ticklen=5,
                           gridwidth=2),
                 showlegend=True
                     )

data = [tracekmeans]
fig1 = dict(data=data, layout= layout)
# fig1.append_trace(contour_list)
py.iplot(fig1, filename="svm")
tracekmeans
#using LDA



lda=LDA(n_components=10)
X_lda = lda.fit_transform(X_std_,Target.values)



traceLDA=go.Scatter(x=X_lda[:,0],
                    y=X_lda[:,1],
                    #name=str(Target),
                    mode='markers',
                    #text=Target,
                    marker=dict(size=8,
                                color=Target,
                                colorscale='jet',
                                showscale=False,
                                line=dict(width=2,
                                         color='rgb(255,255,255)'
                                         ),
                                opacity=0.8
                               )
                )

data=traceLDA

layout=go.Layout(xaxis=dict(title='First Linear discriminant',
                           ticklen=5,
                           gridwidth=2),
                yaxis=dict(title='Second linear discriminant',
                           ticklen=5,
                           gridwidth=2),
                 title='LDA',
                 showlegend=True,
                 hovermode='closest')

fig=dict(data=data,layout=layout)

#py.iplot(fig, filename='styled-scatter')
#py.iplot(fig,filename='LDA')
                
traceLDA
py.iplot(fig,filename='LDA')

