import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



# Import the 3 dimensionality reduction methods

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
df = pd.read_csv('../input/iris-dataset/iris.data.csv', header=None, sep=',')
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']

df.dropna(how="all", inplace=True) # drops the empty line at file-end



df.tail()
# split data table into data X and class labels y



X = df.iloc[:,0:4].values

y = df.iloc[:,4].values




# plotting histograms

data = []



legend = {0:False, 1:False, 2:False, 3:True}



colors = {'Iris-setosa': '#0D76BF', 

          'Iris-versicolor': '#00cc96', 

          'Iris-virginica': '#EF553B'}



for col in range(4):

    for key in colors:

        trace = dict(

            type='histogram',

            x=list(X[y==key, col]),

            opacity=0.75,

            xaxis='x%s' %(col+1),

            marker=dict(color=colors[key]),

            name=key,

            showlegend=legend[col]

        )

        data.append(trace)



layout = dict(

    barmode='overlay',

    xaxis=dict(domain=[0, 0.25], title='sepal length (cm)'),

    xaxis2=dict(domain=[0.3, 0.5], title='sepal width (cm)'),

    xaxis3=dict(domain=[0.55, 0.75], title='petal length (cm)'),

    xaxis4=dict(domain=[0.8, 1], title='petal width (cm)'),

    yaxis=dict(title='count'),

    title='Distribution of the different Iris flower features'

)



fig = dict(data=data, layout=layout)

#py.iplot(fig, filename='exploratory-vis-histogram')
from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)


import numpy as np

mean_vec = np.mean(X_std, axis=0)

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)


print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

cov_mat = np.cov(X_std.T)



eig_vals, eig_vecs = np.linalg.eig(cov_mat)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
cor_mat1 = np.corrcoef(X_std.T)



eig_vals, eig_vecs = np.linalg.eig(cor_mat1)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
cor_mat2 = np.corrcoef(X.T)



eig_vals, eig_vecs = np.linalg.eig(cor_mat2)



print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
u,s,v = np.linalg.svd(X_std.T)

u


for ev in eig_vecs:

    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

print('Everything ok!')
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort()

eig_pairs.reverse()



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)



trace1 = dict(

    type='bar',

    x=['PC %s' %i for i in range(1,5)],

    y=var_exp,

    name='Individual'

)



trace2 = dict(

    type='scatter',

    x=['PC %s' %i for i in range(1,5)], 

    y=cum_var_exp,

    name='Cumulative'

)



data = [trace1, trace2]



layout=dict(

    title='Explained variance by different principal components',

    yaxis=dict(

        title='Explained variance in percent'

    ),

    annotations=list([

        dict(

            x=1.16,

            y=1.05,

            xref='paper',

            yref='paper',

            text='Explained Variance',

            showarrow=False,

        )

    ])

)

fig = dict(data=data, layout=layout)

#py.iplot(fig, filename='')
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 

                      eig_pairs[1][1].reshape(4,1)))



print('Matrix W:\n', matrix_w)


Y = X_std.dot(matrix_w)
data = []



for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):

    trace = dict(

        type='scatter',

        x=Y[y==name,0],

        y=Y[y==name,1],

        mode='markers',

        name=name,

        marker=dict(

            color=col,

            size=12,

            line=dict(

                color='rgba(217, 217, 217, 0.14)',

                width=0.5),

            opacity=0.8)

    )

    data.append(trace)



layout = dict(

    showlegend=True,

    scene=dict(

        xaxis=dict(title='PC1'),

        yaxis=dict(title='PC2')

    )

)



fig = dict(data=data, layout=layout)

#py.iplot(fig, filename='projection-matrix')
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)

Y_sklearn = sklearn_pca.fit_transform(X_std)
data = []



for name, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), colors.values()):



    trace = dict(

        type='scatter',

        x=Y_sklearn[y==name,0],

        y=Y_sklearn[y==name,1],

        mode='markers',

        name=name,

        marker=dict(

            color=col,

            size=12,

            line=dict(

                color='rgba(217, 217, 217, 0.14)',

                width=0.5),

            opacity=0.8)

    )

    data.append(trace)



layout = dict(

        xaxis=dict(title='PC1', showline=False),

        yaxis=dict(title='PC2', showline=False)

)

fig = dict(data=data, layout=layout)

#py.iplot(fig, filename='pca-scikitlearn')