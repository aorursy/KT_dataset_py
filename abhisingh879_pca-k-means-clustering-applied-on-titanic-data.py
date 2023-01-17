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
titanic_df = pd.read_csv("../input/train.csv")

titanic_df = titanic_df.drop(["Name","Ticket","Cabin"],axis=1)

# Remove Row which are having NAN values

titanic_df.dropna(axis = 1, how ='all', inplace = True)

# To check which coulumn has NAN 

titanic_df.isnull().any()

#Replaced Nan values in Age column with it's Mean

titanic_df.ix[:,4]=(titanic_df.ix[:,4]).fillna(titanic_df.mean(axis=0)[3])
def binarize(df):

    for col in ['Sex']:

        df[col] = df[col].map({'female':1, 'male':0})

    return df

def binarize_Embark(df):

    for col in ['Embarked']:

        df[col] = df[col].map({'S':0, 'C':1,'Q':2})

    return df

def label_Survived(df):

    for col in ['Survived']:

        df[col] = df[col].map({0:'NotSurvived', 1:'Survived'})

    return df

titanic_df = binarize(titanic_df)

titanic_df = binarize_Embark(titanic_df)

titanic_df = label_Survived(titanic_df)
x = titanic_df.ix[:,2:8].values

y = titanic_df.ix[:,1].values
# we are using function StandardScaler() function for standarizing the Matrix.

from sklearn.preprocessing import StandardScaler

std = StandardScaler()

std
std.fit(x)
X_std = std.transform(x)
# Below is standard mean & variance featurewise

std.mean_, std.var_
# we can directly use below funtion call in order to get our Matrix standarize

#X_std = StandardScaler().fit_transform(x)

#X_std
mean_vec = np.mean(X_std, axis=0)

#mean_vec = mean_vec[~pd.isnull(mean_vec)]

#mean_vec

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

print('Covariance matrix \n%s' %cov_mat)
#We can get Covarience matrix by direclty using below Method instead of doing like blove result will be same

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
cov_mat = np.cov(X_std.T)

cov_mat



eig_vals, eig_vecs = np.linalg.eig(cov_mat)

np.linalg.eig

print('Eigenvectors \n%s' %eig_vecs)

print('\nEigenvalues \n%s' %eig_vals)
u,s,v = np.linalg.svd(X_std.T)

u
s
v
for ev in eig_vecs:

    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

print('Everything ok!')
import plotly.plotly as py

from plotly.graph_objs import *

import plotly.tools as tls



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

%matplotlib inline

init_notebook_mode(connected=True)
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

#eig_pairs[1][1]

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



trace1 = Bar(

        x=['PC %s' %i for i in range(1,7)],

        y=var_exp,

        showlegend=False)



trace2 = Scatter(

        x=['PC %s' %i for i in range(1,7)], 

        y=cum_var_exp,

        name='cumulative explained variance')



data = Data([trace1,trace2])



layout=Layout(

        yaxis=YAxis(title='Explained variance in percent'),

        title='Explained variance by different principal components')



fig = Figure(data=data, layout=layout)

iplot(fig)
matrix_w = np.hstack((eig_pairs[0][1].reshape(6,1), 

                      eig_pairs[1][1].reshape(6,1)))

print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)
Y.shape
X_std.shape
traces = []



for name in ('NotSurvived','Survived'):



    trace = Scatter(

        x=Y[y==name,0],

        y=Y[y==name,1],

        mode='markers',

        name=name,

        marker=Marker(

            size=10,

            line=Line(

                color='rgba(217, 217, 217, 0.14)',

                width=0.5),

            opacity=0.8))

    traces.append(trace)





data = Data(traces)

layout = Layout(showlegend=True,

                scene=Scene(xaxis=XAxis(title='PC1'),

                yaxis=YAxis(title='PC2'),))



fig = Figure(data=data, layout=layout)

iplot(fig)
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=3)

Y_sklearn = sklearn_pca.fit_transform(X_std)
sklearn_pca.explained_variance_ratio_
Y_sklearn.shape
sklearn_pca.transform()
traces = []



for name in ('NotSurvived','Survived'):



    trace = Scatter(

        x=Y_sklearn[y==name,0],

        y=Y_sklearn[y==name,1],

        mode='markers',

        name=name,

        marker=Marker(

            size=10,

            line=Line(

                color='rgba(217, 217, 217, 0.14)',

                width=0.5),

            opacity=0.8))

    traces.append(trace)





data = Data(traces)

layout = Layout(showlegend=True,

                scene=Scene(xaxis=XAxis(title='PC1'),

                yaxis=YAxis(title='PC2'),))



fig = Figure(data=data, layout=layout)

iplot(fig)
%matplotlib inline



from sklearn import datasets, cluster

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
np.random.seed(2)

k_means = cluster.KMeans(n_clusters=2)

k_means.fit(Y) 

labels = k_means.labels_

v

# check how many of the samples were correctly labeled

correct_labels = sum(y_var == labels)



print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y_var.size))
# plot the clusters in color

fig = plt.figure(1, figsize=(8, 8))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)

plt.cla()



ax.scatter(Y[:, 0], Y[:, 1], c=labels.astype(np.float))



ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.set_xlabel('NotSurvived')

ax.set_ylabel('Survived')





plt.show()



traces = []



for tempvar in ('NotSurvived','Survived'):

    trace = Scatter(

        x=Y[y==tempvar,0],

        y=Y[y==tempvar,1],

        mode='markers',

        name=name,

        marker=Marker(

            size=10,

            line=Line(

                color='rgba(217, 217, 217, 0.14)',

                width=0.5),

            opacity=0.8))

    traces.append(trace)

data = Data(traces)

layout = Layout(showlegend=True,

                scene=Scene(xaxis=XAxis(title='AxisX'),

                yaxis=YAxis(title='AxisY'),))



fig = Figure(data=data, layout=layout)

iplot(fig)