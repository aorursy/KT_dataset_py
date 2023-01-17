# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings 

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading graph libraries

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline
# Loading the dataset 

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.shape
# Since we have a lot of features (including label) in our dataset, it is a good idea to use dimensionality reduction techniques.

# We will first remove the label 
target = train["label"]

train = train.drop("label", axis=1)
# PCA

# Calculating the eigen vectors 

from sklearn.preprocessing import StandardScaler

X = train.values

X_std = StandardScaler().fit_transform(X)



# Calculating eigenvectors and eigenvalues of covariance matrix

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Creating a list of (eigenvalue, eigenvecs) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
# Sorting the eigenvalue and eigenvecs based on eigen values higher to lower 

eig_pairs.sort(key = lambda x: x[0], reverse= True)
# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
np.percentile(cum_var_exp, 50)
# 50% data points are able to explain 97% of variance
trace1 = go.Scatter(

    x=list(range(784)),

    y= cum_var_exp,

    mode='lines+markers',

    name="'Cumulative Explained Variance'",

#     hoverinfo= cum_var_exp,

    line=dict(

        shape='spline',

        color = 'goldenrod'

    )

)

trace2 = go.Scatter(

    x=list(range(784)),

    y= var_exp,

    mode='lines+markers',

    name="'Individual Explained Variance'",

#     hoverinfo= var_exp,

    line=dict(

        shape='linear',

        color = 'black'

    )

)

fig = tls.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.5}],

                          print_grid=True)



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2,1,1)

fig.layout.title = 'Explained Variance plots - Full and Zoomed-in'

fig.layout.xaxis = dict(range=[0, 800], title = 'Feature columns')

fig.layout.yaxis = dict(range=[0, 100], title = 'Explained Variance')



py.iplot(fig)
# As we can see that more than 90% of variance is exaplained by little over 200 features
# Visualizing the eigen values

# Invoke SKlearn's PCA method

from sklearn.decomposition import PCA



n_components = 30

pca = PCA(n_components=n_components).fit(train.values)



# Extracting the PCA components ( eignevalues )

eigenvalues = pca.components_
# We have taken 30 eigenvalues and now plot them so that we can see the features detected by them 



n_row = 4

n_col = 7



# Plot the first 8 eignenvalues

plt.figure(figsize=(13,12))

for i in list(range(n_row * n_col)):

    offset =0

    plt.subplot(n_row, n_col, i + 1)

    plt.imshow(eigenvalues[i].reshape(28,28), cmap='jet')

    title_text = 'Eigenvalue ' + str(i + 1)

    plt.title(title_text, size=6.5)

    plt.xticks(())

    plt.yticks(())

plt.show()
# Above are the top 30 optimal directions or principal components axes of PCA to generate our dataset.

# We can also observe that eigenvalues 28 is more complex direction to explain the variance of the dataset
# Let's visualize the dataset itself 

# plot some of the numbers

plt.figure(figsize=(14,12))

for digit_num in range(0,70):

    plt.subplot(7,10,digit_num+1)

    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()
# PCA using sklearn

N = 6000

X = train[:N].values

# Standardising the values

X_std = StandardScaler().fit_transform(X)



# Call the PCA method with 5 components. 

pca = PCA(n_components=5)

pca.fit(X_std)

X_5d = pca.transform(X_std)



# For cluster coloring in our Plotly plots, remember to also restrict the target values 

Target = target[:N]
trace0 = go.Scatter(

    x = X_5d[:,0],

    y = X_5d[:,1],

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

data = [trace0]



layout = go.Layout(

    title= 'Principal Component Analysis (PCA)',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)





fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
# K-Mean clustering to identify psosible classes
from sklearn.cluster import KMeans # KMeans clustering 

# Set a KMeans clustering with 9 components ( 9 chosen sneakily ;) as hopefully we get back our 9 class labels)

kmeans = KMeans(n_clusters=9)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(X_5d)



trace_Kmeans = go.Scatter(x=X_5d[:, 0], y= X_5d[:, 1], mode="markers",

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



layout = go.Layout(

    title= 'KMeans Clustering',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)



data = [trace_Kmeans]

fig1 = dict(data=data, layout= layout)

# fig1.append_trace(contour_list)

py.iplot(fig1, filename="svm")
# LDA a supervised learnign method 

# LDA implemnetation from scratch



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=5)

# Taking in as second argument the Target as labels

X_LDA_2D = lda.fit_transform(X_std, Target.values )
traceLDA = go.Scatter(

    x = X_LDA_2D[:,0],

    y = X_LDA_2D[:,1],

#     name = Target,

#     hoveron = Target,

    mode = 'markers',

    text = Target,

    showlegend = True,

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

data = [traceLDA]



layout = go.Layout(

    title= 'Linear Discriminant Analysis (LDA)',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Linear Discriminant',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Linear Discriminant',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= False

)



fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
# Invoking the t-SNE method

from sklearn.manifold import TSNE



tsne = TSNE(n_components=2)

tsne_results = tsne.fit_transform(X_std) 
traceTSNE = go.Scatter(

    x = tsne_results[:,0],

    y = tsne_results[:,1],

#     name = Target,

#      hoveron = Target,

    mode = 'markers',

    text = Target,

    showlegend = True,

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

data = [traceTSNE]



layout = dict(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',

              hovermode= 'closest',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False),

              showlegend= False,



             )



fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
# tsne is able to cluster the data better than PCA and LDA, but for visualization purposes only