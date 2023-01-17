import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



np.random.seed(123)
def minmaxscale(x):

    return (x-x.min()) / (x.max()-x.min())



# plot 1

x1 = np.random.uniform(4, 5, 50)

x2 = np.random.uniform(0, 10, 50)



# plot 2

errors = np.random.uniform(0, 2, 50)

x3 = np.random.uniform(5, 3, 50)

x4 = 3*x3 + 3 + errors



# subplots

fig, axarr = plt.subplots(1,2)

fig.set_size_inches(10,4)



axarr[0].scatter(x1, x2)

axarr[0].arrow(4.5, 2, 0, 6, color="red", width=0.10)

arrow = axarr[0].arrow(4.5, 6, 0, -4, color="red", width=0.10)



axarr[0].title.set_text("Consider reducing 2-D data to 1-D")

axarr[0].set_xlabel("feat1 (on x-component of cartesian plane)")

axarr[0].set_ylabel("feat2 (on y-component of cartesian plane)")

axarr[0].legend(["original data"])

axarr[0].legend([arrow,], ['Optimal component',])

axarr[0].set_xticks(range(0, 10))

axarr[0].set_yticks(range(0, 10))



axarr[1].scatter(x3, x4)

arrow = axarr[1].arrow(3, 13, 1.6, 5, color="red", width=0.10)

axarr[1].arrow(4.6, 18, -1.6, -5, color="red", width=0.10)



axarr[1].title.set_text("Consider reducing 2-D data to 1-D")

axarr[1].set_xlabel("feat3 (on x-component of cartesian plane)")

axarr[1].set_ylabel("fea4 (on y-component of cartesian plane)")

axarr[1].legend(["original data", "optimal component"])

axarr[1].legend([arrow,], ['Optimal component',])

axarr[1].set_xticks(range(0, 10))

axarr[1].set_yticks(range(10, 20))



plt.show()
fig = plt.figure(figsize=(7,4))



# cartesian plane start------------------------------------------

begx, begy = 0,0

dx, dy = 4,0

plt.arrow(begx, begy, dx, dy, color="black", width=0.001, head_width=0.04, head_length=0.04)



begx, begy = 0,0

dx, dy = -4,0

plt.arrow(begx, begy, dx, dy, color="black", width=0.001, head_width=0.04, head_length=0.04)



begx, begy = 0,0

dx, dy = 0,3

plt.arrow(begx, begy, dx, dy, color="black", width=0.001, head_width=0.04, head_length=0.04)



begx, begy = 0,0

dx, dy = 0,-4

plt.arrow(begx, begy, dx, dy, color="black", width=0.001, head_width=0.04, head_length=0.04)

# cartesian plane end------------------------------------------





# optimal component

begx, begy = 0,0

endx, endy = 5,3

plt.plot([begx, endx], [begy, endy],linestyle="-", color="red", label="optimal component u_1")



# points

plt.scatter([2.5], [2.5], color="blue", label="original datapoint x_i")

plt.scatter([3], [1.8], color="green", label="projection of x_i")



# distance d

begx, begy = 2.5, 2.5

endx, endy = 3, 1.8

plt.plot([begx, endx], [begy, endy],linestyle="--", color="orange", label="distance d_i")



# original point vector

begx, begy = 0, 0

endx, endy = 2.5, 2.5

plt.plot([begx, endx], [begy, endy],linestyle="-", color="blue", label="x_i datapoint vector magnitude")



# magnitude of projection

begx, begy = 0, 0

endx, endy = 3, 1.8

plt.plot([begx, endx], [begy, endy],linestyle="--", color="green", label="x_i projection magnitude")



plt.title("Right-angled Triangle")

plt.axis('off')

plt.legend()

plt.show()
fig, axarr = plt.subplots(1,4)

fig.set_size_inches(20,4)



# 1. perfect slope

x1 = np.random.uniform(3,4,100)

x2 = 3*x1 + 4



axarr[0].set_title("Fig1: Perfect slope\nλ2 = 0 ")

axarr[0].scatter(x1, x2)

axarr[0].plot([3.2, 3.88], [15.75,13], color="red")

axarr[0].plot(x1, x2, color="red")

axarr[0].grid()

axarr[0].set_xlabel("feat1")

axarr[0].set_ylabel("feat2")





# 2. Fig2:

_x1 = np.random.uniform(3,4,100)

errors = np.random.normal(0,0.13,100)

_x2 = 3*_x1 + 4 + errors



axarr[1].set_title("Fig2: λ1>>λ2")

axarr[1].scatter(_x1, _x2)

axarr[1].plot(x1, x2, color="red")

axarr[1].plot([3.2, 3.83], [15.75,13], color="red")

axarr[1].grid()

axarr[1].set_xlabel("feat1")

axarr[1].set_ylabel("feat2")





# 3. Fig3:

_x1 = np.random.uniform(3,4,100)

errors = np.random.normal(0,0.6,100)

_x2 = 3*_x1 + 4 + errors



axarr[2].set_title("Fig3: λ1>λ2")

axarr[2].scatter(_x1, _x2)

axarr[2].plot(x1, x2, color="red")

axarr[2].plot([3.3, 3.66], [17,11], color="red")

axarr[2].grid()

axarr[2].set_xlabel("feat1")

axarr[2].set_ylabel("feat2")





# 4. Fig4: Perfect circle

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=1, n_features=2, random_state=0)



axarr[3].set_title("Fig4: Perfect circle\nλ1=λ2")

axarr[3].scatter(X.T[0], X.T[1])

axarr[3].plot([-1,3],[2,7], color="red")

axarr[3].plot([-2,4],[7,2], color="red")

axarr[3].grid()

axarr[3].set_xlabel("feat1")

axarr[3].set_ylabel("feat2")



plt.show()
# REPRODUCABLE CODE: Visualisation only

# =====================================



#libraries

import numpy as np

import pandas as pd

from scipy.linalg import eigh 

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns



def get_pca_df(X, labels):

    

    # 1.standardize

    # ==============

    standardized_data = StandardScaler().fit_transform(X)

    covar_matrix = np.matmul(standardized_data.T , standardized_data)



    # original dims of original data (m)

    m = len(covar_matrix)

    

    # 2.Get first-two eigen vectors

    # =============================

    values, vectors = eigh(covar_matrix, eigvals=(m-2, m-1)) #(62, 63) for MNIST_8X8 i.e from end to front

    U = vectors.T # shape (d, m) where



    #print(values)    



    # 3. Genereate new 2-D data (projections)

    # ======================================

    projections = np.matmul(U, standardized_data.T) #(d,m) x (m, n) => (d, n)  [where, n - num_of_samples ]

    # appending label to the new 2-D data 

    projections = np.vstack((projections, labels)).T 



    # creating a new data frame for ploting the labeled points.

    dataframe = pd.DataFrame(data=projections, columns=("1st_principal", "2nd_principal", "label"))

    

    return dataframe



""" # Plotting

# plot bc 

sns.FacetGrid(df2, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()



plt.title("Breast Cancer Dataset \nm={} n={}".format(X_bc.shape[1], X_bc.shape[0]))

plt.show()

"""
from sklearn.datasets import load_digits

digits = load_digits()

X_mnist = digits.data

labels_mnist = digits.target
df = get_pca_df(X_mnist, labels_mnist)

# X - all numerical rvs

# label - categorical rv
# plot

sns.FacetGrid(df, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()

plt.title("MNIST Dataset \nm={} n={}".format(X_mnist.shape[1], X_mnist.shape[0]))

plt.show()
# load bc data

from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()

X_bc = bc.data

labels_bc = bc.target
df2 = get_pca_df(X_bc, labels_bc)

# X - all numerical rvs

# label - categorical rv
# plot bc 

sns.FacetGrid(df2, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()



plt.title("Breast Cancer Dataset \nm={} n={}".format(X_bc.shape[1], X_bc.shape[0]))

plt.show()
# REPRODUCIBLE CODE:

# PCA for dimensionality redcution (non-visualization)

# ====================================================

import matplotlib.pyplot as plt

import numpy as np

from sklearn import decomposition

pca = decomposition.PCA()



def pca_anaysis(sample_data):

    """Sample data is `X` with all numerical rvs"""

    # connfig and transform

    pca.n_components = sample_data.shape[1]

    pca_data = pca.fit_transform(sample_data)



    # calculate cumulative variance

    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

    cum_var_explained = np.cumsum(percentage_var_explained)



    # Plot the PCA spectrum

    plt.figure(1, figsize=(6, 4))



    plt.clf()

    plt.plot(cum_var_explained, linewidth=2)

    plt.axis('tight')

    plt.grid()

    plt.xlabel('n_components')

    plt.ylabel('Cumulative_explained_variance')

    plt.title("PCA Analysis")

    plt.show()

    

import pandas as pd

def pca_get_df(sample_data, labels, new_dims=2):

    """Sample data is `X` with all numerical rvs"""

    pca.n_components = new_dims

    pca_data = pca.fit_transform(sample_data)

    

    df = pd.DataFrame(pca_data)

    df['labels'] = labels

    return df
from sklearn.datasets import load_digits

digits = load_digits()

X_mnist = digits.data

labels_mnist = digits.target
pca_anaysis(X_mnist)
pca_get_df(X_mnist, labels_mnist, new_dims=20).head()
df = pca_get_df(X_mnist, labels_mnist, new_dims=2)
# plot bc 

import seaborn as sns

sns.FacetGrid(df, hue="labels", height=6).map(plt.scatter, 0, 1).add_legend()



plt.title("Mnist Dataset \nm={} n={}".format(X_mnist.shape[1], X_mnist.shape[0]))

plt.show()