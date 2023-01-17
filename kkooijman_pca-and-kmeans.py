# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 

import matplotlib.pyplot as plt # Python defacto plotting library

import seaborn as sns # More snazzy plotting library

%matplotlib inline 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movie = pd.read_csv('../input/tmdb_5000_movies.csv')

credit = pd.read_csv('../input/tmdb_5000_credits.csv')
movie.head()

#credit.head()
str_list = [] # empty list to contain columns with strings

for colname, colvalue in movie.iteritems():

    if type(colvalue[1]) == str:

        str_list.append(colname)

#Get to the numeric columns by inversion

num_list = movie.columns.difference(str_list)
movie_num = movie[num_list]

movie_num.head()
movie_num = movie_num.fillna(value=0, axis=1)
X = movie_num.values

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
list(movie)
movie.plot(y = 'vote_average', x = 'runtime', kind = 'hexbin', gridsize=35, sharex=False, 

           colormap='cubehelix', title='Hexbin of vote_average and runtime',figsize=(12,8))

movie.plot(y ='vote_average', x = 'revenue', kind='hexbin', gridsize = 45, sharex = False,

          colormap = 'cubehelix', title='Hexbin of vote_average and revenue', figsize = (12,8))
f, ax = plt.subplots(figsize=(12,10))

plt.title('Pearson Correlation of Movie Features')

sns.heatmap(movie_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True,

           cmap="YlGnBu", linecolor='black', annot=True)
#Calculating Eigenvecors and eigenvalues of Covariance matrix

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
cum_var_exp
# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 

plt.figure(figsize=(10, 5))

plt.bar(range(len(var_exp)), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')

plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show()
movie_num.describe()
movie['revenue'].plot.hist()
movie['revenue_classes'] = pd.cut(movie['revenue'],10)

movie['vote_classes'] = pd.cut(movie['vote_average'],4, labels=["low", "medium-low","medium-high","high"])

#movie['vote_classes'] = pd.cut(movie['vote_average'],10, labels=["1", "2","3","4","5","6","7","8","9","10"])
list(movie)
X_revenue = movie.ix[:,(0,8,18,19)].values

y_revenue = movie.ix[:,20].values



X_votes = movie.ix[:,(0,8,12,19)].values

y_votes = movie.ix[:,21].values
from matplotlib import pyplot as plt

import numpy as np

import math



feature_dict ={0:'budget',

              1: 'popularity',

              2: 'revenue',

              3: 'vote_count'}



#Use this block for a cut in 4 blocks

'''

label_dict = {1: 'low',

              2: 'medium-low',

              3: 'medium-high',

              4: 'high'}



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(8,6))

    for cnt in range(4):

        for lab in('low', 'medium-low','medium-high','high'):

            plt.hist(X_votes[y_votes==lab, cnt],

                    label = lab,

                    bins = 10,

                    alpha = 0.3,)

            plt.xlabel(feature_dict[cnt])

        plt.legend(loc='upper right', fancybox=True, fontsize=8)

        

        plt.tight_layout()

        plt.show()

'''



#Use this block for a cut in 10 blocks.

label_dict = {0: '0-1',

               1: '1-2',

               2: '2-3',

               3: '3-4',

               4: '4-5',

               5: '5-6',

               6: '6-7',

               7: '7-8',

               8: '8-9',

               9: '9-10'}



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(8,6))

    for cnt in range(4):

        for lab in("1", "2","3","4","5","6","7","8","9","10"):

            plt.hist(X_votes[y_votes==lab, cnt],

                    label = lab,

                    bins = 10,

                    alpha = 0.3,)

            plt.xlabel(feature_dict[cnt])

        plt.legend(loc='upper right', fancybox=True, fontsize=8)

        

        plt.tight_layout()

        plt.show()





from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X_votes)
import numpy as np

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

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

print(var_exp)

print(cum_var_exp)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))



    plt.bar(range(len(var_exp)), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(len(cum_var_exp)), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
matrix_w = np.hstack((eig_pairs[0][1].reshape(7,1),

                      eig_pairs[1][1].reshape(7,1),

                      eig_pairs[2][1].reshape(7,1),

                      eig_pairs[3][1].reshape(7,1)))



print('Matrix W:\n', matrix_w)
eig_pairs[0][1].reshape(7,1)
Y = X_std.dot(matrix_w.T)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))

    for lab, col in zip(('low', 'medium-low','medium-high', 'high'),

                        ('blue', 'red', 'green','orange')):

        plt.scatter(Y[y_votes==lab, 0],

                    Y[y_votes==lab, 1],

                    label=lab,

                    c=col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='lower center')

    plt.tight_layout()

    plt.show()
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components = 4)

Y_sklearn = sklearn_pca.fit_transform(X_std)
with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6,4))

    for lab, col in zip(('low','medium-low','medium-high','high'),

                       ('blue','red','green','orange')):

        plt.scatter(Y_sklearn[y_votes==lab, 0],

                   Y_sklearn[y_votes==lab, 1],

                   label = lab,

                   c = col)

    plt.xlabel('Principal Component 1')

    plt.ylabel('Principal Component 2')

    plt.legend(loc='upper left')

    plt.tight_layout()

    plt.show()

    
X = movie_num.values

# Data Normalization

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=7)

x_7d = pca.fit_transform(X_std)
pca4 = PCA(n_components=4)

x_4d = pca.fit_transform(X_std)
#Set a 3 KMeans clustering

kmeans = KMeans(n_clusters = 3)



#Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_7d)



#Define our own color map

LABEL_COLOR_MAP = {0:'r', 1: 'g', 2: 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_7d[:,0],x_7d[:,2], c= label_color, alpha=0.5) 

plt.show()
#Set a 3 KMeans clustering

kmeans = KMeans(n_clusters = 3)



#Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_4d)



#Define our own color map

LABEL_COLOR_MAP = {0:'r', 1: 'g', 2: 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_4d[:,0],x_4d[:,2], c= label_color, alpha=0.5) 

plt.show()
# Create a temp dataframe from our PCA projection data "x_9d"

df = pd.DataFrame(x_4d)

df = df[[0,1,2]] # only want to visualise relationships between first 3 projections

df['X_cluster'] = X_clustered
# Call Seaborn's pairplot to visualize our KMeans clustering on the PCA projected data

sns.pairplot(df, hue='X_cluster', palette= 'Dark2', diag_kind='kde',size=1.85)