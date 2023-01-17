# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing plotting libraries



import matplotlib.pyplot as plt

import seaborn as sns





# Loading numpy file and coverting it to a DataFrame

X = np.load('/kaggle/input/sign-language-digits-dataset/X.npy')

X = pd.DataFrame(X.reshape(2062,-1))
# Centering the data around the Origin



X_centered = X - X.mean(axis = 0) # It is madndatory to center the data incase of PCA calculation using SVD



U, s, Vt = np.linalg.svd(X_centered) # SVD of zero centered array

c1 = Vt.T[:,0] # first unit Vector for the PC 1



c2 = Vt.T[:,1] # second unit vector for the PC 2
# Inorder to find the reduced datset identified by the PCs, we need to do the matrix multiplication of original dataset and 

# the matrix containg first d coulmns of the matrix V



W2 = Vt.T[:,:2] # selecting the first 2 columns of the Matrix V

X2D = X_centered.dot(W2) # reduced dataset obtained by Matrix Multiplication
X2D.rename(columns = {0: 'PC1', 1:'PC2'}, inplace = True)

print(X2D.head())

print(X2D.shape)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_2D = pca.fit_transform(X)
print(pca.explained_variance_ratio_)

print("Amount of variance explained by Both of these PCs is: {} %".format(np.cumsum(pca.explained_variance_ratio_)[-1] * 100 ))
from sklearn.metrics import mean_squared_error



X2Rec = X2D.dot(W2.T) # Reconstruction of the dataset



print(X2Rec.shape)



print("Reconstruction error is: {}".format(mean_squared_error(X, X2Rec)))

pca = PCA()

# If n_componenets is not set explicitly, the value that gets passes by default is equal to the minimum of n_samples or n_features

X_red_1 = pca.fit_transform(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)







print(pca.explained_variance_ratio_)

print("Number of PCs: {}".format(len(pca.explained_variance_ratio_))) # number of PCs in this case is 2062

print("The shape of reduced dataset is {} rows by {} columns".format(X_red_1.shape[0],X_red_1.shape[0]))



print(cumsum)
d = np.argmax(cumsum >= 1) + 1

print(d)



# It can be seen that 2056 PCs can explain 100 % of variance from the Original Dataset
pca = PCA(n_components = 2056)



X_red_2 = pca.fit_transform(X) # reduced Dataset

X_rec_2 = pca.inverse_transform(X_red_2) # Recovered Dataset



print("The shape of recovered dataset is: {}".format(X_rec_2.shape))
fig = plt.figure(figsize = (12,8))



for i, j in enumerate(range(1,2000,100)):

    

    plt.subplot(4,5, i+1)

    plt.imshow(X.iloc[j,:].values.reshape(64,64))

    

fig.tight_layout(pad=1.0)


fig = plt.figure(figsize = (12,8))



for i, j in enumerate(range(1,2000,100)):

    

    plt.subplot(4,5, i+1)

    plt.imshow(X_rec_2[j].reshape(64,64))

    

fig.tight_layout(pad=1.0)
from sklearn.metrics import mean_squared_error



print("Reconsturction error is: {}".format(mean_squared_error(X, X_rec_2)))


def analyse_PCA(df, variance):

    

    print("Generating & Analysing PCs for a preserved variance of {}".format(variance))

    

    df = df.copy()

    

   

        

    pca = PCA(n_components = variance)

        

    X_red = pca.fit_transform(df)

    

    print("The shape of reduced dataset is {}".format(X_red.shape))

    X_rec = pca.inverse_transform(X_red)

        

    cumsum = np.cumsum(pca.explained_variance_ratio_)

    d = np.argmax(cumsum) + 1 

       

    fig = plt.figure(figsize = (12,8))



    for i, j in enumerate(range(1,2000,100)):

    

        plt.subplot(4,5, i+1)

        plt.imshow(X_rec[j].reshape(64,64))

    

    fig.tight_layout(pad=1.0)

    

    

    print("Reconsturction error is: {}".format(mean_squared_error(X, X_rec)))

    
analyse_PCA(X, 0.8)
# Implementation of randomized PCA



import time



solver = ["auto", "full", "randomized"]





time_taken = []



for i in solver:

    start = time.time()

    random_pca = PCA(n_components = 10, svd_solver = "full")





    X3_red = random_pca.fit_transform(X)

    end = time.time()

    

    print("Time taken with {} svd_solver is {}".format(i,end-start))

    

    time_taken.append(end-start)
