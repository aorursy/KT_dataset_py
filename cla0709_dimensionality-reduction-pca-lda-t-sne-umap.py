import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA
from scipy import optimize

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from matplotlib import pyplot as plt
import imageio
import tqdm
fig = plt.figure(figsize=(20,10))
image = imageio.imread("../input/amer_sign2.png")
plt.imshow(image)
df = pd.read_csv("../input/sign_mnist_train.csv")
df.head()
df.shape
letter2encode = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,
                'N': 12,'O': 13,'P': 14,'Q': 15,'R': 16,'S': 17,'T': 18,'U': 19,'V': 20,'W': 21,'X': 22, 'Y': 23}

def fix_label_gap(l):
    if(l>=9):
        return (l-1)
    else:
        return l

def encode(character):
    return letter2encode[character]

df['label'] = df['label'].apply(fix_label_gap)
WORD = 'THANKS'

word = np.array(list(WORD))
embedded_word = list(map(encode, word))
print(embedded_word)

reduced_df = df[df['label'].isin(embedded_word)]

reduced_df.shape
X = reduced_df.loc[:, reduced_df.columns != 'label'].values

len(X)
y = reduced_df['label'].values

plt.imshow(X[12].reshape(28,28))
X_PCA = PCA(n_components=5).fit_transform(X)
X_LDA = LDA(n_components=5).fit_transform(X,y)
X_TSNE = TSNE().fit_transform(X)
X_UMAP = UMAP(n_neighbors=15,
                      min_dist=0.1,
                      metric='correlation').fit_transform(X)
fig = plt.figure(figsize=(50,40))
plt.subplot(2,2,1)
plt.scatter(X_PCA[:,0], X_PCA[:,1], c=y, cmap='Set1')
plt.title("Principal Component Analysis", fontsize=40)
plt.subplot(2,2,2)
plt.scatter(X_UMAP[:,0], X_UMAP[:,1], c=y, cmap='Set1')
plt.title("Uniform Manifold Approximation and Projections", fontsize=40)
plt.subplot(2,2,3)
plt.scatter(X_LDA[:,0], X_LDA[:,1], c=y, cmap='Set1')
plt.title("Linear Discriminant Analysis", fontsize=40)
plt.subplot(2,2,4)
plt.scatter(X_TSNE[:,0], X_TSNE[:,1], c=y, cmap='Set1')
plt.title("t-Distributed Stochastic Neighbor Embedding", fontsize=40)
plt.show()