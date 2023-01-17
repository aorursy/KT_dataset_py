# ! conda install -c bioconda kalign3
# ! kalign  -i sequences.fasta -o kalign_fast.fasta 
! conda install -y scikit-bio
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import holoviews as hv
from skbio import DNA
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

hv.extension('bokeh')
file = '/kaggle/input/2019ncov-sequences-wuhan-coronavirus/kalign.fasta'
class FastaSeq:
    def __init__(self, name, sequence):
        self.name = name
        self.sequence = DNA(sequence)

    def get_seqs(file):
        items = []
        index = 0
        start = False
        for line in file:
            if line.startswith(">"):
                start = True
                if index >= 1:
                    items.append(aninstance)
                index+=1
                name = line[:-1]
                seq = ''
                aninstance = FastaSeq(name, seq)
            else:
                if start:
                    seq += line[:-1]
                    aninstance = FastaSeq(name, seq)
        if start:
            items.append(aninstance)

        return items
with open(file, "r") as f:
    data = FastaSeq.get_seqs(file=f.readlines())
str(data[3].sequence)[:1000] + ' ...'
replace = {'A':'A',
             'C':'C',
             'G':'G',
             'T':'U',
             'R':'AG',
             'Y':'CU',
             'S':'GC',
             'W':'AU',
             'K':'GU',
             'M':'AC',
             'B':'CGU',
             'D':'AGU',
             'H':'ACU',
             'V':'ACG',
             'N':'ACGU',
             '.':'S',
            '-':'S'}
df = pd.DataFrame(list(map(lambda x: pd.Series(list(str(x.sequence))), data)))
df.shape
gaps = (df == '-').mean(1)
gap_threshold = gaps < 0.25

sequences = (df
             .loc[gap_threshold, :]
             .replace(replace)
             .applymap(str))
sequences.shape
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
encoder = CountVectorizer(analyzer='char', vocabulary=['S', 'C', 'A', 'G', 'U'], strip_accents=None, lowercase=False)
X = hstack([encoder.fit_transform(sequences.loc[:, i]) for i in sequences])
X = np.ascontiguousarray(X
                         .toarray()
                         .astype('float32'))
# svd
svd = TruncatedSVD(2, algorithm='arpack')
Z = svd.fit_transform(X)
svd_components = [f'Component {i} ({round(e*100)}%)' for i, e in enumerate(svd.explained_variance_ratio_.tolist())]

# kpca
kpca = KernelPCA(2, kernel='cosine', n_jobs=-1)
T = kpca.fit_transform(X)
kpca_components = [f'Component {i}' for i in range(2)]

# TSNE
tsne = TSNE(2, perplexity=18, metric='hamming')
U = tsne.fit_transform(X)
tsne_high_components = [f'Component {i}' for i in range(2)]

tsne = TSNE(2, perplexity=5, metric='hamming')
W = tsne.fit_transform(X)
tsne_low_components = [f'Component {i}' for i in range(2)]


tsne = TSNE(2, perplexity=18, metric='cosine')
V = tsne.fit_transform(X)
tsne_cosine_high_components = [f'Component {i}' for i in range(2)]

tsne = TSNE(2, perplexity=5, metric='cosine')
S = tsne.fit_transform(X)
tsne_cosine_low_components = [f'Component {i}' for i in range(2)]

# plots
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
svd_plot = pd.DataFrame((Z - Z.mean(0)) / Z.std(0), columns=svd_components).plot.scatter(x=svd_components[0], y=svd_components[1], title='SVD', ax=axes[0][0])
cosine_kpca = pd.DataFrame(T, columns=kpca_components).plot.scatter(x=kpca_components[0], y=kpca_components[1], title='Cosine KPCA', ax=axes[0][1])
tsne_high = pd.DataFrame(U, columns=tsne_high_components).plot.scatter(x=tsne_high_components[0], y=tsne_high_components[1], title='Hamming TSNE: Perplexity 18', ax=axes[1][0])
tsne_low = pd.DataFrame(W, columns=tsne_low_components).plot.scatter(x=tsne_low_components[0], y=tsne_low_components[1], title='Hamming TSNE: Perplexity 5', ax=axes[1][1])
tsne_high = pd.DataFrame(V, columns=tsne_cosine_high_components).plot.scatter(x=tsne_cosine_high_components[0], y=tsne_cosine_high_components[1], title='Cosine TSNE: Perplexity 18', ax=axes[2][0])
tsne_low = pd.DataFrame(S, columns=tsne_cosine_low_components).plot.scatter(x=tsne_cosine_low_components[0], y=tsne_cosine_low_components[1], title='Cosine TSNE: Perplexity 5', ax=axes[2][1])
from sklearn.metrics import pairwise_distances

def d_stat(X):
    return np.sum(pairwise_distances(X)**2).astype('float32')

def gap(data, labels=None):
    grouper = (pd.DataFrame(data)
     .groupby(labels))
    
    D_k = grouper.apply(d_stat)
    N_k = grouper.count()[0]
    
    W_k = (D_k/(2*N_k)).sum()
    
    D = d_stat(data)
    N = X.shape[0]
    W = D / (2 * N)
    
    return np.log(W) - np.log(W_k)

def orth(v, u):
    return v - (v@v)/(u@u) * u
from sklearn.decomposition import PCA
from functools import partial
from sklearn.metrics import pairwise_distances_argmin

class PrincipleGeneShaving:
    def __init__(self, n_clusters = 2, alpha = 0.1):
        self.n_clusters = n_clusters
        self.alpha = alpha
        
    def fit_transform(self, X):
        # centre each row at zero
        X  = X - X.mean(1).reshape(-1,1)
        svd = PCA(1)
        
        labels = np.full((X.shape[0], self.n_clusters), 0)
        for k in range(self.n_clusters - 1):
            
            # shave
            indexes = [np.arange(X.shape[0]).flatten()]
            S = [X]
            while S[-1].shape[0] > 1:
                P = svd.fit_transform(S[-1].T)

                inner = S[-1] @ P

                threshold = np.quantile(inner, self.alpha)
                not_shaved = (inner > threshold).reshape(-1,1).flatten().copy()
                
                if not_shaved.ndim == 2:
                    print('reshape')
                    ns = np.array(sum(not_shaved.tolist(), []))
                else:
                    ns = not_shaved
                                
                indexes.append(indexes[-1][ns])
                S_prime = S[-1][ns, :]

                if S_prime.shape[0] > 1:
                    S.append(S_prime)
                else:
                    break

            # score
            scores = []
            for index in indexes:
                l = np.full(X.shape[0], k)
                l[index] = k + 1

                scores.append(gap(X, l))

            max_score = np.argmax(scores)

            labels[indexes[max_score], k] += 1
            
            # orthogonalize
            X_bar = (np.ascontiguousarray(X[(labels[:, k] == 1), :].mean(0))
                     .ravel()
                     .astype('float32'))
            
            assert X.shape[1] == X_bar.shape[0]
            X = np.apply_along_axis(partial(orth, u=X_bar), 1, np.ascontiguousarray(X)).astype('float32')
        
        return pairwise_distances_argmin(labels, np.unique(labels, axis=0))
pgs = PrincipleGeneShaving(3, 0.25)
labels = pgs.fit_transform(X)
hv.Bars(pd.Series(labels, name='Count of labels').value_counts()).opts(xlabel='Label')
components = ['Component 1', 'Component 2']
(hv.Scatter(pd.DataFrame(U, columns=components)
            .assign(cluster=labels.astype(str)), 
            kdims=components[0], vdims=[components[1], 'cluster'])
 .opts(color='cluster', cmap='Category10',
       tools=['hover'], size=7,
       width=800, height=600, title='SARS-COV-2: Principle Gene Shaving'))
super_genes = (sequences
               .assign(label=labels)
               .groupby('label').apply(lambda df: df.mode())
               .dropna()
               .drop(columns=['label']))
hv.Raster(super_genes.iloc[[0], :]
           .replace({k: i for i, k in enumerate(list(replace.keys()) + ['S', 'U'])}).astype(np.int).values).opts(width=1000, height=100, title='Super-gene 1')
hv.Raster(super_genes.iloc[[1], :]
           .replace({k: i for i, k in enumerate(list(replace.keys()) + ['S', 'U'])}).astype(np.int).values).opts(width=1000, height=100, title='Super-gene 2')
hv.Raster(super_genes.iloc[[2], :]
           .replace({k: i for i, k in enumerate(list(replace.keys()) + ['S', 'U'])}).astype(np.int).values).opts(width=1000, height=100, title='Super-gene 3')