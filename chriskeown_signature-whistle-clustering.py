import pandas as pd
import librosa
import matplotlib.pylab as plt
import numpy as np
import random
from scipy import signal
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import OrderedDict

%matplotlib inline
df = pd.read_csv("../input/dolphin-vocalizations/signature_whistle_metadata.csv")
signature_whistles = pickle.load(open("../input/dolphin-vocalizations/signature_whistles.p", "rb"))
df = df.loc[df.ID1 == df.ID2]
df = df.drop(166) # This one is wonky
signature_whistles = [signature_whistles[i] for i in df.index]
df.reset_index(inplace=True)
# Get list of dolphin names
dolphins = df.ID1.unique().tolist()
# Create spectrograms
sampling_rate = 48000
spectrograms = [None] * len(df)

for i, whistle in enumerate(signature_whistles):
    spectrograms[i] = signal.spectrogram(whistle, sampling_rate)
specs = []
for _, _, spec in spectrograms:
    specs.append(spec.sum(axis=1))

specs = np.array(specs)

# Normalize the sig whistles
specs = specs / specs.sum(axis=1, keepdims=True)
pca = PCA(n_components=15)
pcs = pca.fit_transform(specs)
print("Remaining variance after PCA: " + str(pca.explained_variance_ratio_.sum()))
print(pcs.shape)

tsne = TSNE(n_components=2, init='pca', random_state=1, perplexity=50)
tsne_transf = tsne.fit_transform(pcs)
print(tsne_transf.shape)
colors = ['r','k','g','b','m','c','#888888']
colmap = dict(zip(dolphins, colors))
fig, axes = plt.subplots(1,2, figsize=(15,10))
# cax = axes[0].scatter(pcs[:,0], pcs[:,1]) # , c=colmap[df.loc[i,'ID1']], s=5)
# print(pcs[:,0].max())
for i in range(pcs.shape[0]):
    cax = axes[0].scatter(pcs[i,0], pcs[i,1], c=colmap[df.loc[i,'ID1']], s=5, label=df.loc[i,'ID1'])
    cax = axes[1].scatter(tsne_transf[i,0], tsne_transf[i,1], c=colmap[df.loc[i,'ID1']], s=5, label=df.loc[i,'ID1'])
    
axes[0].set_title('PCA Normalizated Data')
axes[1].set_title('TSNE Normalizated Data')
handles, labels = axes[0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes[0].legend(by_label.values(), by_label.keys(), numpoints=1, loc='upper right', frameon=True, fontsize=14)
axes[1].legend(by_label.values(), by_label.keys(), numpoints=1, loc='upper right', frameon=True, fontsize=14)
print_components = 5

fig, axes = plt.subplots(print_components-1,print_components-1, figsize=(15,10))
for i in range(0,print_components-1):
    for j in range(1,print_components):
        if (i == 0) and (j == 1) :
            # for the first plot, we have to loop to make the legend work
            for k in range(pcs.shape[0]):
                cax = axes[i,j-1].scatter(pcs[k,i], pcs[k,j], c=colmap[df.loc[k,'ID1']], s=5, label=df.loc[k,'ID1'])
        elif j > i:
            # for plots without a legend, no looping is required
            axes[i,j-1].scatter(pcs[:,i], pcs[:,j], c=df['ID1'].apply(lambda x: colmap[x]), s=5)  
            # axes[i,j-1].axis('off')
        else:
            axes[i,j-1].axis('off')
            
        axes[i,j-1].set_xticklabels([])
        axes[i,j-1].set_yticklabels([])
        axes[i,j-1].set_xlabel('PC ' + str(i+1))
        axes[i,j-1].set_ylabel('PC ' + str(j+1))

fig.suptitle('Top Principal Components', fontsize=16)
# Add legend just to the first plot
handles, labels = axes[0,0].get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
axes[0,0].legend(by_label.values(), by_label.keys(), numpoints=1, loc='upper right', frameon=True, fontsize=11)
