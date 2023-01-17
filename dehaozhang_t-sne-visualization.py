import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head()
train.shape
train.isnull().any().sum()
label = train["label"]
label.value_counts()
train = train.drop(labels = ["label"],axis = 1)
train = StandardScaler().fit_transform(train)
pca = PCA(n_components=2)
pca_res = pca.fit_transform(train)
pca_res.shape
plt.figure(figsize=(16,10))

sns.scatterplot(x = pca_res[:,0], y = pca_res[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full');
%%time
tsne = TSNE(n_components = 2, random_state=0)
tsne_res = tsne.fit_transform(train)
plt.figure(figsize=(16,10))

sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full');
pca = PCA(n_components=50)
pca_res_50 = pca.fit_transform(train)
%%time
tsne_res = tsne.fit_transform(pca_res_50)
plt.figure(figsize=(16,10))

sns.scatterplot(x = tsne_res[:,0], y = tsne_res[:,1], hue = label, palette = sns.hls_palette(10), legend = 'full');