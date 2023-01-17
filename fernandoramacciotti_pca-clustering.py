import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
samples_sub = pd.read_csv('../input/sample_submission.csv')
test = pd.read_csv('../input/test.csv')
print('==== Training ====')
print('no. samples : {:,}'.format(train.shape[0]))
print('no. features: {:,}'.format(train.shape[1]))
print()
print('====== Test ======')
print('no. samples : {:,}'.format(test.shape[0]))
print('no. features: {:,}'.format(test.shape[1]))
train.head(5)
metadata = pd.DataFrame(
    columns=['column', 'dtype', 'desc'])

for i, c in enumerate(train.columns):
    metadata.loc[i, 'column'] = c
    metadata.loc[i, 'dtype'] = train[c].dtype
    if c == 'ID':
        metadata.loc[i, 'desc'] = 'id'
    elif c == 'target':
        metadata.loc[i, 'desc'] = 'target'
    else:
        metadata.loc[i, 'desc'] = 'feature'

metadata.groupby(by=['dtype'])['column'].count()
metadata[metadata['dtype'] == 'object']
fig, ax = plt.subplots(2, 2, figsize=(16, 9))

train.target.plot.hist(ax=ax[0, 0], title='Target variable histogram')
train.target.plot.box(ax=ax[0, 1], title='Target variable boxplot')

train.target.transform(np.log10).plot.hist(ax=ax[1, 0], title='Log10 of target variable histogram')
train.target.transform(np.log10).plot.box(ax=ax[1, 1], title='Log10 of target variable boxplot')

fig.tight_layout()
train.target.describe()
features = metadata[metadata['desc'] == 'feature']['column'].values
tgt_corr = pd.DataFrame(
    index=features,
    columns=['tgt_corr']
)

for f in features:
    tgt_corr.loc[f, 'tgt_corr'] = train.target.corr(train[f])

#tgt_corr.sort_values(by='tgt_corr', axis=1, ascending=False).head(15)
tgt_corr.sort_values('tgt_corr', ascending=False).plot.bar(xticks=[], 
                                                           title='Target Variable Correlations Coeff',
                                                           figsize=(16, 5))
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train[features], np.log10(train.target), test_size=0.2, random_state=0)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

pca = PCA(n_components=0.85)
pca.fit(X_train)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ncomp = len(pca.explained_variance_)
ax1.bar(range(1, ncomp+1), pca.explained_variance_)
ax1.axhline(1, color='red', linestyle='dashdot')
ax1.set_title('Explaning variance of each PC')

ax2.bar(range(1, ncomp+1), pca.explained_variance_ratio_.cumsum() * 100)
ax2.axhline(85, color='red', linestyle='dashdot')
ax2.set_title('Explained cumulative percentage of explained variance by each PC')

fig.tight_layout()
scores = pca.transform(X_train)
fig = plt.figure(figsize=(16,5))

plot_max = 4
n = 0
for i in range(0, plot_max * 2, 2):
    n += 1
    ax = fig.add_subplot(plot_max // 2, 2, n)
    h = ax.scatter(scores[:, i+1], scores[:, i+2], c=y_train)
    ax.set_title('Scores from PC{} vs PC{}'.format(i+1, i+2))

fig.colorbar(h)
fig.tight_layout()
