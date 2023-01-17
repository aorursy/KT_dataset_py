# Based on https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("../input/test.csv") # the train dataset is now a Pandas DataFrame

df = pd.DataFrame(train.TARGET.value_counts())
df['Percentage'] = 100*df['TARGET']/train.shape[0]
df
X = train.iloc[:,:-1]
y = train.TARGET

X['n0'] = (X==0).sum(axis=1)
train['n0'] = X['n0']
ax = train.plot(x = 'n0', y = 'TARGET', style='o')
ax.set_xlim(min(X['n0']),max(X['n0']))
g = sns.FacetGrid(train, hue = 'TARGET', size = 6)
g = g.map(plt.scatter, "num_var4", "n0", edgecolor="w").add_legend()
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
X_n = normalize(X, axis=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_n)
X['PCA1']=X_pca[:,0]
X['PCA2']=X_pca[:,1]

train['PCA1']=X['PCA1']

sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "PCA1") \
   .add_legend()
train['PCA1']>0.23
sns.FacetGrid(train, hue="TARGET", size=6) \
   .map(sns.kdeplot, "num_var4") \
   .add_legend()
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X_normalized = normalize(X, axis=0)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)
X['PCA1'] = X_pca[:,0]
X['PCA2'] = X_pca[:,1]
train['PCA1']=X['PCA1']
train['PCA2']=X['PCA2']

g = sns.FacetGrid(train, hue = 'TARGET', size = 6)
g = g.map(plt.scatter, "PCA1", "PCA2", edgecolor="w").add_legend()
train.num_var4.hist(bins=100)
plt.xlim(0, num_var4.max());

train = train.replace(-999999,2)
sns.pairplot(train[['var15','num_var4','n0','var3','var38','TARGET']], hue="TARGET", size=2, diag_kind="kde");
