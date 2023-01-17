import os, re, gc, warnings

import numpy as np

import pandas as pd

from scipy.stats import wasserstein_distance as wd

from tqdm import tqdm_notebook

warnings.filterwarnings("ignore")

gc.collect()
DATA_DIR = '../input/'

FILES={}

for fn in os.listdir(DATA_DIR):

    FILES[ re.search( r'[^_\.]+', fn).group() ] = DATA_DIR + fn



CAT_COL='wheezy-copper-turtle-magic'    



train = pd.read_csv(FILES['train'],index_col='id')

# test = pd.read_csv(FILES['test'],index_col='id')

CATS = sorted(train[CAT_COL].unique())
feats = train.columns.drop([CAT_COL,'target'])

wd_matrix = pd.DataFrame(index=range(512), columns=feats)
for wctm in tqdm_notebook(CATS):

    for f in feats:

        wd_matrix.loc[wctm][f] = wd(train[(train[CAT_COL]==wctm) & (train['target']==0) ][f], train[(train[CAT_COL]==wctm) & (train['target']==1) ][f])

        
wd_matrix = wd_matrix[ wd_matrix.columns ].astype('float')
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="whitegrid")

cmap = sns.color_palette('YlGn',10)

f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(wd_matrix, xticklabels=8, yticklabels=16, cmap=cmap ,cbar_kws={"shrink": .5} )
# viewing by ranked feature importance within each of 512 models doesn't reveal much

# wd_matrix.rank(axis=1,ascending=True).astype('int')
sns.set(style="whitegrid")

cmap = sns.color_palette('colorblind',10)

f, ax = plt.subplots(figsize=(20, 40))

f.add_subplot(2,1,1)

sns.heatmap(wd_matrix.iloc[:,0:16], vmin=.6, vmax=2, cmap=cmap ,cbar_kws={"shrink": .5} )

f.add_subplot(2,1,2)

sns.heatmap(wd_matrix.filter(like='important'), vmin=.6, vmax=2, xticklabels=1, yticklabels=16, cmap=cmap ,cbar_kws={"shrink": .5} )
wd_matrix.describe()
wd_matrix['WCTM'] = wd_matrix.index
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})



# Create the data

df = pd.melt(wd_matrix, 

             id_vars=['WCTM'], 

             value_vars=feats, 

             var_name='feature', value_name='wDistance')



# Initialize the FacetGrid object

pal = sns.cubehelix_palette(256, rot=-.1, light=.6)

g = sns.FacetGrid(df, row="feature", hue="feature", aspect=3, height=3, palette=pal)



# Draw the densities in a few steps

g.map(sns.kdeplot, "wDistance", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.05)

g.map(sns.kdeplot, "wDistance", clip_on=False, color="w", lw=2, bw=.05)

g.map(plt.axhline, y=0, lw=2, clip_on=False)



# Define and use a simple function to label the plot in axes coordinates

def label(x, color, label):

    ax = plt.gca()

    ax.text(0, .2, label, fontweight="bold", color='black', size='large',

            ha="left", va="center", transform=ax.transAxes)



g.map(label, "wDistance")



# Set the subplots to overlap

g.fig.subplots_adjust(hspace=-.25)



# Remove axes details that don't play well with overlap

g.set_titles("")

g.set(yticks=[])

g.despine(bottom=True, left=True)
df.sort_values('wDistance',ascending=False)[40*512:50*512:250]
df.sort_values('wDistance',ascending=False).hist()

df.sort_values('wDistance',ascending=False)[:45*512].hist()

df.sort_values('wDistance',ascending=False)[:10*512].hist()
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})





df2=df[df['wDistance']>1]



# Initialize the FacetGrid object

pal = sns.cubehelix_palette(256, rot=-.1, light=.6)

g = sns.FacetGrid(df2, row="feature", hue="feature", aspect=5, height=2, palette=pal)



# Draw the densities in a few steps

g.map(sns.kdeplot, "wDistance", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.05)

g.map(sns.kdeplot, "wDistance", clip_on=False, color="w", lw=2, bw=.05)

g.map(plt.axhline, y=0, lw=2, clip_on=False)



# Define and use a simple function to label the plot in axes coordinates

def label(x, color, label):

    ax = plt.gca()

    ax.text(0, .2, label, fontweight="bold", color='black', size='large',

            ha="left", va="center", transform=ax.transAxes)



g.map(label, "wDistance")



# Set the subplots to overlap

g.fig.subplots_adjust(hspace=-.25)



# Remove axes details that don't play well with overlap

g.set_titles("")

g.set(yticks=[])

g.despine(bottom=True, left=True)