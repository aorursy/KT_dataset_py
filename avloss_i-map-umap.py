import pandas as pd

import numpy as np

import umap

from matplotlib import pyplot as plt

# start with reading train features



xr = pd.read_csv('/kaggle/input/lish-moa/train_features.csv', index_col='sig_id')

display(xr.shape)

xr.head()
# Clearly data has loads of features, which might be hard to visualise, we should employ dimensionality reduction methods - perhaps UMAP!



import warnings

from numba.errors import NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaPerformanceWarning) # suppressing those annoyeing waranings



num_cols = xr.select_dtypes(exclude=object).columns



imap = umap.UMAP()

imap.fit(xr[num_cols])

xrd = imap.transform(xr[num_cols])

xrd = pd.DataFrame(xrd, columns=['x','y'], index=xr.index)
# let's read in targets to add some colors to the chart

yr = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

display(yr.shape)

yr.head()
# let's "melt" dataframe and make it into same shape as our train data



yrm = yr.melt('sig_id')

yrm = yrm[yrm.value!=0].set_index('sig_id')

yrm = xrd.join(yrm)



yrm['variable'] = yrm.variable.fillna('None')



yrm
# let's convert those strings to colors to get an idea

import matplotlib.cm as cm



yrm['colors'] = yrm.variable.apply(lambda x:cm.rainbow(hash(x)%256))
# looks like there's som "natural" separation for that "blue" blob, and a bit for orange, but the rest is totally mixed up!



plt.figure(figsize=(20,10))

plt.scatter(yrm.x, yrm.y, s=10, c=yrm.colors, alpha=0.5)
# let's see if our test data looks anything like this! 🧐



xe = pd.read_csv('/kaggle/input/lish-moa/test_features.csv', index_col='sig_id')





xed = imap.transform(xe[num_cols])

xed = pd.DataFrame(xed, columns=['x','y'], index=xe.index)



plt.figure(figsize=(20,10))

plt.scatter(xed.x, xed.y, s=10)
# Luckly it does! 😅

# Let's color it using nearest neighbour method and see what we get



from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)



neighbors.fit(yrm[['x','y']], xrd.values)



_,ind = neighbors.kneighbors(xed[['x','y']], n_neighbors=1)



# now there can be more than one label.. but let's just pretend that there's just one 😑



xed['colors'] = yrm.iloc[ind.squeeze()]['colors'].values





plt.figure(figsize=(20,10))

plt.scatter(xed.x, xed.y, s=10, c=xed.colors)
# well.. not quite first place, but let's give it a go! 😅

# let's only label those points where out of 20 nearest neighbours at least 4 show same result

from collections import Counter



_,ind = neighbors.kneighbors(xed[['x','y']], n_neighbors=20)



labels = [[yrm.iloc[i].variable for i in ii]  for ii in ind.squeeze()] 



submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv', index_col='sig_id')

submission[:] = 0

for label, (i, row) in zip(labels, submission.iterrows()):

    

    var, count = Counter(label).most_common()[0]

    if count <= 4:

        continue

    if var == 'None':

        continue

    submission.loc[i,var] = 1.0

    

    print(var, count)
submission.to_csv('submission.csv')