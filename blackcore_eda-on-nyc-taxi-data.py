import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.describe() # Check out some interesting statistics of the training set
test.describe() # Check out some interesting statistics of the test set
# Converting the "store_and_fwd_flag" feature to a numeric one

train['store_and_fwd_flag'] = np.where(train['store_and_fwd_flag'] == 'N', 0, 1)

train['store_and_fwd_flag'].head()
train_sample = train.sample(frac=0.3, replace=False) # Take a small sample to run all faster

# ...or you can consider the whole dataset when looking at the below plots...but it takes some patience :)
# Let's store our features in an array

feats = np.array(train_sample.iloc[:,[1,4,5,6,7,8,9,10]]).astype(float)



# And let's save those column names, they will come in handy later.

header = train_sample.iloc[:,[1,4,5,6,7,8,9,10]].columns
fig = plt.figure(figsize=(14, 12))

feat_comb_1 = [1, 2, 3, 4, 5]

feat_comb_2 = [1, 2, 3, 4, 5]



feature_array = [feats[:, j] for j in range(len(header))] # This gives the transpose of "feats".



nfeat = len(feat_comb_1)



for j in range(nfeat):

    for k in range(nfeat):

        plt.subplot(nfeat, nfeat, j + 1 + k * nfeat)

        plt.scatter(feature_array[feat_comb_1[j]], feature_array[feat_comb_2[k]])

        plt.xlabel(header[feat_comb_1[j]])

        plt.ylabel(header[feat_comb_2[k]])

        fig.tight_layout()
fig, axes = plt.subplots(figsize=(20, 10))

bp = plt.boxplot(feats)

plt.setp(bp['boxes'], color='black')

plt.setp(bp['whiskers'], color='black')

plt.setp(bp['fliers'], color='red', marker='o')

plt.xlabel('Features')

plt.ylabel('Value')

axes.set_xticklabels(header, rotation=270)

plt.grid()
fig = plt.figure(figsize=(12, 8))

correlation_matrix = np.corrcoef(feats, rowvar=0)



# A nice way to visualise the correlations matrix is to make a

# scatterplot and rather than write values, assign a colour map.

plt.pcolor(correlation_matrix, cmap='hot', vmin=-1, vmax=1)

plt.colorbar()



# Put the major ticks at the middle of each cell.

plt.yticks(np.arange(0.5, 13.5), range(1, 9))

plt.xticks(np.arange(0.5, 13.5), range(1, 9))



plt.show()