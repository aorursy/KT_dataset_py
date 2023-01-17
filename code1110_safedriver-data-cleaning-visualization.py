# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





# import libraries

import seaborn as sns

import matplotlib.pyplot as plt



# import training data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# submit_exp = pd.read_csv("../input/sample_submission.csv")

# print(submit_exp.head())



# train and test

id_train = train["id"]

y_train = train["target"]

X_train = train.drop(["id","target"], axis=1)



id_test = test["id"]

X_test = test.drop(["id"], axis=1)



print(X_train.head())

# print(X_test.head())
# -1 means nan in this case...so put nan back

X_train = X_train.replace(-1, np.NaN)

X_test = X_test.replace(-1, np.NaN)



# concatenate train and test to deal with nan together

Xmat = pd.concat([X_train, X_test])



# visualize the number of nans in each column

# (shamelessly adapted from:

#https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial)

import missingno as msno



msno.matrix(df=X_train.iloc[:,:39], figsize=(20,14), color=(0.5,0,0))
# Columns with many nans itself may be meaningful

def nan2bi(x):

    if np.isnan(x):

        return 1

    else:

        return 0



Xmat = pd.concat([X_train, X_test])

cols = ["ps_reg_03","ps_car_03_cat","ps_car_05_cat"]

for c in cols:

    Xmat[c + "_isnan"] = Xmat[c].apply(nan2bi)

    

# For other columns replace nan with median

Xmat = Xmat.fillna(Xmat.median())



# remove other columns with nan, if any

Xmat = Xmat.dropna(axis=1)

print(Xmat.shape)
# some of binary variables can be skewed

bin_col = [col for col in Xmat.columns if '_bin' in col]

counts = []

for col in bin_col:

    counts.append(100*(Xmat[col]==1).sum()/Xmat.shape[0])



ax = sns.barplot(x=counts, y=bin_col, orient='h')

ax.set(xlabel="% of 1 in a column")

plt.show()

# upon visual inspection, some columns with skewed data are removed

Xmat = Xmat.drop(["ps_ind_10_bin","ps_ind_11_bin","ps_ind_12_bin","ps_ind_13_bin"], axis=1)

print(Xmat.shape)
# check correlation matrix

sns.set(style="white")



# Compute the correlation matrix (let's put y_train back this time)

Xcorrmat = Xmat.iloc[:X_train.shape[0],:]

Xcorrmat['target'] = y_train

corr = Xcorrmat.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20,12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
# drop all the "_calc"

calc_col = [col for col in Xmat.columns if '_calc' in col]

Xmat = Xmat.drop(calc_col, axis=1)



# zscoring as a means of normalization

X_train = Xmat.iloc[:X_train.shape[0],:]

X_test = Xmat.iloc[X_train.shape[0]:,:]

X_train = (X_train - X_train.mean())/X_train.std()

X_test = (X_test - X_test.mean())/X_test.std()



# vizualize

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(X_train, cmap=cmap)

plt.show()

# outlier deletion

X_train = X_train.drop(['ps_ind_14','ps_car_10_cat'], axis=1)

X_test = X_test.drop(['ps_ind_14','ps_car_10_cat'], axis=1)



X_train = (X_train - X_train.mean())/X_train.std()

X_test = (X_test - X_test.mean())/X_test.std()



# vizualize

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(X_train, cmap=cmap)

plt.show()

# feature importance using random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')

rf.fit(X_train, y_train)



print('Training done using Random Forest')



ranking = np.argsort(-rf.feature_importances_)

f, ax = plt.subplots(figsize=(11, 9))

sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')

ax.set_xlabel("feature importance")

plt.show()

# dimensioanlity reduction and visualization

Xdr = X_train

Xdr['target'] = y_train

Xdr1 = Xdr.loc[y_train==1, :]

Xdr0 = Xdr.loc[y_train==0, :]



print('rows for target 1: ' + str(Xdr1.shape[0]))

print('rows for target 0: ' + str(Xdr0.shape[0]))
# random sampling from X_train0, as the target value distribution is skewed

# use N = 20,000 samples for now

N = 20000

np.random.seed(20171021)

Xdr0 = Xdr0.iloc[np.random.choice(Xdr0.shape[0], N), :]

Xdr1 = Xdr1.iloc[np.random.choice(Xdr1.shape[0], N), :]



Xdr = pd.concat([Xdr0, Xdr1])
# pairplot

Xpair =pd.concat([Xdr.iloc[:,ranking[:7]], Xdr['target']], axis=1)



ax = sns.pairplot(Xpair, hue='target')

plt.show()
Xdr = Xdr.drop(['target'], axis=1)



# PCA

from sklearn.decomposition import PCA



pcamat = PCA(n_components=2).fit_transform(Xdr)



plt.figure()

plt.scatter(pcamat[:Xdr0.shape[0],0],pcamat[:Xdr0.shape[0],1], c='b', label='targ 0', alpha=0.3)

plt.scatter(pcamat[Xdr0.shape[0]:,0],pcamat[Xdr0.shape[0]:,1],c='r', label='targ 1', alpha=0.3)

plt.legend()

plt.title('PC space')

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.tight_layout()

plt.show()

print("PCA done")

# TSNE 

from sklearn.manifold import TSNE



tsnemat = TSNE(n_components=2, random_state=0).fit_transform(Xdr)



plt.figure()

plt.scatter(tsnemat[:Xdr0.shape[0],0],tsnemat[:Xdr0.shape[0],1], c='b', label='targ 0', alpha=0.3)

plt.scatter(tsnemat[Xdr0.shape[0]:,0],tsnemat[Xdr0.shape[0]:,1],c='r', label='targ 1', alpha=0.3)

plt.legend()

plt.title('TSNE space')

plt.xlabel('dim 1')

plt.ylabel('dim 2')

plt.tight_layout()

plt.show()

print("TSNE done")
