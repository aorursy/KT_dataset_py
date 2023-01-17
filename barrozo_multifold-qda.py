# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%load_ext autoreload

%autoreload 2



%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.decomposition import PCA

from sklearn.externals import joblib

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.hist(bins=50, figsize=(80,60))

plt.show()
train['wheezy-copper-turtle-magic'].hist()
# PLOT FIRST 8 VARIABLES

plt.figure(figsize=(15,15))

for i in range(8):

    plt.subplot(3,3,i+1)

    sns.distplot(train.iloc[:,i+1],bins=100)

    plt.title(train.columns[i+1] )

    plt.xlabel('')

    

# PLOT GAUSSIAN FOR COMPARISON

plt.subplot(3,3,9)

std = round(np.std(train.iloc[:,8]),2)

data = np.random.normal(0,std,len(train))

sns.distplot(data,bins=100)

plt.xlim((-17,17))

plt.ylim((0,0.37))

plt.title("Gaussian with m=0, std="+str(std))



plt.subplots_adjust(hspace=0.3)

plt.show()
# NORMALITY PLOTS FOR FIRST 8 VARIABLES

plt.figure(figsize=(15,15))

for i in range(8):

    plt.subplot(3,3,i+1)

    stats.probplot(train.iloc[:,i+1], plot=plt)

    plt.title(train.columns[i+1] )

    

# NORMALITY PLOT FOR GAUSSIAN

plt.subplot(3,3,9)

stats.probplot(data, plot=plt)   

plt.title("Gaussian with m=0, std="+str(std))



plt.subplots_adjust(hspace=0.4)

plt.show()
train0 = train[train['wheezy-copper-turtle-magic']==0 ]



# PLOT FIRST 8 VARIABLES

plt.figure(figsize=(15,15))

for i in range(8):

    plt.subplot(3,3,i+1)

    #plt.hist(train0.iloc[:,i+1],bins=10)

    sns.distplot(train0.iloc[:,i+1],bins=10)

    plt.title(train.columns[i+1] )

    plt.xlabel('')

    

# PLOT GAUSSIAN FOR COMPARISON

plt.subplot(3,3,9)

std0 = round(np.std(train0.iloc[:,8]),2)

data0 = np.random.normal(0,std0,2*len(train0))

sns.distplot(data0,bins=10)

plt.xlim((-17,17))

plt.ylim((0,0.1))

plt.title("Gaussian with m=0, std="+str(std0))

    

plt.subplots_adjust(hspace=0.3)

plt.show()



# NORMALITY PLOTS FOR FIRST 8 VARIABLES

plt.figure(figsize=(15,15))

for i in range(8):

    plt.subplot(3,3,i+1)

    stats.probplot(train0.iloc[:,i+1], plot=plt)

    plt.title(train.columns[i+1] )

    

# NORMALITY PLOT FOR GAUSSIAN

plt.subplot(3,3,9)

stats.probplot(data0, plot=plt)   

plt.title("Gaussian with m=0, std="+str(std0))



plt.subplots_adjust(hspace=0.4)

plt.show()
from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# INITIALIZE VARIABLES

oof = np.zeros(len(train))

preds = np.zeros(len(test))

attribs = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



# BUILD 512 SEPARATE NON-LINEAR MODELS

for i in range(512):

    

    if i % 100 == 0:

        print(f'Entering model {i}')

    

    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

    sel = VarianceThreshold(threshold=1.5).fit(train2[attribs])

    train3 = sel.transform(train2[attribs])

    test3 = sel.transform(test2[attribs])

        

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=25, random_state=42)

    for train_index, test_index in skf.split(train3, train2['target']):

        

        # MODEL WITH SU

        clf = QuadraticDiscriminantAnalysis(reg_param=0.6) # I HAD TO PLAY AROUND WITH THE REGULARIZATION. THIS SEEMS TO BE A GOOD VALUE

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        

# PRINT VALIDATION CV AUC

auc = roc_auc_score(train['target'],oof)

print('CV score =',round(auc,5))
sample = pd.read_csv('../input/sample_submission.csv')

sample.target = preds

sample.to_csv("submission.csv", index=False)