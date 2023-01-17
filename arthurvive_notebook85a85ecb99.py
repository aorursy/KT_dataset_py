import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import model_selection, metrics

from sklearn import linear_model



def scorer(mod,X,target):

    return metrics.matthews_corrcoef(target, mod.predict(X))

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

N_lines = 1000

df=pd.read_csv("../input/train_categorical.csv",sep=",", nrows = N_lines, na_filter=False)

df_num=pd.read_csv("../input/train_numeric.csv",sep=",",nrows=N_lines, na_filter=False)
target=df_num.Response.values

folds=8

print(np.sum(target), 112/folds)

np.random.seed(1994)

shuffle_split = cross_validation.StratifiedShuffleSplit(target, n_iter = folds)



bin_feat=np.zeros(df.values.shape, dtype='int')

bin_feat[df.values!=""]=1
C_val=np.linspace(-5, 5, 10)

res=[]

for c in C_val:

    logreg = linear_model.LogisticRegression(class_weight="balanced", C=pow(10,c))

    res+=[cross_validation.cross_val_score(logreg, bin_feat, target, cv=shuffle_split, scoring=scorer, n_jobs=-1)]# scoring=scorer
#print([(np.min(i), np.max(i)) for i in res])

plt.plot(C_val, [np.mean(i) for i in res])
C_val=np.linspace(-5, 0, 1)

res=[]

for c in C_val:

    lass = linear_model.Lasso(alpha=c, normalize=True)

    res+=[cross_validation.cross_val_score(lass, bin_feat, target, cv=shuffle_split, scoring=scorer, n_jobs=-1)]
plt.plot(C_val, [np.mean(i) for i in res])
np.unique(bin_feat)