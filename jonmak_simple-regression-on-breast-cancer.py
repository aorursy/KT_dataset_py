import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

from sklearn import linear_model

from sklearn.decomposition import PCA

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score, KFold
df = pd.read_csv('../input/data.csv')

df = df.drop(df.columns[-1], 1)



X = df.drop(['id', 'diagnosis'], 1)

y = df['diagnosis']
def plot_corr(df=df):

    cor = df.drop('id', 1).corr()

    mask = np.zeros_like(cor, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True

    f, ax1 = plt.subplots()

    sns.heatmap(cor, mask=mask,square=True, ax=ax1)

    plt.xticks(rotation=90)

    plt.yticks(rotation=0)

plot_corr()
def boxplot(X=X,y=y):

    f, ax = plt.subplots(5, 6, sharex=True)

    f.tight_layout()

    for i in range(len(ax.flatten())):

        var = X.columns[i]

        sns.boxplot(y, X[var], ax=ax.flatten()[i])

    plt.suptitle('Distribution of variables against Diagnosis')

boxplot()
def logreg(X=X,y=y):

    kf = KFold(n_splits=8, shuffle=True)

    lr = linear_model.LogisticRegression(penalty='l2', C=1e20, solver='liblinear')

    acc = cross_val_score(lr, X, y, cv=kf)

    return(acc.mean())



def logreg_L1(X=X,y=y):

    kf = KFold(n_splits=8, shuffle=True)

    lasso = linear_model.LogisticRegression(penalty='l1', C=1, solver='liblinear')

    acc = cross_val_score(lasso, X, y, cv=kf)

    return(acc.mean())



def logreg_L2(X=X,y=y):

    kf = KFold(n_splits=8, shuffle=True)

    ridge = linear_model.LogisticRegression(penalty='l2', C=1, solver='liblinear')

    acc = cross_val_score(ridge, X, y, cv=kf)

    return(acc.mean())



print('No regularisation: ' + str(logreg())) 

print('Full Lasso: ' + str(logreg_L1()))

print('Full Ridge: ' + str(logreg_L2()))
def pca_X(X=X):

    pca = PCA(random_state=10)

    X_prep = preprocessing.scale(X)

    pca.fit(X_prep)

    X_pca = pca.fit_transform(X_prep)

    return(X_pca)

X_pca = pca_X()

print('No regularisation, PCA: ' + str(logreg(X_pca)))

print('Full Lasso, PCA: ' + str(logreg_L1(X_pca)))

print('Full Ridge, PCA: ' + str(logreg_L2(X_pca)))