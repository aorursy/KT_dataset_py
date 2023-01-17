ver = 'tsne_v14'

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook as tqdm



%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use(['seaborn-darkgrid'])

plt.rcParams['font.family'] = 'DejaVu Sans'



from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler, Binarizer, KernelCenterer

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, Lasso, Ridge

from sklearn.cluster import DBSCAN



from sklearn.feature_selection import RFE



import eli5



from sklearn.model_selection import validation_curve, learning_curve, cross_val_score, train_test_split

from sklearn.metrics import roc_auc_score

import os

print(os.listdir("../input"))



RANDOM_STATE = 78



from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.shape, test.shape
train.describe()
train['target'].value_counts().sort_index(ascending=False).plot(kind='barh', 

                                                                          figsize=(15,6))

plt.title('Target', fontsize=18)
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']

X_tst = test.drop(['id'], axis=1)
X_train.head()
best_param = {'C': 0.11248300958542848, 'class_weight': 'balanced', 'max_iter': 50000, 'n_jobs': -1, 'penalty': 'l1', 'random_state': 78, 'solver': 'liblinear'}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg0 = LogisticRegression(**best_param)

logreg1 = LogisticRegression(**best_param)

logreg2 = LogisticRegression(**best_param)

logreg3 = LogisticRegression(**best_param)
sc0 = StandardScaler()

sc1 = RobustScaler()

sc2 = Binarizer()

sc3 = KernelCenterer()

X_train0 = sc0.fit_transform(X_train)

X_test0 = sc0.transform(X_tst)

X_train1 = sc1.fit_transform(X_train)

X_test1 = sc1.transform(X_tst)

X_train2 = sc2.fit_transform(X_train)

X_test2 = sc2.transform(X_tst)

X_train3 = sc3.fit_transform(X_train)

X_test3 = sc3.transform(X_tst)
logreg0.fit(X_train0,y_train)

sc = cross_val_score(logreg0, X_train0, y_train, scoring='roc_auc', cv=kfold, n_jobs=-1, verbose=1)

print(sc.mean())



logreg1.fit(X_train1,y_train)

sc = cross_val_score(logreg1, X_train1, y_train, scoring='roc_auc', cv=kfold, n_jobs=-1, verbose=1)

print(sc.mean())



logreg2.fit(X_train2,y_train)

sc = cross_val_score(logreg2, X_train2, y_train, scoring='roc_auc', cv=kfold, n_jobs=-1, verbose=1)

print(sc.mean())



logreg3.fit(X_train3,y_train)

sc = cross_val_score(logreg3, X_train3, y_train, scoring='roc_auc', cv=kfold, n_jobs=-1, verbose=1)

print(sc.mean())
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
pca = PCA().fit(X_train)



plt.figure(figsize=(10,7))

plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)

plt.xlabel('Number of components')

plt.ylabel('Total explained variance')

plt.xlim(0, 300)

plt.yticks(np.arange(0, 1.1, 0.1))



plt.axhline(0.9, c='c')

plt.show();
fig, ax = plt.subplots(2, 2, figsize = (24, 24))

pca = PCA(n_components=0.9)

X_reduced0 = pca.fit_transform(X_train0)

X_reduced1 = pca.fit_transform(X_train1)

X_reduced2 = pca.fit_transform(X_train2)

X_reduced3 = pca.fit_transform(X_train3)



#print('Projecting %d-dimensional data to 2D' % X_train.shape[1])



ax[0,0].scatter(X_reduced0[:, 0], X_reduced0[:, 1], c=y_train,

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[0,0].set_title('PCA projection StdScalar')



ax[1,0].scatter(X_reduced1[:, 0], X_reduced1[:, 1], c=y_train,

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[1,0].set_title('PCA projection Robust')



ax[0,1].scatter(X_reduced2[:, 0], X_reduced2[:, 1], c=y_train,

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[0,1].set_title('PCA projection Binarizer')



ax[1,1].scatter(X_reduced3[:, 0], X_reduced3[:, 1], c=y_train,

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[1,1].set_title('PCA projection KernelCenterer')



print(pca.n_components_)
fig, ax = plt.subplots(2, 2, figsize = (24, 24))

pca = PCA(n_components=0.9)

X_reduced0 = pca.fit_transform(X_test0)

X_reduced1 = pca.fit_transform(X_test1)

X_reduced2 = pca.fit_transform(X_test2)

X_reduced3 = pca.fit_transform(X_test3)



#print('Projecting %d-dimensional data to 2D' % X_train.shape[1])



ax[0,0].scatter(X_reduced0[:, 0], X_reduced0[:, 1], c=logreg0.predict(X_test0),

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[0,0].set_title('PCA projection StdScalar')



ax[1,0].scatter(X_reduced1[:, 0], X_reduced1[:, 1], c=logreg1.predict(X_test1),

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[1,0].set_title('PCA projection Robust')



ax[0,1].scatter(X_reduced2[:, 0], X_reduced2[:, 1], c=logreg2.predict(X_test2),

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[0,1].set_title('PCA projection Binarizer')



ax[1,1].scatter(X_reduced3[:, 0], X_reduced3[:, 1], c=logreg3.predict(X_test3),

            edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

#plt.colorbar()

ax[1,1].set_title('PCA projection KernelCenterer')



print(pca.n_components_)
(fig, subplots) = plt.subplots(2, 5, figsize=(15, 5))

perplexities1 = [2, 4, 5, 6]

perplexities2 = [8, 9, 10, 11]



tsne = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE).fit_transform(X_train)

x_tsne = (tsne.T[0]).T

y_tsne = (tsne.T[1]).T

subplots[0,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[0,0].set_title("Perplexity=%d" % 0)   



for i, perplexity in enumerate(perplexities1):

    ax = subplots[0][i + 1]



    tsne = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train)

    x_tsne = (tsne.T[0]).T

    y_tsne = (tsne.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)



tsne = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train)

x_tsne = (tsne.T[0]).T

y_tsne = (tsne.T[1]).T

subplots[1,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[1,0].set_title("Perplexity=%d" % 7)   



for i, perplexity in enumerate(perplexities2):

    ax = subplots[1][i + 1]



    tsne = TSNE(n_components=2, init='pca',random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train)

    x_tsne = (tsne.T[0]).T

    y_tsne = (tsne.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)    



print('Without scaler')



(fig, subplots) = plt.subplots(2, 5, figsize=(15, 5))

perplexities1 = [2, 4, 5, 6]

perplexities2 = [8, 9, 10, 11]



tsne0 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE).fit_transform(X_train0)

x_tsne = (tsne0.T[0]).T

y_tsne = (tsne0.T[1]).T

subplots[0,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[0,0].set_title("Perplexity=%d" % 0)   



for i, perplexity in enumerate(perplexities1):

    ax = subplots[0][i + 1]



    tsne0 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train0)

    x_tsne = (tsne0.T[0]).T

    y_tsne = (tsne0.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)



tsne0 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train0)

x_tsne = (tsne0.T[0]).T

y_tsne = (tsne0.T[1]).T

subplots[1,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[1,0].set_title("Perplexity=%d" % 7)   



for i, perplexity in enumerate(perplexities2):

    ax = subplots[1][i + 1]



    tsne0 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train0)

    x_tsne = (tsne0.T[0]).T

    y_tsne = (tsne0.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)    



print('With StandartScaler')



(fig, subplots) = plt.subplots(2, 5, figsize=(15, 5))

perplexities1 = [2, 4, 5, 6]

perplexities2 = [8, 9, 10, 11]



tsne1 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE).fit_transform(X_train1)

x_tsne = (tsne1.T[0]).T

y_tsne = (tsne1.T[1]).T

subplots[0,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[0,0].set_title("Perplexity=%d" % 0)   



for i, perplexity in enumerate(perplexities1):

    ax = subplots[0][i + 1]



    tsne1 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train1)

    x_tsne = (tsne1.T[0]).T

    y_tsne = (tsne1.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)



tsne1 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train1)

x_tsne = (tsne1.T[0]).T

y_tsne = (tsne1.T[1]).T

subplots[1,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[1,0].set_title("Perplexity=%d" % 7)   



for i, perplexity in enumerate(perplexities2):

    ax = subplots[1][i + 1]



    tsne1 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train1)

    x_tsne = (tsne1.T[0]).T

    y_tsne = (tsne1.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)    



print('With RobustScaler')
(fig, subplots) = plt.subplots(2, 5, figsize=(15, 5))

perplexities1 = [2, 4, 5, 6]

perplexities2 = [8, 9, 10, 11]



tsne2 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE).fit_transform(X_train2)

x_tsne = (tsne2.T[0]).T

y_tsne = (tsne2.T[1]).T

subplots[0,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[0,0].set_title("Perplexity=%d" % 0)   



for i, perplexity in enumerate(perplexities1):

    ax = subplots[0][i + 1]



    tsne2 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train2)

    x_tsne = (tsne2.T[0]).T

    y_tsne = (tsne2.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)



tsne0 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train2)

x_tsne = (tsne2.T[0]).T

y_tsne = (tsne2.T[1]).T

subplots[1,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

subplots[1,0].set_title("Perplexity=%d" % 7)   



for i, perplexity in enumerate(perplexities2):

    ax = subplots[1][i + 1]



    tsne2 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train2)

    x_tsne = (tsne2.T[0]).T

    y_tsne = (tsne2.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)    



print('With Binarizer')
(fig, subplots) = plt.subplots(2, 5, figsize=(15, 5))

perplexities1 = [2, 4, 5, 6]

perplexities2 = [8, 9, 10, 11]



tsne3 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE).fit_transform(X_train3)

x_tsne = (tsne3.T[0]).T

y_tsne = (tsne3.T[1]).T

subplots[0,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

subplots[0,0].set_title("Perplexity=%d" % 0)   



for i, perplexity in enumerate(perplexities1):

    ax = subplots[0][i + 1]



    tsne3 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train3)

    x_tsne = (tsne3.T[0]).T

    y_tsne = (tsne3.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)



tsne3 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train3)

x_tsne = (tsne3.T[0]).T

y_tsne = (tsne3.T[1]).T

subplots[1,0].scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

subplots[1,0].set_title("Perplexity=%d" % 7)   



for i, perplexity in enumerate(perplexities2):

    ax = subplots[1][i + 1]



    tsne3 = TSNE(n_components=2, init='pca', random_state=RANDOM_STATE, perplexity=perplexity).fit_transform(X_train3)

    x_tsne = (tsne3.T[0]).T

    y_tsne = (tsne3.T[1]).T

    ax.scatter(x_tsne, y_tsne, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

    

    ax.set_title("Perplexity=%d" % perplexity)    



print('With KernelCenterer')
tsne0 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train0)

tsne1 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train1)

tsne2 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train2)

tsne3 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_train3)



x_tsne0 = (tsne0.T[0]).T

y_tsne0 = (tsne0.T[1]).T

x_tsne1 = (tsne1.T[0]).T

y_tsne1 = (tsne1.T[1]).T

x_tsne2 = (tsne2.T[0]).T

y_tsne2 = (tsne2.T[1]).T

x_tsne3 = (tsne3.T[0]).T

y_tsne3 = (tsne3.T[1]).T
fig, ax = plt.subplots(2, 2, figsize = (24, 24))

ax[0,0].scatter(x_tsne0, y_tsne0, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

ax[0,0].set_title('tSNE StdScalar' );



ax[0,1].scatter(x_tsne1, y_tsne1, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

ax[0,1].set_title('tSNE Robust' );



ax[1,0].scatter(x_tsne2, y_tsne2, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

ax[1,0].set_title('tSNE Binarizer' );



ax[1,1].scatter(x_tsne3, y_tsne3, c = y_train, edgecolor='none', alpha=0.7, s=40,

            cmap=plt.cm.get_cmap('bwr', 2))

ax[1,1].set_title('tSNE KernelCenterer' );

X_test0.shape
tsne0 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test0[:5000])

tsne1 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test1[:5000])

tsne2 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test2[:5000])

tsne3 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test3[:5000])



x_tsne0 = (tsne0.T[0]).T

y_tsne0 = (tsne0.T[1]).T

x_tsne1 = (tsne1.T[0]).T

y_tsne1 = (tsne1.T[1]).T

x_tsne2 = (tsne2.T[0]).T

y_tsne2 = (tsne2.T[1]).T

x_tsne3 = (tsne3.T[0]).T

y_tsne3 = (tsne3.T[1]).T
predict0 = logreg0.predict(X_test0[:5000])

predict1 = logreg1.predict(X_test1[:5000])

predict2 = logreg2.predict(X_test2[:5000])

predict3 = logreg3.predict(X_test3[:5000])
fig, ax = plt.subplots(2, 2, figsize = (24, 24))

ax[0,0].scatter(x_tsne0, y_tsne0, c = predict0, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[0,0].set_title('tSNE StdScalar' );



ax[0,1].scatter(x_tsne1, y_tsne1, c = predict1, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[0,1].set_title('tSNE Robust' );



ax[1,0].scatter(x_tsne2, y_tsne2, c = predict2, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[1,0].set_title('tSNE Binarizer' );



ax[1,1].scatter(x_tsne3, y_tsne3, c = predict3, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[1,1].set_title('tSNE KernelCenterer' );
dbscan = DBSCAN(eps=6, n_jobs=-1)

clast = dbscan.fit_predict(pd.DataFrame([x_tsne0,y_tsne0]).T.values)


plt.scatter(x_tsne0, y_tsne0, c = clast, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

plt.colorbar()

plt.title('tSNE DBSCAN StdScalar' );
tsne0 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test0[5000:10000])

tsne1 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test1[5000:10000])

tsne2 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test2[5000:10000])

tsne3 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test3[5000:10000])



x_tsne0 = (tsne0.T[0]).T

y_tsne0 = (tsne0.T[1]).T

x_tsne1 = (tsne1.T[0]).T

y_tsne1 = (tsne1.T[1]).T

x_tsne2 = (tsne2.T[0]).T

y_tsne2 = (tsne2.T[1]).T

x_tsne3 = (tsne3.T[0]).T

y_tsne3 = (tsne3.T[1]).T
predict0 = logreg0.predict(X_test0[5000:10000])

predict1 = logreg1.predict(X_test1[5000:10000])

predict2 = logreg2.predict(X_test2[5000:10000])

predict3 = logreg3.predict(X_test3[5000:10000])


fig, ax = plt.subplots(2, 2, figsize = (24, 24))

ax[0,0].scatter(x_tsne0, y_tsne0, c = predict0, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[0,0].set_title('tSNE StdScalar' );



ax[0,1].scatter(x_tsne1, y_tsne1, c = predict1, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[0,1].set_title('tSNE Robust' );



ax[1,0].scatter(x_tsne2, y_tsne2, c = predict2, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[1,0].set_title('tSNE Binarizer' );



ax[1,1].scatter(x_tsne3, y_tsne3, c = predict3, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[1,1].set_title('tSNE KernelCenterer' );
dbscan = DBSCAN(eps=6, n_jobs=-1)

clast = dbscan.fit_predict(pd.DataFrame([x_tsne0,y_tsne0]).T.values)



plt.scatter(x_tsne0, y_tsne0, c = clast, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

plt.colorbar()

plt.title('tSNE DBSCAN StdScalar [5000:10000]');
tsne0 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test0)

tsne1 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test1)

tsne2 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test2)

tsne3 = TSNE(init='pca', random_state=RANDOM_STATE, perplexity=7).fit_transform(X_test3)



x_tsne0 = (tsne0.T[0]).T

y_tsne0 = (tsne0.T[1]).T

x_tsne1 = (tsne1.T[0]).T

y_tsne1 = (tsne1.T[1]).T

x_tsne2 = (tsne2.T[0]).T

y_tsne2 = (tsne2.T[1]).T

x_tsne3 = (tsne3.T[0]).T

y_tsne3 = (tsne3.T[1]).T
predict0 = logreg0.predict(X_test0)

predict1 = logreg1.predict(X_test1)

predict2 = logreg2.predict(X_test2)

predict3 = logreg3.predict(X_test3)
fig, ax = plt.subplots(2, 2, figsize = (24, 24))

ax[0,0].scatter(x_tsne0, y_tsne0, c = predict0, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[0,0].set_title('tSNE StdScalar' );



ax[0,1].scatter(x_tsne1, y_tsne1, c = predict1, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[0,1].set_title('tSNE Robust' );



ax[1,0].scatter(x_tsne2, y_tsne2, c = predict2, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[1,0].set_title('tSNE Binarizer' );



ax[1,1].scatter(x_tsne3, y_tsne3, c = predict3, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

ax[1,1].set_title('tSNE KernelCenterer' );
dbscan = DBSCAN(eps=6, n_jobs=-1)

clast = dbscan.fit_predict(pd.DataFrame([x_tsne0,y_tsne0]).T.values)



plt.scatter(x_tsne0, y_tsne0, c = clast, edgecolor='none', alpha=0.7, s=40, cmap=plt.cm.get_cmap('bwr', 2))

plt.colorbar()

plt.title('tSNE DBSCAN StdScalar' );
#tsne0 = TSNE(n_components=2, init='pca').fit_transform(X_tst)



#x_tsne0 = (tsne0.T[0]).T

#y_tsne0 = (tsne0.T[1]).T
#fig, ax = plt.subplots(1, 1, figsize = (12, 12))

#ax.scatter(x_tsne0, y_tsne0, c = logreg1.predict(X_tst), edgecolor='none', alpha=0.7, s=40,

            #cmap=plt.cm.get_cmap('bwr', 2))

#ax.set_title('tSNE ' );