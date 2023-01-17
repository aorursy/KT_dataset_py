import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn

from sklearn.preprocessing import LabelEncoder



plt.style.use('ggplot')



%matplotlib inline
iris = pd.read_csv('../input/Iris.csv')
iris.head()
from sklearn.cross_validation import train_test_split

X = iris[[c for c in iris.columns if c != "Species" and c!='Id']]

Y = iris["Species"]

Y = LabelEncoder().fit_transform(Y)

Id = iris['Id']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

type(x_train), type(y_train)
iris.shape, x_train.shape, y_train.shape
x_train.hist()
iris[['Id','Species']].groupby("Species").count().reset_index()
seaborn.pairplot(x_train)
seaborn.set(font="monospace")

cmap = seaborn.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)

seaborn.clustermap(x_train.corr(), linewidths=.5, cmap=cmap)
from sklearn import tree

clf1 = tree.DecisionTreeClassifier(min_samples_leaf=10, min_samples_split=10)

clf1.fit(x_train, y_train)
from sklearn.ensemble import AdaBoostClassifier

clf2 = AdaBoostClassifier(clf1,

                         algorithm='SAMME',

                         n_estimators=800,

                         learning_rate=0.5)

clf2.fit(x_train, y_train)
clf1.classes_, "nb classes", len(clf1.classes_)
clf2.classes_, "nb classes", len(clf2.classes_)
from sklearn.metrics import confusion_matrix

y_pred = clf1.predict(x_test)

conf1 = confusion_matrix(y_test, y_pred)

conf1
y_pred = clf2.predict(x_test)

conf2 = confusion_matrix(y_test, y_pred)

conf2
y_pred = clf2.predict(x_test)

y_prob = clf2.decision_function(x_test)

y_min = y_pred.min()

import numpy

y_score = numpy.array( [y_prob[i,p-y_min] for i,p in enumerate(y_pred)] )

y_score[:5], y_pred[:5], y_prob[:5,:]
from sklearn.metrics import roc_curve, auc

fpr = dict()

tpr = dict()

roc_auc = dict()

nb_obs = dict()

for i in clf2.classes_:

    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_score)

    roc_auc[i] = auc(fpr[i], tpr[i])

    nb_obs[i] = (y_test == i).sum()

roc_auc, nb_obs
i = "all"

fpr[i], tpr[i], _ = roc_curve(y_test == y_pred, y_score)

roc_auc[i] = auc(fpr[i], tpr[i])

nb_obs[i] = (y_test == y_pred).sum()

roc_auc_best, nb_obs_best = roc_auc, nb_obs

roc_auc, nb_obs
fig, axes = plt.subplots(1,2, figsize=(14,5), sharey=True)

for cl in (0,1,2):

    axes[0].plot(fpr[cl], tpr[cl], label='ROC classe %d (area = %0.2f)' % (cl, roc_auc[cl]))

for cl in ("all",):

    axes[1].plot(fpr[cl], tpr[cl], label='ROC classe (area = %0.2f)' % (roc_auc[cl]))

axes[0].legend()

axes[1].legend()
clft = tree.DecisionTreeClassifier(min_samples_leaf=10, min_samples_split=10)

clf4 = AdaBoostClassifier(clft,

                         algorithm='SAMME',

                         n_estimators=800,

                         learning_rate=0.5)

clf4.fit(x_train, y_train)
from sklearn.metrics import auc, precision_recall_curve

y_pred4 = clf4.predict(x_test)

y_prob4 = clf4.predict_proba(x_test)

y_min4 = y_pred4.min()

import numpy

y_score4 = numpy.array( [y_prob4[i,p-y_min4] for i,p in enumerate(y_pred4)] )

y_score4[:5]

precision = dict()

recall = dict()

threshold = dict()

nb_obs = dict()

for i in clf4.classes_:

    precision[i], recall[i], threshold[i] = precision_recall_curve(y_test == i, y_score4)

    nb_obs[i] = (y_test == i).sum()

i = "all"

precision[i], recall[i], threshold[i] = precision_recall_curve(y_test == y_pred4, y_score4)

nb_obs[i] = (y_test == y_pred4).sum()
fig, axes = plt.subplots(1,2, figsize=(14,5))

for cl in (0,1,2,'all'):

    axes[0].plot(precision[cl], recall[cl], label='Precision/Rappel classe %s' % str(cl))

cl = 'all'

axes[1].plot(threshold[cl], recall[cl][1:], label='recall - all')

axes[1].plot(threshold[cl], precision[cl][1:], label='precision - all')

axes[1].set_xlabel("threshold")

axes[0].set_xlabel("precision")

axes[0].set_ylabel("recall")

axes[0].legend()

axes[1].legend()
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
X_scale = StandardScaler().fit_transform(x_train)



# Sklearn TSNE



tsne = TSNE(n_components=2, init='pca', random_state=0)

X_t = tsne.fit_transform(X_scale)



plt.scatter(X_t[np.where(y_train == 0), 0],

                   X_t[np.where(y_train == 0), 1],

                   marker='x', color='g',

                   linewidth='1', alpha=0.8, label='Species 1')

plt.scatter(X_t[np.where(y_train == 1), 0],

                   X_t[np.where(y_train == 1), 1],

                   marker='v', color='r',

                   linewidth='1', alpha=0.8, label='Species 2')



plt.scatter(X_t[np.where(y_train == 2), 0],

                   X_t[np.where(y_train == 2), 1],

                   marker='o', color='b',

                   linewidth='1', alpha=0.8, label='Species 3')



plt.xlabel('Dim 1')

plt.ylabel('Dim 2')

plt.title('T-SNE')

plt.legend(loc='best')
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA



fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(x_train)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_train,

           cmap=plt.cm.Paired)

ax.set_title("First three PCA directions")

ax.set_xlabel("1st eigenvector")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("2nd eigenvector")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("3rd eigenvector")

ax.w_zaxis.set_ticklabels([])



plt.show()