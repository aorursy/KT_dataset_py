# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt



import xgboost as xgb



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
db = pd.read_csv('../input/creditcard.csv')

db.head()
db.describe()
fraud = db[db.Class == 1]

normal = db[db.Class == 0]
# Feature extraction based on the method of Currie32

# https://www.kaggle.com/currie32/d/dalpozz/creditcardfraud/predicting-fraud-with-tensorflow



features = db.ix[:,1:29].columns

plt.figure(figsize = (12, 28*4))

gs = gridspec.GridSpec(28, 1)



for i, cn in enumerate(db[features]):

    ax = plt.subplot(gs[i])

    sns.distplot(fraud[cn])

    sns.distplot(normal[cn])

    ax.set_xlabel('')

    ax.set_title('Histogram of feature {}'.format(str(cn)))
# Unsupervised with T-SNE, where similar distribution features are dropped

db2 = pd.read_csv('../input/creditcard.csv')

db2.drop(['V8', 'V13', 'V15', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], axis = 1)

db_tsne = pd.concat([db2[db2.Class == 1], db2[db2.Class == 0].sample(n = 1000)], axis = 0)
scaler = StandardScaler()

y = db_tsne.ix[:,-1].values

db_tsne = db_tsne.drop('Class', axis = 1)

db_std = scaler.fit_transform(db_tsne)
tsne = TSNE(n_components = 2, perplexity = 50, n_iter = 2000, verbose = 1)

tsne_2d = tsne.fit_transform(db_std)

plt.figure()

plt.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c = y)
# Drop similar distribution features

db = pd.read_csv('../input/creditcard.csv')

db.drop(['V8', 'V13', 'V15', 'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'], axis = 1)

y = db.pop('Class')

X = db
X_train, X_test = train_test_split(X, test_size = 0.3, random_state = 42)

y_train, y_test = train_test_split(y, test_size = 0.3, random_state = 42)
# Train xgboost to detection credit card fraud. Because the set is very inbalanced we measure the performance

# in the area under the curve of the ROC.

dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test, y_test)

param = {'max_depth' : 3, 'eta' : 0.1, 'objective' : 'binary:logistic', 'eval_metric' : 'auc' ,'seed' : 42}

num_round = 200

bst = xgb.train(param, dtrain, num_round, [(dtest, 'test'), (dtrain, 'train')])
xgb.plot_importance(bst)
preds = bst.predict(dtest)

fpr, tpr, thresholds = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)
plt.clf()

plt.plot(fpr, tpr, label = 'ROC curve (area = {})'.format(roc_auc))

plt.xscale('log')

plt.xlim([0.000001, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False positive rate')

plt.xlabel('True positive rate')

plt.title('Credit card fraud detection ROC')

plt.legend(loc = 'lower right')