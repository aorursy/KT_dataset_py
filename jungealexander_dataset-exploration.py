# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from collections import Counter

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn import manifold, metrics, model_selection, svm

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cc_df = pd.DataFrame.from_csv('../input/creditcard.csv', index_col=None)

cc_df.head()
print('Loaded {:d} instances and {:d} variables.'.format(len(cc_df), len(cc_df.columns)))

count = Counter(cc_df.loc[:,'Class'])

print('Genuine instances = {:d}; fraudulent instances = {:d}'.format(count[0], count[1]))

print('In test: ratio fraudulent:genuine = {:f}'.format(count[1]/count[0]))
sample_size = 2000

cc_sample = cc_df.sample(n=sample_size, random_state=0, axis=0)



t0 = time.time()

tsne = manifold.TSNE(n_components=2, init='random', random_state=42)

Y = tsne.fit_transform(cc_sample.drop(['Time', 'Amount', 'Class'], 1))

t1 = time.time()

print("t-SNE: %.2g sec" % (t1 - t0))
plt.figure(figsize=(8, 10))

plt.figure(1)

plt.subplot(311)

plt.scatter(Y[:, 0], Y[:, 1], c=cc_sample.loc[:,'Time'], cmap=plt.cm.Spectral)

plt.title('Time')

plt.axis('tight')



amount_scaled = np.log(cc_sample.loc[:,'Amount'])

plt.subplot(312)

plt.scatter(Y[:, 0], Y[:, 1], c=amount_scaled, cmap=plt.cm.Spectral)

plt.title('Amount')

plt.axis('tight')



plt.subplot(313)

plt.scatter(Y[:, 0], Y[:, 1], c=cc_sample.loc[:,'Class'], cmap=plt.cm.Spectral)

plt.title('Class')

plt.axis('tight')

plt.show()
v = sns.violinplot(x='Class', y='Time', data=cc_df)
sns.violinplot(x='Class', y='Amount', data=cc_df)
classes = cc_df['Class']

log_amount = np.log(cc_df['Amount'])

mask = np.isfinite(log_amount)

print('Removing {:d} infinte values after log-transform.'.format(np.logical_not(mask).sum()))

classes_removed = classes[np.logical_not(mask)]

count_removed = Counter(classes_removed)

print('In removed: Genuine instances = {:d}; fraudulent instances = {:d}'.format(

    count_removed[0], count_removed[1]))

print('In removed: ratio fraudulent:genuine = {:f}'.format(count_removed[1]/count_removed[0]))

log_amount = log_amount[mask]

classes = classes[mask]
sns.violinplot(x=classes, y=log_amount)
cc_fraud = cc_df.loc[cc_df['Class'].isin([1]),:]

len(cc_fraud)

t0 = time.time()

tsne = manifold.TSNE(n_components=2, init='random', random_state=42)

Y = tsne.fit_transform(cc_fraud.drop(['Time', 'Amount', 'Class'], 1))

t1 = time.time()

print("t-SNE: %.2g sec" % (t1 - t0))



plt.figure(figsize=(8, 10))

plt.figure(1)

plt.subplot(311)

plt.scatter(Y[:, 0], Y[:, 1], c=cc_fraud.loc[:,'Time'], cmap=plt.cm.Spectral)

plt.title('Time - fraudulent only')

plt.axis('tight')



amount_scaled = np.log(cc_fraud.loc[:,'Amount'])

plt.subplot(312)

plt.scatter(Y[:, 0], Y[:, 1], c=amount_scaled, cmap=plt.cm.Spectral)

plt.title('Amount  - fraudulent only')

plt.axis('tight')
train_frac = 0.7 

test_frac = 1 - train_frac

split_index = int(len(cc_df) * train_frac)

cc_train = cc_df.loc[:split_index,:]

cc_test = cc_df.loc[split_index:,:]

#cc_train, cc_test = model_selection.train_test_split(cc_df, test_size=0.3,

#                                                     random_state=0, stratify=cc_df['Class'])
print('Trainining set contains {:d} instances and {:d} variables.'.format(

    len(cc_train), len(cc_train.columns)))

count_train = Counter(cc_train.loc[:,'Class'])

print('In train: Genuine instances = {:d}; fraudulent instances = {:d}'.format(

    count_train[0], count_train[1]))

print('In train: ratio fraudulent:genuine = {:f}'.format(count_train[1]/count_train[0]))

print()

print('Test set contains {:d} instances and {:d} variables.'.format(

    len(cc_test), len(cc_test.columns)))

count_test = Counter(cc_test.loc[:,'Class'])

print('In test: Genuine instances = {:d}; fraudulent instances = {:d}'.format(

    count_test[0], count_test[1]))

print('In test: ratio fraudulent:genuine = {:f}'.format(count_test[1]/count_test[0]))
cc_train_fraud = cc_train.loc[cc_train['Class'].isin([1]), :]

len(cc_train_fraud)

cc_train_genuine = cc_train.loc[cc_train['Class'].isin([0]), :].sample(

    n=len(cc_train_fraud) * 3, random_state=0, axis=0)

cc_train = pd.concat([cc_train_genuine, cc_train_fraud])



print('After sampling, trainining set contains {:d} instances and {:d} variables.'.format(

    len(cc_train), len(cc_train.columns)))

count_train = Counter(cc_train.loc[:,'Class'])

print('In train: Genuine instances = {:d}; fraudulent instances = {:d}'.format(

    count_train[0], count_train[1]))

print('In train: ratio fraudulent:genuine = {:f}'.format(count_train[1]/count_train[0]))
sns.violinplot(x='Class', y='Time', data=cc_train)
X_train = cc_train.drop(['Class'], 1)

y_train = cc_train['Class']
params = [

  #{'C': [1, 10, 100], 'kernel': ['linear']},

  {'C': [0.1, 1, 10, 100, 1000, 10000], 'gamma': [0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},

 ]

grid_search = model_selection.GridSearchCV(svm.SVC(), params, cv=5, verbose=1,

                                           scoring='roc_auc')

grid_search.fit(X_train, y_train)
print('Scoring function: {}'.format(grid_search.scorer_))



best_params = grid_search.best_params_

print("Best parameters set found :")

print(best_params)

print()



print("Grid scores:")

means = grid_search.cv_results_['mean_test_score']

stds = grid_search.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

    print()
best_estimator = grid_search.best_estimator_

X_test = cc_test.drop(['Class'], 1)

y_test = cc_test['Class']

y_hat = best_estimator.predict(X_test)

print(metrics.classification_report(y_test, y_hat))