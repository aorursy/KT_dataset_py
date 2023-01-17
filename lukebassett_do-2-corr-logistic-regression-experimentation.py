import numpy as np

import pandas as pd



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

X = np.array(train.drop(['id','target'], axis=1))

y = np.array(train['target'])

X_test = np.array(test.iloc[:,1:])
# get correlation of each feature with target

corr = train.corr()['target'][2:]

sns.boxplot(corr)
from sklearn.model_selection import StratifiedKFold



def repeat_cross_val(model, X, y, n_iters=10, n_folds=5):

    

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True)

    scores = np.zeros([n_iters, n_folds])

    

    for i in range(n_iters):

        for j, (cv_train, cv_test) in enumerate(folds.split(X,y)):

            model.fit(X[cv_train], y[cv_train])

            scores[i,j] = model.score(X[cv_test],y[cv_test])    

    return scores.mean()
#set correltaion threshold and filter training data

corr_thresh = 0.1

high_corr = abs(corr)>corr_thresh

X_corr = X[:,high_corr]



X_corr.shape[1]/X.shape[1]
# give it a quick test

from sklearn.linear_model import LogisticRegression



lrc = LogisticRegression(penalty='l1', solver='liblinear')

print(repeat_cross_val(model=lrc, X=X_corr, y=y))
# Testing values from 0-0.3

corr_test = np.arange(0, 0.3, 0.01)

cv_score = np.zeros(corr_test.shape[0])



lrc = LogisticRegression(penalty='l1', solver='liblinear')



for i, c in enumerate(corr_test):

    high_corr = abs(corr)>c

    X_corr = X[:,high_corr]

    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=25)



plt.scatter(x=corr_test, y=cv_score)

plt.xlabel('correlation threshold')

plt.ylabel('cv score')

plt.title('Testing correlation threshold')
corr_thresh = 0.11

high_corr = abs(corr)>corr_thresh

X_corr = X[:,high_corr]

X_corr.shape
lrc = LogisticRegression(penalty='l1', solver='liblinear')

print('l1: {0:.3f}'.format(repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=250)))

lrc = LogisticRegression(penalty='l2', solver='liblinear')

print('l2: {0:.3f}'.format(repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=250)))
# Testing values from 0-0.3

C_test = np.array([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]) 

cv_score = np.zeros(C_test.shape[0])

for i, C in enumerate(C_test):

    lrc = LogisticRegression(penalty='l2', solver='liblinear', C=C)

    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=50)

plt.scatter(x=C_test, y=cv_score)

plt.xlabel('C-value')

plt.ylabel('cv score')

plt.title('Testing C-value')

plt.xscale('log')

plt.xlim((0.00000001,10000))
%%time

C_test = np.logspace(-4.0, 0, 60)

cv_score = np.zeros(C_test.shape[0])

for i, C in enumerate(C_test):

    lrc = LogisticRegression(penalty='l2', solver='liblinear', C=C)

    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=50)

plt.scatter(x=C_test, y=cv_score)

plt.xlabel('C-value')

plt.ylabel('cv score')

plt.title('Testing C-value')

plt.xscale('log')

plt.xlim((10**-4.2,1))
# Testing values from 0-0.3

corr_test = np.arange(0, 0.3, 0.01)

cv_score = np.zeros(corr_test.shape[0])

lrc = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)

for i, c in enumerate(corr_test):

    high_corr = abs(corr)>c

    X_corr = X[:,high_corr]

    cv_score[i] = repeat_cross_val(model=lrc, X=X_corr, y=y, n_iters=25)

plt.scatter(x=corr_test, y=cv_score)

plt.xlabel('correlation threshold')

plt.ylabel('cv score')

plt.title('Testing correlation threshold')
# reset X_corr

corr_thresh = 0.11

high_corr = abs(corr)>corr_thresh

X_corr = X[:,high_corr]



lrc = LogisticRegression(penalty='l2', solver='liblinear', C=0.05)

lrc.fit(X_corr, y)
predict = lrc.predict(X_test[:,high_corr]) # this got a .704

predict_prob = lrc.predict_proba(X_test[:,high_corr]) # wow! this got a 0.786

print(predict[0], predict_prob[:,1])
sub = pd.DataFrame({

    'id': test['id'],

    'target': predict_prob[:,1]

})

print(sub.head())

print(pd.read_csv('../input/sample_submission.csv').head())
sub.to_csv('submission.csv', index=False)