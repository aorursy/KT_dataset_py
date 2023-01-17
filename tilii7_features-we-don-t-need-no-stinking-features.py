__author__ = 'Tilii: https://kaggle.com/tilii7'



import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)

        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))



train = pd.read_csv('../input/train.csv', dtype={'id': np.int32, 'target': np.int8})

X = train.drop(['id', 'target'], axis=1).values

y = train['target'].values

test = pd.read_csv('../input/test.csv', dtype={'id': np.int32})

X_test = test.drop(['id'], axis=1).values



all_features = [x for x in train.drop(['id', 'target'], axis=1).columns]
folds = 5

step = 2



rfc = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10, n_jobs=4)



rfecv = RFECV(

              estimator=rfc,

              step=step,

              cv=StratifiedKFold(

                                 n_splits=folds,

                                 shuffle=False,

                                 random_state=1001).split(X,y),

              scoring='roc_auc',

              n_jobs=1,

              verbose=2)
starttime = timer(None)

start_time = timer(None)

rfecv.fit(X, y)

timer(start_time)
print('\n Optimal number of features: %d' % rfecv.n_features_)

sel_features = [f for f, s in zip(all_features, rfecv.support_) if s]

print('\n The selected features are {}:'.format(sel_features))
plt.figure(figsize=(12, 9))

plt.xlabel('Number of features tested x 2')

plt.ylabel('Cross-validation score (AUC)')

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.savefig('Porto-RFECV-01.png', dpi=150)

plt.show()
ranking = pd.DataFrame({'Features': all_features})

ranking['Rank'] = np.asarray(rfecv.ranking_)

ranking.sort_values('Rank', inplace=True)

ranking.to_csv('Porto-RFECV-ranking-01.csv', index=False)
score = round((np.max(rfecv.grid_scores_) * 2 - 1), 5)

test['target'] = rfecv.predict_proba(X_test)[:,1]

test = test[['id', 'target']]

now = datetime.now()

sub_file = 'submission_5fold-RFECV-RandomForest-01_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

print("\n Writing submission file: %s" % sub_file)

test.to_csv(sub_file, index=False)

timer(starttime)