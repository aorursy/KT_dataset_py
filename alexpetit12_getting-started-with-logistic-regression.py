import warnings



import pandas as pd

import numpy as np



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold



from timeit import default_timer as timer



warnings.filterwarnings('ignore')
%%time

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
target = train_df['target']
features = [c for c in train_df if c not in ['ID_code', 'target']]
kfolds = StratifiedKFold(n_splits=10, random_state=1).split(train_df.values, target.values)



oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))



for k, (trn_idx, val_idx) in enumerate(kfolds):

    print('Fold: {}'.format(k))

    start = timer()

    

    clf = LogisticRegression(solver='lbfgs', max_iter=1500, C=10)

    clf.fit(train_df.iloc[trn_idx][features], target[trn_idx])

    oof[val_idx] = clf.predict_proba(train_df.iloc[val_idx][features])[:, 1]

    predictions += clf.predict_proba(test_df[features])[:, 1]

    

    print(timer() - start)

    

predictions /= kfolds.n_splits

    

print('CV score: {}'.format(roc_auc_score(target.values, oof)))
submission = pd.DataFrame({'ID_code': test_df['ID_code'].values})

submission['target'] = predictions

submission.to_csv('submission.csv', index=False)