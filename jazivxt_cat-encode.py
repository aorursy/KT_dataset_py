import numpy as np

import pandas as pd

from sklearn import *



train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')

train.shape, test.shape
def features(df):

    df['ord_5l'] = df['ord_5'].str.lower()

    df['ord_5.0'] = df['ord_5'].map(lambda x: x[0])

    df['ord_5l.0'] = df['ord_5l'].map(lambda x: x[0])

    df['ord_5.1'] = df['ord_5'].map(lambda x: x[1])

    df['ord_5l.1'] = df['ord_5l'].map(lambda x: x[1])

    return df

    

train = features(train)

test = features(test)
col = [c for c in train.columns if c not in ['id','target', 'nom_4']]

for c in col:

    if train[c].nunique() > 2:

        enc1 = {e2:e1 for e1, e2 in enumerate(train[c].value_counts().index)} #counts

        enc2 = {e2:e1 for e1, e2 in enumerate(sorted(train[c].unique()))} #alpha

        enc3 = {e1:e2 for e1, e2 in train.groupby([c])['target'].agg('sum').rank(ascending=1).reset_index().values} #target_sum_rank

        enc4 = {e1:e2 for e1, e2 in train.groupby([c], as_index=False)['target'].agg('mean').values} #target_mean

        train[c+'enc1'] = train[c].map(enc1)

        train[c+'enc2'] = train[c].map(enc2)

        train[c+'enc3'] = train[c].map(enc3)

        train[c+'enc4'] = train[c].map(enc4)

        train.drop(columns=[c], inplace=True)

        test[c+'enc1'] = test[c].map(enc1)

        test[c+'enc2'] = test[c].map(enc2)

        test[c+'enc3'] = test[c].map(enc3)

        test[c+'enc4'] = test[c].map(enc4)

        test.drop(columns=[c], inplace=True)

    else:

        enc1 = {e2:e1 for e1, e2 in enumerate(sorted(train[c].unique()))} #alpha

        train[c] = train[c].map(enc1)

        test[c] = test[c].map(enc1) 



train.shape, test.shape
col = [c for c in train.columns if c not in ['id', 'target', 'nom_4']]



sub = []

for v in train.nom_4.unique():

    train2 = train[train['nom_4']==v].reset_index(drop=True)

    test2 = test[test['nom_4']==v].reset_index(drop=True)

    x1, x2, y1, y2 = model_selection.train_test_split(train2[col], train2['target'], test_size=0.2, random_state=5)

    #model = svm.NuSVC(nu=0.4, kernel='rbf', degree=3, gamma='scale', coef0=0.0, probability=True, tol=0.001, random_state=3)

    model = ensemble.RandomForestClassifier(n_jobs=-1, random_state=3, n_estimators=200)

    model.fit(x1, y1)

    print(v, metrics.roc_auc_score(y2, model.predict_proba(x2)[:,1]))

    fi = pd.DataFrame({'col':col, 'fi': model.feature_importances_})

    _ = fi.sort_values(by=['fi'], ascending=[False])[:20].plot(kind='barh', x='col', y='fi', figsize=(10,10))

    #col2 = fi.sort_values(by=['fi'], ascending=[False])['col'][:40]

    model.fit(train2[col], train2['target'])

    test2['target'] = model.predict_proba(test2[col].fillna(-1))[:,1]

    sub.append(test2)

sub = pd.concat(sub)

sub[['id', 'target']].to_csv('submission.csv', index=False)