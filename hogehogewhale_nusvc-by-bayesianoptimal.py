import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_validate, StratifiedKFold

from sklearn.svm import NuSVC

from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization



import warnings

warnings.filterwarnings('ignore')



pd.options.display.max_columns = None
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train_sample = train[train['wheezy-copper-turtle-magic']==0]

X_train = train_sample.drop(['id', 'wheezy-copper-turtle-magic', 'target'], axis=1)

y_train = train_sample['target']
def f(nu, tol):

    model = NuSVC(

        nu = nu,

        tol = tol,

        random_state = 42

    )

    

    result = cross_validate(model, X_train, y_train, scoring='roc_auc')



    return np.mean(result['test_score'])
pbounds = {

        'nu':(0.01, 1),

        'tol':(1e-4, 1e-2)

}



optimizer = BayesianOptimization(f=f, pbounds=pbounds)



optimizer.maximize()
oof = np.zeros(len(train))

preds = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
for i in range(512):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx_train = train2.index

    idx_test = test2.index

    train2.reset_index(drop=True,inplace=True)



    df = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    df_pca = PCA(n_components=40, random_state=4).fit_transform(df[cols])

    df2 = StandardScaler().fit_transform(df_pca)

    split = train2.shape[0]

    train3 = df2[:split]

    test3 = df2[split:]

    

    skf = StratifiedKFold(n_splits=5, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):

        clf = NuSVC(probability=True, nu=0.1135, tol=0.008654)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx_train[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx_test] += clf.predict_proba(test3)[:,1] / skf.n_splits
print(roc_auc_score(train['target'], oof))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)