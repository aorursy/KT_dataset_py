import numpy as np, pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    train2.reset_index(drop=True,inplace=True)

    

    clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)

    clf.fit(train2[cols], train2['target'])

    

    for j in range(i, 512):

        val = train[train['wheezy-copper-turtle-magic']==j]

        preds = clf.predict_proba(val[cols])[:,1]

        auc = roc_auc_score(val['target'], preds)

        if auc > 0.65 or j==i:

            print(f'magic-{i}-{j}  auc={auc:.4}' )

    print('-'*40)