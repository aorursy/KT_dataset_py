import numpy as np, pandas as pd, os

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook as tqdm

np.random.seed(42)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# the following fast shuffling is from https://stackoverflow.com/questions/50554272/randomly-shuffle-items-in-each-row-of-numpy-array dicussed in this rich kernel: https://www.kaggle.com/jiweiliu/fast-inplace-shuffle-for-augmentation

def disarrange(a, axis=-1):

    """

    Shuffle `a` in-place along the given axis.



    Apply numpy.random.shuffle to the given axis of `a`.

    Each one-dimensional slice is shuffled independently.

    """

    b = a.swapaxes(axis, -1)

    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,

    # so `a` is shuffled in place, too.

    shp = b.shape[:-1]

    for ndx in np.ndindex(shp):

        np.random.shuffle(b[ndx])

    return



def augment_fast2(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        disarrange(x1,axis=0)

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        disarrange(x1,axis=0)

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y

    
# INITIALIZE VARIABLES

cols = [c for c in train.columns if c not in ['id', 'target']]

cols.remove('wheezy-copper-turtle-magic')

interactions = np.zeros((512,255))

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in tqdm(range(512)):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    test2.reset_index(drop=True,inplace=True)

    

    skf = StratifiedKFold(n_splits=25, random_state=42)

    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):

        # LOGISTIC REGRESSION MODEL

        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)

        X_t, y_t = augment_fast2(train2.loc[train_index][cols].values, train2.loc[train_index]['target'].values)

        clf.fit(X_t, y_t)

        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]

        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / 25.0

        # RECORD INTERACTIONS

        for j in range(255):

            if clf.coef_[0][j]>0: interactions[i,j] = 1

            elif clf.coef_[0][j]<0: interactions[i,j] = -1

    #if i%25==0: print(i)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('LR with interactions scores CV =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)