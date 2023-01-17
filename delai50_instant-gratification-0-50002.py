import numpy as np, pandas as pd, matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold



from tqdm import tqdm_notebook



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

train_orig = train.copy()

test = pd.read_csv('../input/test.csv')
# FIRST RUN OF QDA

oof_qda_vt = np.zeros(len(train))

preds_qda_vt = np.zeros(len(test))



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    

    # VT

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=1.5).fit_transform(data[cols]))

    train4 = data2[:train2.shape[0]]; test4 = data2[train2.shape[0]:]

    

    skf = StratifiedKFold(n_splits=5, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):

        

        # QDA VT

        clf_qda_vt = QuadraticDiscriminantAnalysis(reg_param=0.111)

        clf_qda_vt.fit(train4[train_index,:],train2.loc[train_index]['target'])

        oof_qda_vt[idx1[test_index]] = clf_qda_vt.predict_proba(train4[test_index,:])[:,1]

        preds_qda_vt[idx2] += clf_qda_vt.predict_proba(test4)[:,1] / skf.n_splits



print('QDA VT AUC: {}'.format(roc_auc_score(train_orig['target'], oof_qda_vt)))
# SECOND RUN INCLUDING PSEUDOLABELED DATA

test['target'] = preds_qda_vt

oof_qda_vt = np.zeros(len(train))

preds_qda_vt = np.zeros(len(test))



for i in tqdm_notebook(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    train2_pse = train2.copy(); idx1 = train2.index

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    

    # Add pseudolabeled data

    test2_pse = test2[ (test2['target']<=0.00001) | (test2['target']>=0.99999) ].copy()

    test2_pse.loc[ test2_pse['target']>=0.5, 'target' ] = 1

    test2_pse.loc[ test2_pse['target']<0.5, 'target' ] = 0 

    train2_pse = pd.concat([train2_pse, test2_pse], axis=0)

    train2_pse.reset_index(drop=True, inplace=True)

    

    # VARIANCE THRESHOLD

    sc = StandardScaler()

    vt = VarianceThreshold(threshold=1.5).fit(train2[cols])

    sc.fit( vt.transform(train2[cols]) )

    train4_pse = sc.transform( vt.transform(train2_pse[cols]) )

    train4 = sc.transform( vt.transform(train2[cols]) )

    test4 = sc.transform( vt.transform(test2[cols]) )

    

    # STRATIFIED K FOLD

    skf = StratifiedKFold(n_splits=5, random_state=42)

    for train_index, test_index in skf.split(train4_pse, train2_pse['target']):

        

        test_index2 = test_index[ test_index<len(train4) ]

        

        # QDA VT

        clf_qda_vt = QuadraticDiscriminantAnalysis(reg_param=0.111)

        clf_qda_vt.fit(train4_pse[train_index,:],train2_pse.loc[train_index]['target'])

        oof_qda_vt[idx1[test_index2]] = clf_qda_vt.predict_proba(train4[test_index2,:])[:,1]

        preds_qda_vt[test2.index] += clf_qda_vt.predict_proba(test4)[:,1] / skf.n_splits



print('QDA VT AUC: {}'.format(roc_auc_score(train_orig['target'], oof_qda_vt)))
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds_qda_vt

sub.to_csv('submission.csv', index=False)
plt.hist(sub['target'], bins=100)

plt.title('Predictions')

plt.show()