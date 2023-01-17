import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.svm import NuSVC

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from tqdm import tqdm_notebook

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import VarianceThreshold



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
oof_svnu = np.zeros(len(train)) 

pred_te_svnu = np.zeros(len(test))



oof_qda = np.zeros(len(train)) 

pred_te_qda = np.zeros(len(test))
for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    #### Preprocessing for both train and test data

    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(PCA(svd_solver='full',n_components='mle').fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=1.5).fit_transform(data[cols]))

    train4 = data2[:train2.shape[0]]; test4 = data2[train2.shape[0]:]

    

    skf = StratifiedKFold(n_splits=25, random_state=42)

    

    for train_index, test_index in skf.split(train2, train2['target']):

        qda_clf = QuadraticDiscriminantAnalysis(reg_param=0.111)

        qda_clf.fit(train4[train_index,:],train2.loc[train_index]['target'])

        oof_qda[idx1[test_index]] = qda_clf.predict_proba(train4[test_index,:])[:,1]

        pred_te_qda[idx2] += qda_clf.predict_proba(test4)[:,1] / skf.n_splits

        

        nusvc_clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)

        nusvc_clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_svnu[idx1[test_index]] = nusvc_clf.predict_proba(train3[test_index,:])[:,1]

        pred_te_svnu[idx2] += nusvc_clf.predict_proba(test3)[:,1] / skf.n_splits
print('nusvc', roc_auc_score(train['target'], oof_svnu))

print('qda', roc_auc_score(train['target'], oof_qda))

print('blend 1', roc_auc_score(train['target'], oof_qda*0.8+oof_svnu*0.2))

print('blend 2', roc_auc_score(train['target'], oof_qda*0.7+oof_svnu*0.3))

print('blend 3', roc_auc_score(train['target'], oof_qda*0.65+oof_svnu*0.35))

print('blend 4', roc_auc_score(train['target'], oof_qda*0.6+oof_svnu*0.4))

print('blend 5', roc_auc_score(train['target'], oof_qda*0.5+oof_svnu*0.5))

print('blend 6', roc_auc_score(train['target'], oof_qda*0.4+oof_svnu*0.6))

print('blend 7', roc_auc_score(train['target'], oof_qda*0.3+oof_svnu*0.7))
sub_nusvc = pd.read_csv('../input/sample_submission.csv')

sub_qda = pd.read_csv('../input/sample_submission.csv')

sub_blend1 = pd.read_csv('../input/sample_submission.csv')

sub_blend2 = pd.read_csv('../input/sample_submission.csv')

sub_blend3 = pd.read_csv('../input/sample_submission.csv')

sub_blend4 = pd.read_csv('../input/sample_submission.csv')

sub_blend5 = pd.read_csv('../input/sample_submission.csv')

sub_blend6 = pd.read_csv('../input/sample_submission.csv')

sub_blend7 = pd.read_csv('../input/sample_submission.csv')
sub_qda['target'] = pred_te_qda

sub_qda.to_csv('QDA_Baseline2.csv', index=False)



sub_nusvc['target'] = pred_te_svnu

sub_nusvc.to_csv('NUsvc_Baseline.csv', index=False)



sub_blend1['target'] = pred_te_svnu*0.2 + pred_te_qda*0.8

sub_blend1.to_csv('blend_qda0.8_svnu0.2.csv', index=False)



sub_blend2['target'] = pred_te_svnu*0.3 + pred_te_qda*0.7

sub_blend2.to_csv('blend_qda0.7_svnu0.3.csv', index=False)



sub_blend3['target'] = pred_te_svnu*0.35 + pred_te_qda*0.65

sub_blend3.to_csv('blend_qda0.65_svnu0.35.csv', index=False)



sub_blend4['target'] = pred_te_svnu*0.4 + pred_te_qda*0.6

sub_blend4.to_csv('blend_qda0.6_svnu0.4.csv', index=False)



sub_blend5['target'] = pred_te_svnu*0.5 + pred_te_qda*0.5

sub_blend5.to_csv('blend_qda0.5_svnu0.5.csv', index=False)



sub_blend6['target'] = pred_te_svnu*0.6 + pred_te_qda*0.4

sub_blend6.to_csv('blend_qda0.4_svnu0.6.csv', index=False)



sub_blend7['target'] = pred_te_svnu*0.7 + pred_te_qda*0.3

sub_blend7.to_csv('blend_qda0.3_svnu0.7.csv', index=False)