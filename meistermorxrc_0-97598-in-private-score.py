import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.mixture import GaussianMixture

from sklearn.datasets import make_classification

from tqdm import tqdm_notebook

import pickle

import warnings

import multiprocessing

warnings.filterwarnings('ignore')
add_sample = False

sample_size = 128

max_iter = 100

loop1 = 5

loop2 = 20
def calibration(pred,true_rate):

    curr_rate = pred.mean()

    change_weight = np.log(curr_rate/(1-curr_rate)) - np.log(true_rate/(1-true_rate))

    score_adj = np.exp(np.log(pred/(1-pred)) - change_weight)/(np.exp(np.log(pred/(1-pred)) - change_weight)+1)

    return score_adj 
def load_data(data):

    return pd.read_csv(data)



with multiprocessing.Pool() as pool:

    trn, tst, sub = pool.map(load_data, ['../input/instant-gratification/train.csv', '../input/instant-gratification/test.csv','../input/instant-gratification/sample_submission.csv'])
with open('../input/instant/comp_result140.pkl','rb') as f:

    out_info = pickle.load(f)
cols = np.array([c for c in trn.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic','pred']])

oof = np.zeros(len(trn))

preds = np.zeros(len(tst))



oof_loop = np.zeros(len(trn))

preds_loop = np.zeros(len(tst))



pre_init = {}

for i in tqdm_notebook(range(512)):

    pre_init[i] = [None,None]

    trn2 = trn[trn['wheezy-copper-turtle-magic']==i]

    tst2 = tst[tst['wheezy-copper-turtle-magic']==i]

    idx1 = trn2.index; idx2 = tst2.index

    trn2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(trn2[cols]), pd.DataFrame(tst2[cols])])

    usecols = [var for var in cols if data[var].std()>2]

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])

    

    trn3 = data2[:trn2.shape[0]]; tst3 = data2[trn2.shape[0]:]

    skf = StratifiedKFold(n_splits=11, random_state=42)

    oof_tmp = np.zeros(trn2.shape[0])

    for trn_index, tst_index in skf.split(trn2, trn2['target']):

        clf = QuadraticDiscriminantAnalysis(0.6)

        clf.fit(trn3[trn_index,:],trn2.loc[trn_index]['target'])

        oof[idx1[tst_index]] = clf.predict_proba(trn3[tst_index,:])[:,1]

        oof_tmp[tst_index] = clf.predict_proba(trn3[tst_index,:])[:,1]

        preds[idx2] += clf.predict_proba(tst3)[:,1] / skf.n_splits

    auc_bench = roc_auc_score(trn2['target'], oof_tmp)

    auc_basic = auc_bench

    

    model_cnt = 0

    for loop in range(loop1):

        components = out_info[i][1].shape[0]

        trn2['tgt'] = np.nan

        gmm = GaussianMixture(n_components=components,precisions_init=out_info[i][1])

        trn2.loc[trn2['target']==1,'tgt'] = gmm.fit_predict(trn2[trn2['target']==1][usecols].values)+components

        

        precisions_pos = gmm.precisions_

        gmm = GaussianMixture(n_components=components,precisions_init=out_info[i][2])

        trn2.loc[trn2['target']==0,'tgt'] = gmm.fit_predict(trn2[trn2['target']==0][usecols].values)

        

        precisions_neg = gmm.precisions_

        trn3 = trn2[usecols].values



        skf = StratifiedKFold(n_splits=11, random_state=42)

        oof_tmp = np.zeros(trn2.shape[0])

        preds_tmp = np.zeros(tst3.shape[0])

        for trn_index, tst_index in skf.split(trn2, trn2['tgt']):

            clf = QuadraticDiscriminantAnalysis(0.6)

            try:

                clf.fit(trn3[trn_index,:],trn2.loc[trn_index]['tgt'])

            except:

                break

            oof_tmp[tst_index] = clf.predict_proba(trn3[tst_index,:])[:,components:].sum(axis=1)

            preds_tmp += clf.predict_proba(tst3)[:,components:].sum(axis=1) / skf.n_splits

        auc_tmp = roc_auc_score(trn2['target'],oof_tmp)

        

        if auc_basic < auc_tmp:

            if auc_bench < auc_tmp:

                auc_bench = auc_tmp

            preds_loop[idx2] += preds_tmp

            oof_loop[idx1] += oof_tmp

            pre_init[i] = [precisions_pos,precisions_neg]

            model_cnt += 1

        if model_cnt>0:

            preds[idx2] = preds_loop[idx2]/model_cnt

            oof[idx1] = oof_loop[idx1]/model_cnt

    '''preds[idx2] = preds[idx2]/model_cnt

    oof[idx1] = oof[idx1]/model_cnt'''

    print(i,auc_bench-auc_basic)

auc = roc_auc_score(trn['target'], oof)

print(f'AUC: {auc:.5}')
trn['pred'] = oof

tst['pred'] = preds

tst['target'] = np.nan

tst.loc[tst['pred']>=0.99,'target']= 1

tst.loc[tst['pred']<=0.01,'target']= 0

trn_new = trn.copy()

trn_new['target_old'] = trn_new['target']

trn_new.loc[trn_new['pred']>=0.995,'target'] = 1

trn_new.loc[trn_new['pred']<=0.005,'target'] = 0

trn_new = pd.concat([trn_new,tst[tst['target'].notnull()]]).reset_index(drop=True)
cols = np.array([c for c in trn.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic','target_old','pred']])

oof = np.zeros(len(trn_new))

preds = np.zeros(len(tst))

oof_qda = np.zeros(len(trn_new))

preds_qda = np.zeros(len(tst))

oof_nusvc = np.zeros(len(trn_new))

preds_nusvc = np.zeros(len(tst))
from sklearn.svm import NuSVC
for i in tqdm_notebook(range(512)):

    trn2 = trn_new[trn_new['wheezy-copper-turtle-magic']==i]

    tst2 = tst[tst['wheezy-copper-turtle-magic']==i]

    idx1 = trn2.index; idx2 = tst2.index

    trn2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(trn2[cols]), pd.DataFrame(tst2[cols])])

    usecols = [var for var in cols if data[var].std()>2]

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])

    #data2 = StandardScaler().fit_transform(data2)





    trn3 = data2[:trn2.shape[0]]; tst3 = data2[trn2.shape[0]:]

    skf = StratifiedKFold(n_splits=11, random_state=42)

    oof_tmp_qda = np.zeros(trn2.shape[0])

    oof_tmp_nusvc = np.zeros(trn2.shape[0])

    for trn_index, tst_index in skf.split(trn2, trn2['target']):

        clf_qda = QuadraticDiscriminantAnalysis(0.6)

        clf_qda.fit(trn3[trn_index,:],trn2.loc[trn_index]['target'])

        oof_qda[idx1[tst_index]] = clf_qda.predict_proba(trn3[tst_index,:])[:,1]

        oof_tmp_qda[tst_index] = clf_qda.predict_proba(trn3[tst_index,:])[:,1]

        preds_qda[idx2] += clf_qda.predict_proba(tst3)[:,1] / skf.n_splits

        

        clf_nusvc = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.5, coef0=0.053)

        clf_nusvc.fit(trn3[trn_index,:],trn2.loc[trn_index]['target'])

        oof_nusvc[idx1[tst_index]] = clf_nusvc.predict_proba(trn3[tst_index,:])[:,1]

        oof_tmp_nusvc[tst_index] = clf_nusvc.predict_proba(trn3[tst_index,:])[:,1]

        preds_nusvc[idx2] += clf_nusvc.predict_proba(tst3)[:,1] / skf.n_splits

    trn2['pred_qda'] = oof_tmp_qda

    trn2['pred_nusvc'] = oof_tmp_nusvc

    

    auc_bench_qda = roc_auc_score(trn2.loc[trn2['target_old'].notnull()]['target_old'], trn2.loc[trn2['target_old'].notnull()]['pred_qda'])

    auc_basic_qda = auc_bench_qda

    

    auc_bench_nusvc = roc_auc_score(trn2.loc[trn2['target_old'].notnull()]['target_old'], trn2.loc[trn2['target_old'].notnull()]['pred_nusvc'])

    auc_basic_nusvc = auc_bench_nusvc



    best_com = 0

    for loop in range(6):

        components = out_info[i][1].shape[0]

        trn2['tgt'] = np.nan

        gmm1 = GaussianMixture(n_components=components,max_iter = max_iter)#,precisions_init=pre_init[i][0])

        trn2.loc[trn2['target']==1,'tgt'] = gmm1.fit_predict(trn2[trn2['target']==1][usecols].values)+components

        gmm2 = GaussianMixture(n_components=components,max_iter = max_iter)#,precisions_init=pre_init[i][1])

        trn2.loc[trn2['target']==0,'tgt'] = gmm2.fit_predict(trn2[trn2['target']==0][usecols].values)

        trn3 = trn2[usecols].values          

        if add_sample:

            tmp = gmm1.sample(n_samples = sample_size) 

            trn2_pos = pd.DataFrame(tmp[0],columns=usecols)

            trn2_pos['tgt'] = tmp[1]+components

            trn2_pos['target'] = np.nan

            tmp = gmm2.sample(n_samples = sample_size) 

            trn2_neg = pd.DataFrame(tmp[0],columns=usecols)

            trn2_neg['tgt'] = tmp[1]

            trn2_neg['target'] = np.nan

            trn2_add = pd.concat([trn2,trn2_pos,trn2_neg]).sample(frac=1).reset_index(drop=True)

        else:

            trn2_add = trn2.copy()

        trn3 = trn2_add[usecols].values

        skf = StratifiedKFold(n_splits=11, random_state=42)

        oof_tmp_qda = np.zeros(trn2_add.shape[0])

        preds_tmp_qda = np.zeros(tst3.shape[0])

        oof_tmp_nusvc = np.zeros(trn2_add.shape[0])

        preds_tmp_nusvc = np.zeros(tst3.shape[0])

        for trn_index, tst_index in skf.split(trn2_add, trn2_add['tgt']):

            clf_qda = QuadraticDiscriminantAnalysis(0.6)

            clf_nusvc = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.5, coef0=0.053)

            try:

                clf_qda.fit(trn3[trn_index,:],trn2_add.loc[trn_index]['tgt'])

                clf_nusvc.fit(trn3[trn_index,:],trn2_add.loc[trn_index]['tgt'])

            except:

                break

            oof_tmp_qda[tst_index] = clf_qda.predict_proba(trn3[tst_index,:])[:,components:].sum(axis=1)

            preds_tmp_qda += clf_qda.predict_proba(tst3)[:,components:].sum(axis=1) / skf.n_splits

            oof_tmp_nusvc[tst_index] = clf_nusvc.predict_proba(trn3[tst_index,:])[:,components:].sum(axis=1)

            preds_tmp_nusvc += clf_nusvc.predict_proba(tst3)[:,components:].sum(axis=1) / skf.n_splits

        trn2_add['pred_qda'] = oof_tmp_qda

        trn2_add['pred_nusvc'] = oof_tmp_nusvc

        auc_tmp_qda = roc_auc_score(trn2_add[trn2_add['target_old'].notnull()]['target_old'],trn2_add[trn2_add['target_old'].notnull()]['pred_qda'])

        auc_tmp_nusvc = roc_auc_score(trn2_add[trn2_add['target_old'].notnull()]['target_old'],trn2_add[trn2_add['target_old'].notnull()]['pred_nusvc'])

        if auc_bench_qda < auc_tmp_qda:

            del trn2['pred_qda']

            auc_bench_qda = auc_tmp_qda

            trn2 = trn2.merge(trn2_add[['id','pred_qda']],'left','id')

            preds_qda[idx2] = preds_tmp_qda

            oof_qda[idx1] = trn2['pred_qda'].values

            #best_com = components

        if auc_bench_nusvc < auc_tmp_nusvc:

            del trn2['pred_nusvc']

            auc_bench_nusvc = auc_tmp_nusvc

            trn2 = trn2.merge(trn2_add[['id','pred_nusvc']],'left','id')

            preds_nusvc[idx2] = preds_tmp_nusvc

            oof_nusvc[idx1] = trn2['pred_nusvc'].values

            #best_com = components

    print("第{}轮QDA的CV为：{}，basic的CV为：{}".format(i,auc_bench_qda,auc_basic_qda))

    print("第{}轮NuSVC的CV为：{}，basic的CV为：{}".format(i,auc_bench_nusvc,auc_basic_nusvc))

trn_new['pred_qda'] = oof_qda

trn_new['pred_nusvc'] = oof_nusvc
auc_qda = roc_auc_score(trn_new.loc[trn_new['target_old'].notnull()]['target_old'], trn_new.loc[trn_new['target_old'].notnull()]['pred_qda'])

auc_nusvc = roc_auc_score(trn_new.loc[trn_new['target_old'].notnull()]['target_old'], trn_new.loc[trn_new['target_old'].notnull()]['pred_nusvc'])
print(auc_qda)

print(auc_nusvc)
def find_the_threshold2(n,m,oof_qda,oof_nusvc,oof2,a,b):

    for i in range(oof_qda.shape[0]):

        if oof_qda[i]>n:

            oof2[i] = oof_qda[i]

        elif  oof_qda[i]<m:

            oof2[i] = oof_qda[i]

        else:

            oof2[i] = oof_nusvc[i]*a+oof_qda[i]*b

    return oof2
oof2 = np.zeros(len(trn))

print(roc_auc_score(trn_new.loc[trn_new['target_old'].notnull()]['target_old'],find_the_threshold2(0.99997,0.00007,

                                                                                                   trn_new.loc[trn_new['target_old'].notnull()]['pred_qda'],

                                                                                                   trn_new.loc[trn_new['target_old'].notnull()]['pred_nusvc'],

                                                                                                   oof2,

                                                                                                   0.5,0.5)))
for i in range(preds_qda.shape[0]):

    if preds_qda[i]>0.99997:

        preds[i] = preds_qda[i]

    elif preds_qda[i]<0.00007:

        preds[i] = preds_qda[i]

    else:

        preds[i] = preds_nusvc[i]*0.5+preds_qda[i]*0.5
sub['target'] = preds

sub.to_csv('submission.csv',index=False)