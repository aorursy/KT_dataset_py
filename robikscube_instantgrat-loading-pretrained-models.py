import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt

import seaborn as sns

import pickle

import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.svm import NuSVC

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from tqdm import tqdm

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm

from datetime import datetime
train = pd.read_csv('../input/instant-gratification/train.csv')

test = pd.read_csv('../input/instant-gratification/test.csv')
m003 = '../input/instantgrat-models/M003-qda_dict-0.9637405729608689CV.pkl'

qda_dict = pickle.load(open( m003, "rb" ))
startTime = datetime.now()





BASE_DIR = '../input/instant-gratification/'

RANDOM_STATE = 529

MODEL_NUMBER = 'M003'



train = pd.read_csv('{}train.csv'.format(BASE_DIR))

test = pd.read_csv('{}test.csv'.format(BASE_DIR))



oof_qda = np.zeros(len(train))

pred_te_qda = np.zeros(len(test))



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



#qda_dict = {}



for i in tqdm(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(PCA(svd_solver='full',n_components='mle').fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=1.5).fit_transform(data[cols]))

    train4 = data2[:train2.shape[0]]; test4 = data2[train2.shape[0]:]



    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)

    # qda_dict[i] = {}

    fold = 0

    skf = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE)

    for train_index, test_index in skf.split(train2, train2['target']):

        # clf = QuadraticDiscriminantAnalysis(reg_param=0.111)

        # clf.fit(train4[train_index,:],train2.loc[train_index]['target'])

        clf = qda_dict[i][fold]

        oof_qda[idx1[test_index]] = clf.predict_proba(train4[test_index,:])[:,1]

        pred_te_qda[idx2] += clf.predict_proba(test4)[:,1] / skf.n_splits

        # qda_dict[i][fold] = clf

        fold += 1



CV_SCORE = roc_auc_score(train['target'], oof_qda)

print('qda', roc_auc_score(train['target'], oof_qda))
### SAVE RESULTS

oof_qda = oof_qda.reshape(-1, 1)

pred_te_qda = pred_te_qda.reshape(-1, 1)



np.save('{}-oof.np'.format(MODEL_NUMBER), oof_qda)

np.save('{}-pred_te_qda.np'.format(MODEL_NUMBER), pred_te_qda)



# print('Saving model file')

# f = open("{}models/{}-qda_dict-{}CV.pkl".format(BASE_DIR, MODEL_NUMBER, CV_SCORE), "wb")

# pickle.dump(qda_dict, f)

# f.close()



ss = pd.read_csv('../input/instant-gratification/sample_submission.csv'.format(BASE_DIR))

ss['target'] = pred_te_qda

ss.to_csv('{}-submission-{}CV.csv'.format(MODEL_NUMBER, CV_SCORE), index=False)



oof_df = train[['id','target']].copy()

oof_df[MODEL_NUMBER] = oof_qda

oof_df.to_csv('{}-oof-{}CV.csv'.format(MODEL_NUMBER, CV_SCORE), index=False)



seconds_to_run = datetime.now() - startTime

print('Completed in {:.4f} seconds'.format(seconds_to_run.seconds))

print('Completed in {:.4f} minutes'.format(seconds_to_run.seconds/60))

print('Completed in {:.4f} hours'.format(seconds_to_run.seconds/60/60))