import numpy as np # linear algebra

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tqdm

import gc

import sys

import warnings

warnings.filterwarnings("ignore")
date = pd.read_csv('../input/bosch-production-line-performance/train_date.csv.zip', nrows=10000)

numeric = pd.read_csv('../input/bosch-production-line-performance/train_numeric.csv.zip', nrows=10000)

category = pd.read_csv('../input/bosch-production-line-performance/train_categorical.csv.zip', nrows=10000)
date
numeric
category
num_feats = ['Id',

       'L3_S30_F3514', 'L0_S9_F200', 'L3_S29_F3430', 'L0_S11_F314',

       'L0_S0_F18', 'L3_S35_F3896', 'L0_S12_F350', 'L3_S36_F3918',

       'L0_S0_F20', 'L3_S30_F3684', 'L1_S24_F1632', 'L0_S2_F48',

       'L3_S29_F3345', 'L0_S18_F449', 'L0_S21_F497', 'L3_S29_F3433',

       'L3_S30_F3764', 'L0_S1_F24', 'L3_S30_F3554', 'L0_S11_F322',

       'L3_S30_F3564', 'L3_S29_F3327', 'L0_S2_F36', 'L0_S9_F180',

       'L3_S33_F3855', 'L0_S0_F4', 'L0_S21_F477', 'L0_S5_F114',

       'L0_S6_F122', 'L1_S24_F1122', 'L0_S9_F165', 'L0_S18_F439',

       'L1_S24_F1490', 'L0_S6_F132', 'L3_S29_F3379', 'L3_S29_F3336',

       'L0_S3_F80', 'L3_S30_F3749', 'L1_S24_F1763', 'L0_S10_F219',

 'Response']
length = date.drop('Id', axis=1).count()

date_cols = length.reset_index().sort_values(by=0, ascending=False)

stations = sorted(date_cols['index'].str.split('_',expand=True)[1].unique().tolist())

date_cols['station'] = date_cols['index'].str.split('_',expand=True)[1]

date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()
data = None

for chunk in pd.read_csv('../input/bosch-production-line-performance/train_date.csv.zip',usecols=['Id'] + date_cols,chunksize=50000,low_memory=False):



    chunk.columns = ['Id'] + stations

    chunk['start_station'] = -1

    chunk['end_station'] = -1

    

    for s in stations:

        chunk[s] = 1 * (chunk[s] >= 0)

        id_not_null = chunk[chunk[s] == 1].Id

        chunk.loc[(chunk['start_station']== -1) & (chunk.Id.isin(id_not_null)),'start_station'] = int(s[1:])

        chunk.loc[chunk.Id.isin(id_not_null),'end_station'] = int(s[1:])   

    data = pd.concat([data, chunk])
for chunk in pd.read_csv('../input/bosch-production-line-performance/test_date.csv.zip',usecols=['Id'] + date_cols,chunksize=50000,low_memory=False):

    

    chunk.columns = ['Id'] + stations

    chunk['start_station'] = -1

    chunk['end_station'] = -1

    for s in stations:

        chunk[s] = 1 * (chunk[s] >= 0)

        id_not_null = chunk[chunk[s] == 1].Id

        chunk.loc[(chunk['start_station']== -1) & (chunk.Id.isin(id_not_null)),'start_station'] = int(s[1:])

        chunk.loc[chunk.Id.isin(id_not_null),'end_station'] = int(s[1:])   

    data = pd.concat([data, chunk])

del chunk

gc.collect()   
data = data[['Id','start_station','end_station']]

usefuldatefeatures = ['Id']+date_cols
minmaxfeatures = None

for chunk in pd.read_csv('../input/bosch-production-line-performance/train_date.csv.zip',usecols=usefuldatefeatures,chunksize=50000,low_memory=False):

    features = chunk.columns.values.tolist()

    features.remove('Id')

    df_mindate_chunk = chunk[['Id']].copy()

    df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values

    df_mindate_chunk['maxdate'] = chunk[features].max(axis=1).values

    df_mindate_chunk['min_time_station'] =  chunk[features].idxmin(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)

    df_mindate_chunk['max_time_station'] =  chunk[features].idxmax(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)

    minmaxfeatures = pd.concat([minmaxfeatures, df_mindate_chunk])



del chunk

gc.collect()
for chunk in pd.read_csv('../input/bosch-production-line-performance/test_date.csv.zip',usecols=usefuldatefeatures,chunksize=50000,low_memory=False):

    features = chunk.columns.values.tolist()

    features.remove('Id')

    df_mindate_chunk = chunk[['Id']].copy()

    df_mindate_chunk['mindate'] = chunk[features].min(axis=1).values

    df_mindate_chunk['maxdate'] = chunk[features].max(axis=1).values

    df_mindate_chunk['min_time_station'] =  chunk[features].idxmin(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)

    df_mindate_chunk['max_time_station'] =  chunk[features].idxmax(axis = 1).apply(lambda s: int(s.split('_')[1][1:]) if s is not np.nan else -1)

    minmaxfeatures = pd.concat([minmaxfeatures, df_mindate_chunk])



del chunk

gc.collect()
minmaxfeatures.sort_values(by=['mindate', 'Id'], inplace=True)

minmaxfeatures['min_Id_rev'] = -minmaxfeatures.Id.diff().shift(-1)

minmaxfeatures['min_Id'] = minmaxfeatures.Id.diff()
cols = [['Id']+date_cols,num_feats]
traindata = None

testdata = None
trainfiles = ['train_date.csv.zip','train_numeric.csv.zip']

testfiles = ['test_date.csv.zip','test_numeric.csv.zip']
for i,f in enumerate(trainfiles):

    

    subset = None

    

    for chunk in pd.read_csv('../input/bosch-production-line-performance/' + f,usecols=cols[i],chunksize=100000,low_memory=False):

        subset = pd.concat([subset, chunk])

    

    if traindata is None:

        traindata = subset.copy()

    else:

        traindata = pd.merge(traindata, subset.copy(), on="Id")

        

del subset,chunk

gc.collect()

del cols[1][-1]
for i, f in enumerate(testfiles):

    subset = None

    

    for chunk in pd.read_csv('../input/bosch-production-line-performance/' + f,usecols=cols[i],chunksize=100000,low_memory=False):

        subset = pd.concat([subset, chunk])

        

    if testdata is None:

        testdata = subset.copy()

    else:

        testdata = pd.merge(testdata, subset.copy(), on="Id")

    

del subset,chunk

gc.collect()
traindata = traindata.merge(minmaxfeatures, on='Id')

traindata = traindata.merge(data, on='Id')

testdata = testdata.merge(minmaxfeatures, on='Id')

testdata = testdata.merge(data, on='Id')
del minmaxfeatures,data

gc.collect()
train = traindata[::2]

valid = traindata[1::2]
del traindata

gc.collect()
def mcc(tp, tn, fp, fn):

    num = tp * tn - fp * fn

    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if den == 0:

        return 0

    else:

        return num / np.sqrt(den)
def eval_mcc(y_true, y_prob):

    idx = np.argsort(y_prob)

    y_true_sort = y_true[idx]

    n = y_true.shape[0]

    nump = 1.0 * np.sum(y_true) 

    numn = n - nump 

    tp,fp = nump,numn

    tn,fn = 0.0,0.0

    best_mcc = 0.0

    best_id = -1

    mccs = np.zeros(n)

    for i in range(n):

        if y_true_sort[i] == 1:

            tp -= 1.0

            fn += 1.0

        else:

            fp -= 1.0

            tn += 1.0

        new_mcc = mcc(tp, tn, fp, fn)

        mccs[i] = new_mcc

        if new_mcc >= best_mcc:

            best_mcc = new_mcc

            best_id = i

    return best_mcc
def mcc_eval(y_prob, dtrain):

    y_true = dtrain.get_label()

    best_mcc = eval_mcc(y_true, y_prob)

    return 'MCC', best_mcc
import xgboost as xgb

params = {'objective':"binary:logistic",

          'max_depth':25,

          'base_score':0.005,

          'eval_metric':'auc',

          'n_jobs':-1

}
trainm = xgb.DMatrix(train.drop(['Response','Id'],axis=1),train['Response'])

validm = xgb.DMatrix(valid.drop(['Response','Id'],axis=1),valid['Response'])



test = xgb.DMatrix(testdata.drop(['Id'],axis=1))
#del train,valid,testdata

#gc.collect()
watchlist = [(trainm, 'train'), (validm, 'val')]

clf = xgb.train(params, trainm,

                num_boost_round=100,

                evals=watchlist,

                early_stopping_rounds=20,

                feval=mcc_eval,

                maximize=True

                )
predictions = clf.predict(validm)
from sklearn.metrics import matthews_corrcoef

thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([matthews_corrcoef(valid.Response, predictions>threshold) for threshold in thresholds])



plt.plot(thresholds, mcc)

best_prob = thresholds[mcc.argmax()]

best_prob
fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(clf,ax=ax)
test = clf.predict(test)
testdata['Response'] = (test>best_prob).astype(int)

testdata[['Id','Response']].to_csv("submitwoId.csv",index=False)
!gzip submitwoId.csv