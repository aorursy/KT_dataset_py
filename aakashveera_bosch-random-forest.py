import numpy as np

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
traindata
testdata
traindata.fillna(value=0,inplace=True)

testdata.fillna(value=0,inplace=True)
np.set_printoptions(suppress=True)
import gc
total = traindata[traindata['Response']==0].sample(frac=1).head(400000)

total = pd.concat([total,traindata[traindata['Response']==1]]).sample(frac=1)
del traindata

gc.collect()
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=500,n_jobs=-1,verbose=1,random_state=11)

model.fit(total.drop(['Response','Id'],axis=1),total['Response'])
test = model.predict(testdata.drop(['Id'],axis=1))
testdata['Response'] = test

testdata[['Id','Response']].to_csv("submit.csv",index=False)
!gzip submit.csv
total