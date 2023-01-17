import numpy as np 

import pandas as pd 



%matplotlib inline

import matplotlib.pyplot as plt

from tqdm import tqdm 

from xgboost import XGBRegressor as xgbr

from sklearn.pipeline import make_pipeline as mpp

from sklearn.preprocessing import Imputer as imp

from sklearn.ensemble import AdaBoostRegressor as adr

from sklearn.model_selection import cross_val_score as cvs

from sklearn.linear_model import Lasso as lss

from sklearn.linear_model import Ridge as rdg

from sklearn.linear_model import ElasticNet as ecn

from sklearn.metrics import accuracy_score as aus

from sklearn.linear_model import LinearRegression as lmr 

from sklearn.metrics import mean_absolute_error as mae

from sklearn.preprocessing import StandardScaler as scl

from sklearn.model_selection import ShuffleSplit as sst



import time 

import math
%%time

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train.head()
train.shape
tsx = train['acoustic_data'].values[::100]

tsy = train['time_to_failure'].values[::100]



fig, ax = plt.subplots(figsize = (16, 8))

plt.title('Tester')

plt.plot(tsx, color = 'r')

ax.set_ylabel('Data')

plt.legend(['acoustic_data'])

axinv = ax.twinx()



plt.plot(tsy, color = 'b')

axinv.set_ylabel('time_to_failure')

plt.legend(['time_to_failure'], loc = (0.875, 0.9))

plt.grid(False)
rows = 150_000

sgt = int(np.floor(train.shape[0] / rows))

rjp = 75000

njp = int(5)



def data_feature(ar, abs_values = False):

    

    idx = np.array(range(len(ar)))

    

    if abs_values:

        ar = np.abs(ar)

    lr = lmr()

    lr.fit(idx.reshape(-1, 1), ar)

    return lr.coef_[0]



def unknow_func(x, lsa, lla):

    

    sta = np.cumsum(x ** 2)

    sta = np.require(sta, dtype = np.float)

    lta = sta.copy()

    

    sta[lsa:] = sta[lsa:] - sta[:-lsa]

    sta /= lsa

    lta[lla:] = lta[lla:] - lta[:-lla]

    lta /= lla

         

    sta[:lla - 1] = 0

    

    dty = np.finfo(0.0).tiny

    idx = lta < dty

    lta[idx] = dty

    

    return sta / lta
xtr = pd.DataFrame(index = range(sgt), dtype = np.float64)

ytr = pd.DataFrame(index = range(sgt), dtype = np.float64, columns = ['time_to_failure'])

     

mn = train['acoustic_data'].mean()

sd = train['acoustic_data'].std()

mx = train['acoustic_data'].max()

mi = train['acoustic_data'].min()

tt = np.abs(train['acoustic_data']).sum() 



for sgt in tqdm(range(sgt)) :

    sg = train.iloc[sgt * rows + njp * rjp : sgt * rows + rows + njp * rjp]

    

    X = pd.Series(sg['acoustic_data'].values)

    y = train['time_to_failure'].values[-1]

    

    xtr.loc[sgt + njp * sgt, 'part1'] = X.mean()

    xtr.loc[sgt + njp * sgt, 'part2'] = X.std()

    xtr.loc[sgt + njp * sgt, 'part3'] = X.max()

    xtr.loc[sgt + njp * sgt, 'part4'] = X.min()

    xtr.loc[sgt + njp * sgt, 'part5'] = X.kurtosis()

    xtr.loc[sgt + njp * sgt, 'part6'] = X.skew()

    xtr.loc[sgt + njp * sgt, 'part7.0'] = X.quantile()

    xtr.loc[sgt + njp * sgt, 'part7.1'] = np.count_nonzero(X < np.quantile(X,0.05))

    xtr.loc[sgt + njp * sgt, 'part7.2'] = np.count_nonzero(X < np.quantile(X,0.010))

    xtr.loc[sgt + njp * sgt, 'part7.3'] = np.count_nonzero(X > np.quantile(X,0.015))

    xtr.loc[sgt + njp * sgt, 'part7.4'] = np.count_nonzero(X > np.quantile(X,0.020))

    xtr.loc[sgt + njp * sgt, 'part7.5'] = np.count_nonzero(X < np.quantile(X,0.025))

    xtr.loc[sgt + njp * sgt, 'part7.6'] = np.count_nonzero(X < np.quantile(X,0.030))

    xtr.loc[sgt + njp * sgt, 'part7.7 '] = np.count_nonzero(X > np.quantile(X,0.035))

    xtr.loc[sgt + njp * sgt, 'part7.8'] = np.count_nonzero(X > np.quantile(X,0.040))

    xtr.loc[sgt + njp * sgt, 'part7.9'] = np.count_nonzero(X < np.quantile(X,0.045))

    xtr.loc[sgt + njp * sgt, 'part7.10'] = np.count_nonzero(X < np.quantile(X,0.050))

    xtr.loc[sgt + njp * sgt, 'part8'] = data_feature(X)

    xtr.loc[sgt + njp * sgt, 'part9'] = data_feature(X, abs_values = True)

    xtr.loc[sgt + njp * sgt, 'part10.0'] = unknow_func(X, 500, 10000).mean()

    xtr.loc[sgt + njp * sgt, 'part10.1'] = unknow_func(X, 625, 25000).mean()

    xtr.loc[sgt + njp * sgt, 'part11'] = np.abs(X).max()

    xtr.loc[sgt + njp * sgt, 'part12'] = np.abs(X).min()

    xtr.loc[sgt + njp * sgt, 'part13'] = X.mean() - X.std()

    xtr.loc[sgt + njp * sgt, 'part14'] = X.max() - X.min()

    xtr.loc[sgt + njp * sgt, 'part15'] = np.abs(X).max() - np.abs(X).min()



    for win in [25, 225, 3375] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.1' + str(win)] = xrllsd.mean()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.2' + str(win)] = xrllsd.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.3' + str(win)] = xrllsd.max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.4' + str(win)] = xrllsd.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.11' + str(win)] = np.abs(xrllsd).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition1.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.1' + str(win)] = xrllmn.mean()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.2' + str(win)] = xrllmn.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.3' + str(win)] = xrllmn.max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.4' + str(win)] = xrllmn.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.11' + str(win)] = np.abs(xrllmn).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.17' + str(win)] =  np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition2.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

        

    for win in [50, 375, 6225] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.1' + str(win)] = xrllsd.mean()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.2' + str(win)] = xrllsd.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.3' + str(win)] = xrllsd.max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.4' + str(win)] = xrllsd.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.11' + str(win)] = np.abs(xrllsd).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition3.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.1' + str(win)] = xrllmn.mean()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.2' + str(win)] = xrllmn.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.3' + str(win)] = xrllmn.max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.4' + str(win)] = xrllmn.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.11' + str(win)] = np.abs(xrllmn).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.17' + str(win)] = np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtr.loc[sgt + njp * sgt, 'WindowsPartition4.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

            

    ytr.loc[sgt + njp * sgt, 'time_to_failure'] = y

    pass
sc = scl()

sc.fit(xtr)



sxtr = pd.DataFrame(sc.transform(xtr), columns = xtr.columns)



ytr.head()
lm = lmr()

lm.fit(sxtr, ytr)



lmpr = lm.predict(sxtr)

s = mae(ytr, lmpr)

print(s)



xgbm = xgbr()

xgbm.fit(sxtr, ytr)



xgbpr = xgbm.predict(sxtr)

xgbs = mae(ytr, xgbpr)

print(xgbs)



laso = lss(alpha = 0.1)

laso.fit(sxtr, ytr)





lspr = laso.predict(sxtr)

lsss = mae(ytr, lspr)

print(lsss)



rdgm = rdg()

rdgm.fit(sxtr, ytr)



rdgr = rdgm.predict(sxtr)

rdgs = mae(ytr, rdgr)

print(rdgs)



ecnm = ecn()

ecnm.fit(sxtr, ytr)



ecnpr = ecnm.predict(sxtr)

ecnms = mae(ytr, ecnpr)

print(ecnms)



adrm = adr()

adrm.fit(sxtr, ytr)



adrpr = adrm.predict(sxtr)

adrms = mae(ytr, adrpr)

print(adrms)
def sigmoid(z):

    

    return 1 / (1 + np.exp(-z))



a = np.array(xtr)

b = sigmoid(a)
def cost(theta, e, d, lr) :

    theta = np.matrix(theta) 

    e = np.matrix(e)

    d = np.matrix(d)

        

    fst = np.multiply(-e, np.log(sigmoid(d * theta.T)))

    scd = np.multiply((1 - e), np.log(1 - sigmoid(d * theta.T)))

    reg = (lr / 2 * len(e)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    

    return np.sum(fst - scd) / (len(e)) + reg 



e = train['acoustic_data'].values[-1]

f = train['time_to_failure'].values[-1]



ef = cost(0.13741, e, f, 0.25)
sub = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')



xtst = pd.DataFrame(columns = xtr.columns, dtype = np.float64, index = sub.index)



for i, seg_id in enumerate(tqdm(xtst.index)) :

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    X = seg['acoustic_data']

    

    xtst.loc[seg_id, 'part1'] = X.mean()

    xtst.loc[seg_id, 'part2'] = X.std()

    xtst.loc[seg_id, 'part3'] = X.max()

    xtst.loc[seg_id, 'part4'] = X.min()

    xtst.loc[seg_id, 'part5'] = X.kurtosis()

    xtst.loc[seg_id, 'part6'] = X.skew()

    xtst.loc[seg_id, 'part7.0'] = X.quantile()

    xtst.loc[seg_id, 'part7.1'] = np.count_nonzero(X < np.quantile(X,0.05))

    xtst.loc[seg_id, 'part7.2'] = np.count_nonzero(X < np.quantile(X,0.010))

    xtst.loc[seg_id, 'part7.3'] = np.count_nonzero(X > np.quantile(X,0.015))

    xtst.loc[seg_id, 'part7.4'] = np.count_nonzero(X > np.quantile(X,0.020))

    xtst.loc[seg_id, 'part7.5'] = np.count_nonzero(X < np.quantile(X,0.025))

    xtst.loc[seg_id, 'part7.6'] = np.count_nonzero(X < np.quantile(X,0.030))

    xtst.loc[seg_id, 'part7.7 '] = np.count_nonzero(X > np.quantile(X,0.035))

    xtst.loc[seg_id, 'part7.8'] = np.count_nonzero(X > np.quantile(X,0.040))

    xtst.loc[seg_id, 'part7.9'] = np.count_nonzero(X < np.quantile(X,0.045))

    xtst.loc[seg_id, 'part7.10'] = np.count_nonzero(X < np.quantile(X,0.050))

    xtst.loc[seg_id, 'part8'] = data_feature(X)

    xtst.loc[seg_id, 'part9'] = data_feature(X, abs_values = True)

    xtst.loc[seg_id, 'part10.0'] = unknow_func(X, 500, 10000).mean()

    xtst.loc[seg_id, 'part10.1'] = unknow_func(X, 625, 25000).mean()

    xtst.loc[seg_id, 'part11'] = np.abs(X).max()

    xtst.loc[seg_id, 'part12'] = np.abs(X).min()

    xtst.loc[seg_id, 'part13'] = X.mean() - X.std()

    xtst.loc[seg_id, 'part14'] = X.max() - X.min()

    xtst.loc[seg_id, 'part15'] = np.abs(X).max() - np.abs(X).min()

    

    for win in [25, 225, 3375] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtst.loc[seg_id, 'WindowsPartition1.1' + str(win)] = xrllsd.mean()

        xtst.loc[seg_id, 'WindowsPartition1.2' + str(win)] = xrllsd.std()

        xtst.loc[seg_id, 'WindowsPartition1.3' + str(win)] = xrllsd.max()

        xtst.loc[seg_id, 'WindowsPartition1.4' + str(win)] = xrllsd.min()

        xtst.loc[seg_id, 'WindowsPartition1.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition1.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition1.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtst.loc[seg_id, 'WindowsPartition1.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtst.loc[seg_id, 'WindowsPartition1.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtst.loc[seg_id, 'WindowsPartition1.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtst.loc[seg_id, 'WindowsPartition1.11' + str(win)] = np.abs(xrllsd).max()

        xtst.loc[seg_id, 'WindowsPartition1.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtst.loc[seg_id, 'WindowsPartition1.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtst.loc[seg_id, 'WindowsPartition1.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition1.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition1.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtst.loc[seg_id, 'WindowsPartition1.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtst.loc[seg_id, 'WindowsPartition1.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtst.loc[seg_id, 'WindowsPartition1.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtst.loc[seg_id, 'WindowsPartition1.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        

        

        xtst.loc[seg_id, 'WindowsPartition2.1' + str(win)] = xrllmn.mean()

        xtst.loc[seg_id, 'WindowsPartition2.2' + str(win)] = xrllmn.std()

        xtst.loc[seg_id, 'WindowsPartition2.3' + str(win)] = xrllmn.max()

        xtst.loc[seg_id, 'WindowsPartition2.4' + str(win)] = xrllmn.min()

        xtst.loc[seg_id, 'WindowsPartition2.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition2.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition2.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtst.loc[seg_id, 'WindowsPartition2.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtst.loc[seg_id, 'WindowsPartition2.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtst.loc[seg_id, 'WindowsPartition2.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtst.loc[seg_id, 'WindowsPartition2.11' + str(win)] = np.abs(xrllmn).max()        

        xtst.loc[seg_id, 'WindowsPartition2.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtst.loc[seg_id, 'WindowsPartition2.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtst.loc[seg_id, 'WindowsPartition2.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition2.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition2.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtst.loc[seg_id, 'WindowsPartition2.17' + str(win)] = np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtst.loc[seg_id, 'WindowsPartition2.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtst.loc[seg_id, 'WindowsPartition2.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtst.loc[seg_id, 'WindowsPartition2.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

        

    for win in [50, 375, 6225] :

        xrllsd = X.rolling(win).std().dropna().values 

        xrllmn = X.rolling(win).std().dropna().values

        

        xtst.loc[seg_id, 'WindowsPartition3.1' + str(win)] = xrllsd.mean()

        xtst.loc[seg_id, 'WindowsPartition3.2' + str(win)] = xrllsd.std()

        xtst.loc[seg_id, 'WindowsPartition3.3' + str(win)] = xrllsd.max()

        xtst.loc[seg_id, 'WindowsPartition3.4' + str(win)] = xrllsd.min()

        xtst.loc[seg_id, 'WindowsPartition3.5' + str(win)] = np.quantile(xrllsd, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition3.6' + str(win)] = np.quantile(xrllsd, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition3.7' + str(win)] = np.quantile(xrllsd, 0.125)

        xtst.loc[seg_id, 'WindowsPartition3.8' + str(win)] = np.quantile(xrllsd, 0.105)

        xtst.loc[seg_id, 'WindowsPartition3.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtst.loc[seg_id, 'WindowsPartition3.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0])

        xtst.loc[seg_id, 'WindowsPartition3.11' + str(win)] = np.abs(xrllsd).max()

        xtst.loc[seg_id, 'WindowsPartition3.12' + str(win)] = xrllsd.mean() - xrllsd.std()

        xtst.loc[seg_id, 'WindowsPartition3.13' + str(win)] = xrllsd.max()  - xrllsd.min()

        xtst.loc[seg_id, 'WindowsPartition3.14' + str(win)] = np.quantile(xrllsd, 0.0250) - np.quantile(xrllsd, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition3.15' + str(win)] = np.quantile(xrllsd, 0.0900) - np.quantile(xrllsd, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition3.16' + str(win)] = np.quantile(xrllsd, 0.250) - np.quantile(xrllsd, 0.125)

        xtst.loc[seg_id, 'WindowsPartition3.17' + str(win)] = np.quantile(xrllsd, 0.210) - np.quantile(xrllsd, 0.105)

        xtst.loc[seg_id, 'WindowsPartition3.18' + str(win)] = np.quantile(xrllsd, 0.2575) 

        xtst.loc[seg_id, 'WindowsPartition3.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) - np.abs(xrllsd).max()

        xtst.loc[seg_id, 'WindowsPartition3.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllsd) / xrllsd[:-1]))[0]) + np.abs(xrllsd).max()

        

        xtst.loc[seg_id, 'WindowsPartition4.1' + str(win)] = xrllmn.mean()

        xtst.loc[seg_id, 'WindowsPartition4.2' + str(win)] = xrllmn.std()

        xtst.loc[seg_id, 'WindowsPartition4.3' + str(win)] = xrllmn.max()

        xtst.loc[seg_id, 'WindowsPartition4.4' + str(win)] = xrllmn.min()

        xtst.loc[seg_id, 'WindowsPartition4.5' + str(win)] = np.quantile(xrllmn, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition4.6' + str(win)] = np.quantile(xrllmn, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition4.7' + str(win)] = np.quantile(xrllmn, 0.125)

        xtst.loc[seg_id, 'WindowsPartition4.8' + str(win)] = np.quantile(xrllmn, 0.105)

        xtst.loc[seg_id, 'WindowsPartition4.9' + str(win)] = np.mean(np.diff(xrllsd))

        xtst.loc[seg_id, 'WindowsPartition4.10' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0])

        xtst.loc[seg_id, 'WindowsPartition4.11' + str(win)] = np.abs(xrllmn).max()

        xtst.loc[seg_id, 'WindowsPartition4.12' + str(win)] = xrllmn.mean() - xrllmn.std()

        xtst.loc[seg_id, 'WindowsPartition4.13' + str(win)] = xrllmn.max()  - xrllmn.min()

        xtst.loc[seg_id, 'WindowsPartition4.14' + str(win)] = np.quantile(xrllmn, 0.0250) - np.quantile(xrllmn, 0.0125)

        xtst.loc[seg_id, 'WindowsPartition4.15' + str(win)] = np.quantile(xrllmn, 0.0900) - np.quantile(xrllmn, 0.0750)

        xtst.loc[seg_id, 'WindowsPartition4.16' + str(win)] = np.quantile(xrllmn, 0.250) - np.quantile(xrllmn, 0.125)

        xtst.loc[seg_id, 'WindowsPartition4.17' + str(win)] =  np.quantile(xrllmn, 0.210) - np.quantile(xrllmn, 0.105)

        xtst.loc[seg_id, 'WindowsPartition4.18' + str(win)] = np.quantile(xrllmn, 0.2575)

        xtst.loc[seg_id, 'WindowsPartition4.19' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) - np.abs(xrllmn).max()

        xtst.loc[seg_id, 'WindowsPartition4.20' + str(win)] = np.mean(np.nonzero((np.diff(xrllmn) / xrllmn[:-1]))[0]) + np.abs(xrllmn).max()

            

    pass
sc.fit(xtst)

xtsc = pd.DataFrame(sc.transform(xtst), columns = xtst.columns)



mp1 = mpp(imp(), xgbr())

mp2 = mpp(imp(), rdg())

mp3 = mpp(imp(), adr())

mp4 = mpp(imp(), lss())

mp1.fit(sxtr, ytr)

mp2.fit(sxtr, ytr)

mp3.fit(sxtr, ytr)

mp4.fit(sxtr, ytr)



mppr1 = mp1.predict(xtsc)

mppr2 = mp2.predict(xtsc)

mppr3 = mp3.predict(xtsc)

mppr4 = mp4.predict(xtsc)



ab = (lm.predict(xtsc) + ecnm.predict(xtsc))  / 4

cd = (mppr1 + mppr2 + mppr3 + mppr4) / 16 



zz = ((ab + cd) / 2) + ef
sub['time_to_failure'] = zz 



print(sub)

sub.to_csv('sub.csv', index = True)