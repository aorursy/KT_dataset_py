import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import logging
from astropy.stats import LombScargle
from astropy.time import Time
train = pd.read_csv('../input/training_set.csv')
print(train.shape)
meta_train = pd.read_csv('../input/training_set_metadata.csv')
print(meta_train.shape)
train['mjd'] = Time(train.mjd.values, format='mjd').iso
train['mjd'] = pd.to_datetime(train['mjd'])
x = train.copy()
x.head()
x = x.sort_values(by=['object_id','passband','mjd'])
x['cc'] = x.groupby(['object_id','passband'])['mjd'].cumcount()

x.mjd = x.groupby(['object_id','passband'])['mjd'].diff()
x.loc[~x.mjd.isnull(),'mjd'] = x.loc[~x.mjd.isnull(),'mjd'].apply(lambda a: a.total_seconds())
x.loc[x.mjd.isnull(),'mjd'] = 0
x.mjd = x.mjd.astype('float32')
x.head()
x = x.set_index(['object_id','passband','cc'])
x = x.unstack()
x.head()
cols = ['_'.join(str(s).strip() for s in col if s) if len(col)==2 else col for col in x.columns ]
cols
x.columns = cols
x.head()
mjdcolumns = [a  for a in x.columns if a.startswith('mjd')]
fluxcolumns = [a  for a in x.columns if a.startswith('flux') and not a.startswith('flux_err')]
def GenerateLS(r):
    t = r[mjdcolumns].dropna().apply(lambda a: a/(3600)).cumsum().astype('float32')
    p = r[fluxcolumns].dropna().astype('float32')
    frequency, power = LombScargle(t.values,p.values).autopower(nyquist_factor=1)
    w = pd.DataFrame()
    w['f'] = (frequency*100000).astype(int)
    w['p'] = power
    w = w.groupby('f').sum().reset_index(drop=False)
    return  {'f'+str(int(i)):j for i,j in zip(w.f,w.p)}
    
d = x.apply(lambda r: GenerateLS(r),axis=1)
trn_all_predictions = pd.DataFrame(list(d))
trn_all_predictions.head()
meta_train = meta_train.set_index('object_id')
x = x.join(meta_train,on='object_id',how='left')
x = x.reset_index(drop=False)

x.head()
trn_all_predictions.insert(0,'passband',x.passband.ravel())
trn_all_predictions.insert(0,'object_id',x.object_id.ravel())
trn_all_predictions = trn_all_predictions.join(meta_train,on='object_id',how='left')
trn_all_predictions.head()
cols = []
for i in range(186):
    cols.append('f'+str(i))

f, ax = plt.subplots(6)
f.set_figheight(15)
f.set_figwidth(15)
for i in range(6):
    ax[i].xaxis.set_ticks(np.arange(0, 186, 20))   
ax[0].plot(trn_all_predictions[(trn_all_predictions.passband==0)][cols].mean())
ax[1].plot(trn_all_predictions[(trn_all_predictions.passband==1)][cols].mean())
ax[2].plot(trn_all_predictions[(trn_all_predictions.passband==2)][cols].mean())
ax[3].plot(trn_all_predictions[(trn_all_predictions.passband==3)][cols].mean())
ax[4].plot(trn_all_predictions[(trn_all_predictions.passband==4)][cols].mean())
ax[5].plot(trn_all_predictions[(trn_all_predictions.passband==5)][cols].mean())
uniques = sorted(trn_all_predictions.target.unique())
f, ax = plt.subplots(len(uniques),6)
f.set_figheight(30)
f.set_figwidth(15)
for a in range(len(uniques)):
    for b in range(6):
        ax[a,b].xaxis.set_ticks(np.arange(0, 186, 50))   
   
for i in range(len(uniques)):
    ax[i][0].plot(trn_all_predictions[(trn_all_predictions.target==uniques[i])&(trn_all_predictions.passband==0)][cols].mean()-trn_all_predictions[(trn_all_predictions.passband==0)][cols].mean())
    ax[i][1].plot(trn_all_predictions[(trn_all_predictions.target==uniques[i])&(trn_all_predictions.passband==1)][cols].mean()-trn_all_predictions[(trn_all_predictions.passband==1)][cols].mean())
    ax[i][2].plot(trn_all_predictions[(trn_all_predictions.target==uniques[i])&(trn_all_predictions.passband==2)][cols].mean()-trn_all_predictions[(trn_all_predictions.passband==2)][cols].mean())
    ax[i][3].plot(trn_all_predictions[(trn_all_predictions.target==uniques[i])&(trn_all_predictions.passband==3)][cols].mean()-trn_all_predictions[(trn_all_predictions.passband==3)][cols].mean())
    ax[i][4].plot(trn_all_predictions[(trn_all_predictions.target==uniques[i])&(trn_all_predictions.passband==4)][cols].mean()-trn_all_predictions[(trn_all_predictions.passband==4)][cols].mean())
    ax[i][5].plot(trn_all_predictions[(trn_all_predictions.target==uniques[i])&(trn_all_predictions.passband==5)][cols].mean()-trn_all_predictions[(trn_all_predictions.passband==5)][cols].mean())
