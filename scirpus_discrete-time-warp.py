import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier
meta_train = pd.read_csv('../input/training_set_metadata.csv')
train = pd.read_csv('../input/training_set.csv')
from tqdm import tqdm_notebook
total = None
mydata = []
for oid in tqdm_notebook(train.object_id.unique()[:10]):
    x = None
    for pb in range(6):
        sub = train[(train.object_id==oid)&(train.passband==pb)].copy()
        x = sub[['mjd','flux']].diff().fillna(0)
        indices = sorted(x[x.mjd>100].mjd.nlargest(2).index.values)
        
        x = x[['mjd']].cumsum().fillna(0)
        x['cl'] = -1
        if(len(indices)>0):
            x.loc[(x.index<indices[0]),'cl'] = 0
        if(len(indices)>1):
            x.loc[(x.index<indices[1])&(x.index>=indices[0]),'cl'] = 1
            x.loc[(x.index>=indices[1]),'cl'] = 2
        else:
            x.loc[(x.index>=indices[0]),'cl'] = 1
        x['object_id'] = sub.object_id
        x.mjd = (x.mjd/10).astype(int)
        x['passband'] = sub.passband
        x['detected'] = sub.detected
        x['flux_err'] = sub.flux_err
        x['flux'] = sub.flux
            
        mydata.append(x)
total = pd.concat(mydata)

 
total.head()

full_train = total.groupby(['object_id','mjd'])['flux'].mean().unstack().rename_axis(None).rename_axis(None, 1).fillna(0)
full_train.head()

def mydist(x, y, **kwargs):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance

X = full_train.values
Y = meta_train.target.values
knncustom = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree',
                                 metric=mydist, n_jobs=1)
knncustom.fit(X[:10],Y[:10])
knncustom.predict_proba(X[:10])
Y[:10]
knncustom.classes_
