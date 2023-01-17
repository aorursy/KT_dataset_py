import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

%matplotlib inline
def get_inputs(data, metadata):
    metadata.drop(['ra','decl','gal_l','gal_b','mwebv','hostgal_photoz','ddf','distmod'],inplace=True,axis=1)
    data['flux_ratio_sq'] = np.power(data['flux'] / data['flux_err'], 2.0)
    data['flux_by_flux_ratio_sq'] = data['flux'] * data['flux_ratio_sq']
    aggdata = data.groupby(['object_id','passband']).agg({'mjd': ['min', 'max', 'size'],
                                             'flux': ['min', 'max', 'mean', 'median', 'std','skew'],
                                             'flux_err': ['min', 'max', 'mean', 'median', 'std','skew'],
                                             'flux_by_flux_ratio_sq': ['sum'],    
                                             'flux_ratio_sq': ['sum'],                      
                                             'detected': ['mean','std']}).reset_index(drop=False)
    
    cols = ['_'.join(str(s).strip() for s in col if s) if len(col)==2 else col for col in aggdata.columns ]
    aggdata.columns = cols
    aggdata = aggdata.merge(metadata,on='object_id',how='left')
    aggdata.insert(1,'delta_passband', aggdata.mjd_max-aggdata.mjd_min)
    aggdata.drop(['mjd_min','mjd_max'],inplace=True,axis=1)
    aggdata['flux_diff'] = aggdata['flux_max'] - aggdata['flux_min']
    aggdata['flux_dif2'] = (aggdata['flux_max'] - aggdata['flux_min']) / aggdata['flux_mean']
    aggdata['flux_w_mean'] = aggdata['flux_by_flux_ratio_sq_sum'] / aggdata['flux_ratio_sq_sum']
    aggdata['flux_dif3'] = (aggdata['flux_max'] - aggdata['flux_min']) / aggdata['flux_w_mean']
    return aggdata
meta_train = pd.read_csv('../input/training_set_metadata.csv')
train = pd.read_csv('../input/training_set.csv')
traindata = get_inputs(train,meta_train)
features = list(set(traindata.columns).difference(set(['target','object_id','hostgal_photoz_err','hostgal_specz'])))
allfeatures = ['object_id']+features
alldata = traindata.loc[:,allfeatures].copy()
for c in features:
    print(c)
    if(alldata[c].min()<0):
        alldata.loc[~alldata[c].isnull(),c] = np.sign(alldata.loc[~alldata[c].isnull(),c])*np.log1p(np.abs(alldata.loc[~alldata[c].isnull(),c]))
    elif((alldata[c].max()-alldata[c].min())>10):
        alldata.loc[~alldata[c].isnull(),c] = np.log1p(alldata.loc[~alldata[c].isnull(),c])
alldata.fillna(alldata.mean(),inplace=True)
ss = None
ss = StandardScaler()
alldata.loc[:,features] = ss.fit_transform(alldata.loc[:,features])
print(ss.mean_)
print(ss.scale_)
model = TSNE(n_components=2, perplexity=30,random_state=0)
tsnedata = model.fit_transform(alldata[features])
plt.plot(tsnedata[:,0])
plt.plot(tsnedata[:,1])