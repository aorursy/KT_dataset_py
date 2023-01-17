import gc

import numpy as np

import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
def prepare(data, return_probs=False):

    LIMIT = 20

    

    cl = data[data['is_attributed'] == 1]

    vc_app_pos = cl['app'].value_counts(True)[:LIMIT].to_dict()

    vc_dev_pos = cl['device'].value_counts(True)[:LIMIT].to_dict()

    vc_os_pos = cl['os'].value_counts(True)[:LIMIT].to_dict()

    vc_chnl_pos = cl['channel'].value_counts(True)[:LIMIT].to_dict()



    cl = data[data['is_attributed'] == 0]

    vc_app_neg = cl['app'].value_counts(True)[:LIMIT].to_dict()

    vc_dev_neg = cl['device'].value_counts(True)[:LIMIT].to_dict()

    vc_os_neg = cl['os'].value_counts(True)[:LIMIT].to_dict()

    vc_chnl_neg = cl['channel'].value_counts(True)[:LIMIT].to_dict()



    del cl

    gc.collect()



    name = 'pos'

    data['is_' + name + '_app_'] = data['app'].apply(lambda x: vc_app_pos.get(x, 0))

    data['is_' + name + '_dev_'] = data['device'].apply(lambda x: vc_dev_pos.get(x, 0))

    data['is_' + name + '_os_'] = data['os'].apply(lambda x: vc_os_pos.get(x, 0))

    data['is_' + name + '_chnl_'] = data['channel'].apply(lambda x: vc_chnl_pos.get(x, 0))

    gc.collect()



    name = 'neg'

    data['is_' + name + '_app_'] = data['app'].apply(lambda x: vc_app_neg.get(x, 0))

    data['is_' + name + '_dev_'] = data['device'].apply(lambda x: vc_dev_neg.get(x, 0))

    data['is_' + name + '_os_'] = data['os'].apply(lambda x: vc_os_neg.get(x, 0))

    data['is_' + name + '_chnl_'] = data['channel'].apply(lambda x: vc_chnl_neg.get(x, 0))

    gc.collect()



    data.drop(['app', 'device', 'os', 'channel'], axis=1, inplace=True)

    gc.collect()



    X = csr_matrix(data.drop('is_attributed', axis=1).values, dtype='float64')

    y = data['is_attributed'].values.flatten()

    

    if return_probs:

        return X, y, vc_app_pos, vc_dev_pos, vc_os_pos, vc_chnl_pos, vc_app_neg, vc_dev_neg, vc_os_neg, vc_chnl_neg

    else:

        return X, y
data = pd.read_csv('../input/train.csv', 

                   usecols=[1,2,3,4,7], 

                   engine='c', 

                   encoding='ascii', 

                   na_filter=False, 

                   dtype=np.uint16,

                   nrows=40000000,

                   skiprows=1,)

data.columns = ['app', 'device', 'os', 'channel', 'is_attributed']

gc.collect()
X, y = prepare(data)

del data

gc.collect()
y_mean = y.mean()

prior = [1. - y_mean, y_mean]

m = BernoulliNB(class_prior=prior).partial_fit(X, y, classes=[0,1])

gc.collect()
y_pred = m.predict_proba(X)[:,1]

del X

gc.collect()

y_pred_round = np.round(y_pred).astype('uint8')
metrics = [

    accuracy_score(y, y_pred_round),

    roc_auc_score(y, y_pred),

    f1_score(y, y_pred_round),

    recall_score(y, y_pred_round),

    precision_score(y, y_pred_round),

]

metric_names = ['accuracy', 'roc_auc', 'f1', 'recall', 'precision']

del y, y_pred, y_pred_round

gc.collect()

pd.Series(dict(zip(metric_names, metrics)))
data = pd.read_csv('../input/train.csv', 

                   usecols=[1,2,3,4,7], 

                   engine='c', 

                   encoding='ascii', 

                   na_filter=False, 

                   dtype=np.uint16,

                   skiprows=40000001,

                   nrows=40000000,

                   header=None)

data.columns = ['app', 'device', 'os', 'channel', 'is_attributed']

gc.collect()
X, y = prepare(data)

del data

gc.collect()
y_pred = m.predict_proba(X)[:,1]

y_pred_round = np.round(y_pred).astype('uint8')
metrics = [

    accuracy_score(y, y_pred_round),

    roc_auc_score(y, y_pred),

    f1_score(y, y_pred_round),

    recall_score(y, y_pred_round),

    precision_score(y, y_pred_round),

]

del y_pred, y_pred_round

gc.collect()

metric_names = ['accuracy', 'roc_auc', 'f1', 'recall', 'precision']

pd.Series(dict(zip(metric_names, metrics)))
m.partial_fit(X, y)

del X

del y

gc.collect()
data = pd.read_csv('../input/train.csv', 

                   usecols=[1,2,3,4,7], 

                   engine='c', 

                   encoding='ascii', 

                   na_filter=False, 

                   dtype=np.uint16,

                   skiprows=80000001,

                   nrows=40000000,

                   header=None)

data.columns = ['app', 'device', 'os', 'channel', 'is_attributed']

gc.collect()



X, y = prepare(data)

del data

m.partial_fit(X, y)

del X, y

gc.collect()
data = pd.read_csv('../input/train.csv', 

                   usecols=[1,2,3,4,7], 

                   engine='c', 

                   encoding='ascii', 

                   na_filter=False, 

                   dtype=np.uint16,

                   skiprows=120000001,

                   header=None)

data.columns = ['app', 'device', 'os', 'channel', 'is_attributed']

gc.collect()



X, y, vc_app_pos, vc_dev_pos, vc_os_pos, vc_chnl_pos, vc_app_neg, vc_dev_neg, vc_os_neg, vc_chnl_neg = prepare(data, return_probs=True)

del data

m.partial_fit(X, y)

del X, y

gc.collect()
data = pd.read_csv('../input/test.csv', 

                   usecols=['click_id', 'app', 'device', 'os', 'channel'],

                   engine='c', 

                   encoding='ascii', 

                   na_filter=False, 

                   dtype=np.uint32)
ids = data['click_id'].values.flatten().tolist()

data.drop('click_id', axis=1, inplace=True)
name = 'pos'

data['is_' + name + '_app_'] = data['app'].apply(lambda x: vc_app_pos.get(x, 0))

data['is_' + name + '_dev_'] = data['device'].apply(lambda x: vc_dev_pos.get(x, 0))

data['is_' + name + '_os_'] = data['os'].apply(lambda x: vc_os_pos.get(x, 0))

data['is_' + name + '_chnl_'] = data['channel'].apply(lambda x: vc_chnl_pos.get(x, 0))

gc.collect()



name = 'neg'

data['is_' + name + '_app_'] = data['app'].apply(lambda x: vc_app_neg.get(x, 0))

data['is_' + name + '_dev_'] = data['device'].apply(lambda x: vc_dev_neg.get(x, 0))

data['is_' + name + '_os_'] = data['os'].apply(lambda x: vc_os_neg.get(x, 0))

data['is_' + name + '_chnl_'] = data['channel'].apply(lambda x: vc_chnl_neg.get(x, 0))

gc.collect()



data.drop(['app', 'device', 'os', 'channel'], axis=1, inplace=True)

gc.collect()



X = csr_matrix(data.values, dtype='float64')



del data

gc.collect()
ranked = pd.Series(m.predict_proba(X)[:,1]).rank(method='min').astype('float64')

ranked /= len(ranked)

ranked -= 0.5 # center at zero

ranked *= 12 # stretch out to the interval (-6, 6)

ranked = 1 / (1 + np.exp(-ranked)) # apply sigmoid

pd.DataFrame({

    'click_id': ids,

    'is_attributed': ranked

}).to_csv('result.csv', index=False, float_format='%.4f')