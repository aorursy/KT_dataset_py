import gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8'}
train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['click_id','ip', 'app', 'device', 'os', 'channel', 'click_time']
train = pd.read_csv('../input/train.csv',skiprows=range(1,131886954),dtype=dtypes, usecols=train_cols) # Just use day 
test = pd.read_csv('../input/test.csv',dtype=dtypes, usecols=test_cols)
train['click_time'] = pd.to_datetime(train.click_time)
train['hour'] = train.click_time.dt.hour.astype('uint8')
test['click_time'] = pd.to_datetime(test.click_time)
test['hour'] = test.click_time.dt.hour.astype('uint8')

train['dummy'] = (train.app.astype(str)+'_' +train.channel.astype(str)+'_'+train.hour.astype(str)).apply(hash) % 2**26
test['dummy'] = (test.app.astype(str)+'_'+ test.channel.astype(str)+'_'+test.hour.astype(str)).apply(hash) % 2**26
train['app_channel'], test['app_channel'] = target_encode(train['dummy'], 
                                                          test['dummy'], 
                                                          target=train.is_attributed, 
                                                          min_samples_leaf=100,
                                                          smoothing=0,
                                                          noise_level=0.0)
test.drop('dummy',inplace=True,axis=1)
gc.collect()
train.drop('dummy',inplace=True,axis=1)
gc.collect()
train['dummy'] = (train.ip.astype(str)+'_'+train.hour.astype(str)).apply(hash) % 2**26
test['dummy'] = (test.ip.astype(str)+'_'+test.hour.astype(str)).apply(hash) % 2**26
train['ip'], test['ip'] = target_encode(train['dummy'], 
                                       test['dummy'], 
                                       target=train.is_attributed, 
                                       min_samples_leaf=100,
                                       smoothing=0,
                                       noise_level=0.0)
test.drop('dummy',inplace=True,axis=1)
gc.collect()
train.drop('dummy',inplace=True,axis=1)
gc.collect()
train['dummy'] = (train.app.astype(str)+'_'+train.hour.astype(str)).apply(hash) % 2**26
test['dummy'] = (test.app.astype(str)+'_'+test.hour.astype(str)).apply(hash) % 2**26
train['app'], test['app'] = target_encode(train['dummy'], 
                                       test['dummy'], 
                                       target=train.is_attributed, 
                                       min_samples_leaf=100,
                                       smoothing=0,
                                       noise_level=0.0)
test.drop('dummy',inplace=True,axis=1)
gc.collect()
train.drop('dummy',inplace=True,axis=1)
gc.collect()
train['dummy'] = (train.device.astype(str)+'_'+train.hour.astype(str)).apply(hash) % 2**26
test['dummy'] = (test.device.astype(str)+'_'+test.hour.astype(str)).apply(hash) % 2**26
train['device'], test['device'] = target_encode(train['dummy'], 
                                               test['dummy'], 
                                               target=train.is_attributed, 
                                               min_samples_leaf=100,
                                               smoothing=0,
                                               noise_level=0.0)
test.drop('dummy',inplace=True,axis=1)
gc.collect()
train.drop('dummy',inplace=True,axis=1)
gc.collect()
train['dummy'] = (train.os.astype(str)+'_'+train.hour.astype(str)).apply(hash) % 2**26
test['dummy'] = (test.os.astype(str)+'_'+test.hour.astype(str)).apply(hash) % 2**26
train['os'], test['os'] = target_encode(train['dummy'], 
                                       test['dummy'], 
                                       target=train.is_attributed, 
                                       min_samples_leaf=100,
                                       smoothing=0,
                                       noise_level=0.0)
test.drop('dummy',inplace=True,axis=1)
gc.collect()
train.drop('dummy',inplace=True,axis=1)
gc.collect()
train['dummy'] = (train.channel.astype(str)+'_'+train.hour.astype(str)).apply(hash) % 2**26
test['dummy'] = (test.channel.astype(str)+'_'+test.hour.astype(str)).apply(hash) % 2**26
train['channel'], test['channel'] = target_encode(train['dummy'], 
                                                   test['dummy'], 
                                                   target=train.is_attributed, 
                                                   min_samples_leaf=100,
                                                   smoothing=0,
                                                   noise_level=0.0)
test.drop('dummy',inplace=True,axis=1)
gc.collect()
train.drop('dummy',inplace=True,axis=1)
gc.collect()
train['dummy'] = (train.hour.astype(str)).apply(hash) % 2**26
test['dummy'] = (test.hour.astype(str)).apply(hash) % 2**26
train['hour'], test['hour'] = target_encode(train['dummy'], 
                                           test['dummy'], 
                                           target=train.is_attributed, 
                                           min_samples_leaf=100,
                                           smoothing=0,
                                           noise_level=0.0)
test.drop('dummy',inplace=True,axis=1)
gc.collect()
train.drop('dummy',inplace=True,axis=1)
gc.collect()
def Output(p):
    return 1./(1.+np.exp(-p))

def GP(data):
    return Output(np.tanh((((-1.0) + (((np.where(data["os"]>0, (((((data["app_channel"]) > (np.tanh((data["app_channel"]))))*1.)) * 2.0), -1.0 )) * 2.0)))/2.0)) +
                  np.tanh(((np.where(data["app"]>0, np.where(data["os"]>0, (((((data["channel"]) * 2.0)) > (data["hour"]))*1.), -2.0 ), -2.0 )) * 2.0)) +
                  np.tanh((((((((((((data["channel"]) + (((((data["app"]) * 2.0)) * 2.0)))/2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)))
roc_auc_score(train.is_attributed,GP(train))
del train
gc.collect()
sub = pd.DataFrame()
sub['click_id'] = test.click_id.values
sub['is_attributed'] = GP(test).values
sub.to_csv('xxx.csv.gz',compression='gzip',index=False)