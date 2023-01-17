import pandas as pd
import time
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack
import gc
import lightgbm as lgb
from sklearn.model_selection import train_test_split

path = '../input/'


def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time']).dt.date
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)    
    return df
start_time = time.time()

train = pd.read_csv(path+"train.csv", nrows=20000000)
train.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
train = dataPreProcessTime(train)

y = train['is_attributed'].astype(float)  
nrow_train = train.shape[0] 

train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)

test = pd.read_csv(path+"test.csv")
test = dataPreProcessTime(test)

sub = pd.DataFrame()
sub['click_id'] = test['click_id']
test.drop('click_id', axis=1, inplace=True)

merge: pd.DataFrame = pd.concat([train, test])
    
lb = LabelBinarizer(sparse_output=True)
merge_ip = lb.fit_transform(merge['ip'])
merge_app = lb.fit_transform(merge['app'])
merge_device = lb.fit_transform(merge['device'])
merge_os = lb.fit_transform(merge['os'])
merge_channel = lb.fit_transform(merge['channel'])

sparse_merge = hstack((merge_ip, merge_app, merge_device, merge_os, merge_channel)).tocsr()
sparse_merge = sparse_merge.astype(float)

X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_train:]

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3, random_state=100)

print('[{}] Data completed.'.format(time.time() - start_time))
gc.collect()

start_time = time.time()
print("LGB startting")

params = {
        'learning_rate': 0.6,
        'objective': 'binary',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'auc', 
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.65,
        'nthread': 4,
        'min_data_in_leaf': 100,
        'max_bin': 31
    }


d_train = lgb.Dataset(train_X, label=train_y)
d_val = lgb.Dataset(valid_X, label=valid_y)
watchlist = [d_train, d_val]
model = lgb.train(params, train_set=d_train, num_boost_round=7000, valid_sets=watchlist, verbose_eval=1000)

print('[{}] Finish LGB Training'.format(time.time() - start_time))

sub['is_attributed'] = model.predict(X_test)
sub.to_csv('lgb_sub.csv',index=False)
