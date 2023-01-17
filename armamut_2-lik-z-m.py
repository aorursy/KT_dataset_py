import numpy as np
import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

from collections import Counter
%matplotlib inline
# Load data
train = pd.read_csv('../input/armut_challenge_training.csv', index_col=0, parse_dates=['createdate'])
train = train.sort_values(by=['userid', 'createdate'])
test = pd.read_csv('../input/armut_challenge_test.csv', index_col=0)
test = test.sort_values(by='userid')
display(train.head(3))
display(test.head(3))
# Get long sequences of services
train_services = train.groupby('userid').apply(lambda x:list(x.serviceid.values))
train_services.head()
# Get long sequences of dates
train['createdate_day'] = ((pd.to_datetime('1985-05-06') - train.createdate) / pd.to_timedelta('1D')).values
train_days = train.groupby('userid').apply(lambda x:list(x.createdate_day.values))
train_days.head()
feat_len = train.groupby('userid').size().to_frame('feat_len').reset_index()
feat_len.head()
# Build sequences from train set. This shouldn't be confused with given test set.
# We'll use it as an oof set.
SEQ_SIZE = 5
train_sequences = []
train_sequence_ids = []
train_sequence_dates = []
for uid, s in train_services.items():
    train_sequences.extend([s[i:i+SEQ_SIZE] for i in range(len(s)-SEQ_SIZE+1)])
    train_sequence_ids.extend( [uid] * (len(s)-SEQ_SIZE+1) )
    td = train_days[uid]
    train_sequence_dates.extend([td[i:i+SEQ_SIZE-1] for i in range(len(td)-SEQ_SIZE+1)]) #td[3:-1])

train_sequences = pd.DataFrame(np.array(train_sequences),
                                columns=['last_4', 'last_3', 'last_2', 'last_1', 'serviceid'])
train_sequences['userid'] = train_sequence_ids
train_sequences = train_sequences[train_sequences.columns[::-1]]

train_sequence_dates = pd.DataFrame(np.array(train_sequence_dates),
                                     columns=['date_4', 'date_3', 'date_2', 'date_1'])
train_sequence_dates = train_sequence_dates[train_sequence_dates.columns[::-1]]

train_sequences = pd.concat([train_sequences, train_sequence_dates], axis=1, sort=False)

train_sequences = train_sequences.merge(feat_len, on='userid', how='inner')

train_sequences['date_43'] = train_sequences['date_4'] - train_sequences['date_3']
train_sequences['date_32'] = train_sequences['date_3'] - train_sequences['date_2']
train_sequences['date_21'] = train_sequences['date_2'] - train_sequences['date_1']

print(train_sequences.shape)
train_sequences.head()
# Build sequences from train set. We'll use it on both oof test and prediction.
SEQ_SIZE = 5
test_sequences = []
test_sequence_dates = []
for uid, s in train_services.items():
    test_sequences.append(s[-SEQ_SIZE+1:]) # 4 adet
    td = train_days[uid]
    test_sequence_dates.append(td[-SEQ_SIZE+1:])

test_sequences = pd.DataFrame(np.array(test_sequences),
                              columns=['last_4', 'last_3', 'last_2', 'last_1'])
test_sequences['userid'] = train_services.index
test_sequences = test_sequences[test_sequences.columns[::-1]]

test_sequence_dates = pd.DataFrame(np.array(test_sequence_dates),
                                   columns=['date_4', 'date_3', 'date_2', 'date_1'])
test_sequence_dates = test_sequence_dates[test_sequence_dates.columns[::-1]]
test_sequences = pd.concat([test_sequences, test_sequence_dates], axis=1, sort=False)
test_sequences = test_sequences.merge(feat_len, on='userid', how='inner')

test_sequences['date_43'] = test_sequences['date_4'] - test_sequences['date_3']
test_sequences['date_32'] = test_sequences['date_3'] - test_sequences['date_2']
test_sequences['date_21'] = test_sequences['date_2'] - test_sequences['date_1']

test_sequences = test_sequences.merge(test, on='userid', how='left')

print(test_sequences.shape)
test_sequences.head()
# Prepare train and test set for training
tmp = train_sequences.copy()

# Create targets
#tmp['target'] = 1*(tmp.serviceid == target_service_id)
tmp[f'target_last_1'] = 1*(tmp['last_1'] == tmp.serviceid)
tmp[f'target_last_2'] = 1*(tmp['last_2'] == tmp.serviceid)
tmp[f'target_last_3'] = 1*(tmp['last_3'] == tmp.serviceid)
tmp[f'target_last_4'] = 1*(tmp['last_4'] == tmp.serviceid)
    
# Split test and train
tmp = tmp.reset_index()
#df_test = tmp.groupby('userid').last().reset_index()
#df_train = tmp[~tmp['index'].isin(df_test['index'])].copy()

from sklearn.model_selection import train_test_split
uids = tmp.userid.unique()
uid_train, uid_test = train_test_split(uids, test_size=0.2, random_state=84)
df_train = tmp[tmp.userid.isin(set(uid_train))].copy()
df_test  = tmp[tmp.userid.isin(set(uid_test))].copy()

print(f'df_train.shape: {df_train.shape}')
print(f'df_test.shape: {df_test.shape}')
col_cat = [
    'last_1', 'last_2', 'last_3', 'last_4',
    'mc_1', 'mc_2', 'mc_3', 'mc_4'
]
col_targets = [c for c in tmp.columns if 'target' in c]
col_preds = [c for c in tmp.columns if 'pred' in c]
col_not_use = ['userid', 'serviceid', 'winner', 'index'] + col_targets + col_preds
col_not_use += [
    'mc_1', 'mc_2', 'mc_3', 'mc_4',
    'date_1', 'date_2', 'date_3', 'date_4'
]
col_use = [c for c in tmp.columns if c not in col_not_use]
col_cat = [c for c in col_cat if c in col_use]

print('col_use', col_use)
print('col_targets',col_targets)
print(f'df_train.shape: {df_train.shape}')
print(f'df_test.shape: {df_test.shape}')

from lightgbm import LGBMClassifier
models = dict()
for col_target in col_targets:
    
    print(f'Model for {col_target}')
    
    model = LGBMClassifier(objective='binary', random_state=42, learning_rate=0.1,
                           n_estimators=2000,
                           reg_alpha=5,
                           reg_lambda=5,)
    model.fit(df_train[df_train[col_target].notnull()][col_use],
              df_train[df_train[col_target].notnull()][col_target],
              #categorical_feature=col_cat,
              early_stopping_rounds=100,
              eval_set=(df_test[df_test[col_target].notnull()][col_use],
                        df_test[df_test[col_target].notnull()][col_target]),
              eval_metric=['binary_logloss'],
              verbose=100)
    col_pred = col_target.replace('target', 'pred')
    models[col_pred] = model
    preds = model.predict_proba(df_test[col_use])[:, 1]
    print(pd.crosstab(preds>0.5, df_test[col_target]))
    
    print('OK')
print('Done')
tmp = df_test.copy()

# Create targets
#tmp['target'] = 1*(tmp.serviceid == target_service_id)
tmp[f'target_last_1'] = 1*(tmp['last_1'] == tmp.serviceid)
tmp[f'target_last_2'] = 1*(tmp['last_2'] == tmp.serviceid)
tmp[f'target_last_3'] = 1*(tmp['last_3'] == tmp.serviceid)
tmp[f'target_last_4'] = 1*(tmp['last_4'] == tmp.serviceid)

# Make predictions
for col_pred, model in models.items():
    tmp[col_pred] = model.predict_proba(tmp[col_use])[:, 1]

# Calibrate predictions for test set.
from itertools import product

r = np.round(np.arange(0.9, 1.1, 0.02), 2)
ks = []
for k2, k3, k4 in product(r, r, r):
    k1 = 1
    ttt = tmp.copy()
    ttt['pred_last_1'] *= k1
    ttt['pred_last_2'] *= k2
    ttt['pred_last_3'] *= k3
    ttt['pred_last_4'] *= k4
    ix = tmp.index
    kkk = ttt.loc[ix, ['pred_last_1','pred_last_2','pred_last_3','pred_last_4']].idxmax(axis=1)
    ttt.loc[ix, 'winner'] = kkk
    ix = (ttt['winner'] == 'pred_last_1')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_1']
    ix = (ttt['winner'] == 'pred_last_2')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_2']
    ix = (ttt['winner'] == 'pred_last_3')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_3']
    ix = (ttt['winner'] == 'pred_last_4')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_4']
    rr = (ttt['serviceid'] == ttt['pred']).mean()
    print([k1, k2, k3, k4, rr], ' '*10, end='\r')
    ks.append([k1, k2, k3, k4, rr])
print('')
print('Done.')

ks = pd.DataFrame(ks)
ks.sort_values(by=4, ascending=False, inplace=True)
ks.head(10)
# Get OOF set
tmp = test_sequences[test_sequences['serviceid'].notnull()].copy()

# Create targets
#tmp['target'] = 1*(tmp.serviceid == target_service_id)
tmp[f'target_last_1'] = 1*(tmp['last_1'] == tmp.serviceid)
tmp[f'target_last_2'] = 1*(tmp['last_2'] == tmp.serviceid)
tmp[f'target_last_3'] = 1*(tmp['last_3'] == tmp.serviceid)
tmp[f'target_last_4'] = 1*(tmp['last_4'] == tmp.serviceid)

# Make predictions
for col_pred, model in models.items():
    tmp[col_pred] = model.predict_proba(tmp[col_use])[:, 1]
    
tmp.head()
# Calibrate predictions for OOF set.
from itertools import product

r = np.round(np.arange(0.9, 1.1, 0.02), 2)
ks = []
for k2, k3, k4 in product(r, r, r):
    k1 = 1
    ttt = tmp.copy()
    ttt['pred_last_1'] *= k1
    ttt['pred_last_2'] *= k2
    ttt['pred_last_3'] *= k3
    ttt['pred_last_4'] *= k4
    ix = tmp.index
    kkk = ttt.loc[ix, ['pred_last_1','pred_last_2','pred_last_3','pred_last_4']].idxmax(axis=1)
    ttt.loc[ix, 'winner'] = kkk
    ix = (ttt['winner'] == 'pred_last_1')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_1']
    ix = (ttt['winner'] == 'pred_last_2')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_2']
    ix = (ttt['winner'] == 'pred_last_3')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_3']
    ix = (ttt['winner'] == 'pred_last_4')
    ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_4']
    rr = (ttt['serviceid'] == ttt['pred']).mean()
    print([k1, k2, k3, k4, rr], ' '*10, end='\r')
    ks.append([k1, k2, k3, k4, rr])
print('')
print('Done.')

ks = pd.DataFrame(ks)
ks.sort_values(by=4, ascending=False, inplace=True)
ks.head(10)
tmp = test_sequences[test_sequences['serviceid'].isnull()].copy()

# Create targets
#tmp['target'] = 1*(tmp.serviceid == target_service_id)
tmp[f'target_last_1'] = 1*(tmp['last_1'] == tmp.serviceid)
tmp[f'target_last_2'] = 1*(tmp['last_2'] == tmp.serviceid)
tmp[f'target_last_3'] = 1*(tmp['last_3'] == tmp.serviceid)
tmp[f'target_last_4'] = 1*(tmp['last_4'] == tmp.serviceid)

tmp.head()
ttt = tmp.copy()

#k1, k2, k3, k4 = 1.0, 1.06, 0.92, 1.06
#k1, k2, k3, k4 = 1.0, 0.92, 1.04, 0.98

#k1, k2, k3, k4 = 1.0, 1.04, 0.90, 0.94
k1, k2, k3, k4 = 1.0, 1.08, 0.98, 0.98

for col_pred, model in models.items():
    ttt[col_pred] = model.predict_proba(ttt[col_use])[:, 1]
ttt['pred_last_1'] *= k1
ttt['pred_last_2'] *= k2
ttt['pred_last_3'] *= k3
ttt['pred_last_4'] *= k4
ix = ttt.index
ttt['pred'] = 0
kkk = ttt.loc[ix, ['pred_last_1','pred_last_2','pred_last_3','pred_last_4']].idxmax(axis=1)
ttt.loc[ix, 'winner'] = kkk
ix = (ttt['winner'] == 'pred_last_1')
ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_1']
ix = (ttt['winner'] == 'pred_last_2')
ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_2']
ix = (ttt['winner'] == 'pred_last_3')
ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_3']
ix = (ttt['winner'] == 'pred_last_4')
ttt.loc[ix, 'pred'] = ttt.loc[ix, 'last_4']
#rr = (ttt['serviceid'] == ttt['pred']).mean()
print([k1, k2, k3, k4])
ttt['serviceid'] = ttt['pred']
ttt.head()
filename = f'submission_7_84_{k1*100:03.0f}_{k2*100:03.0f}_{k3*100:03.0f}_{k4*100:03.0f}.csv'
filename
ttt[['userid', 'serviceid']].to_csv(filename, index=False)
