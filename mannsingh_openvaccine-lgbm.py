import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import Counter
from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
train = pd.read_json("../input/stanford-covid-vaccine/train.json",lines=True)
test = pd.read_json("../input/stanford-covid-vaccine/test.json",lines=True)
ss = pd.read_csv("../input/stanford-covid-vaccine/sample_submission.csv")
train = train.set_index('index')
test = test.set_index('index')
ss
train.head(3)
test.seq_length.value_counts()
test.head(3)
print("Size of training examples: ",np.shape(train))
print("Size of test examples: ",np.shape(test))
print('========= train columns ==========')
print([c for c in train.columns])

print('========= test columns ==========')
print([c for c in test.columns])
train.info()
# read npy data file
bpps_list = os.listdir('../input/stanford-covid-vaccine/bpps/')
bpps_npy = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_list[25]}')
print('Count of npy files: ', len(bpps_list))
print('Size of image: ', bpps_npy.shape)
#Size of all .npy files are not same
NO_OF_EXAMPLES = 15
fig = plt.figure(figsize=(15, 15))
for i in range(NO_OF_EXAMPLES):
    bpps_eg = np.load(f'../input/stanford-covid-vaccine/bpps/{bpps_list[i]}')
    sub = fig.add_subplot(5,5, i + 1)
    sub.imshow(bpps_eg)
Counter(train['sequence'].values[0])
Counter(train['predicted_loop_type'].values[0])
def featurize(df):
    
    df['A_percent'] = df['sequence'].apply(lambda s: s.count('A'))/107
    df['G_percent'] = df['sequence'].apply(lambda s: s.count('G'))/107
    df['U_percent'] = df['sequence'].apply(lambda s: s.count('U'))/107
    df['C_percent'] = df['sequence'].apply(lambda s: s.count('C'))/107
    
    df['total_dot_count'] = df['structure'].apply(lambda s: s.count('.'))/107
    df['total_ob_count'] = df['structure'].apply(lambda s: s.count('('))/107
    df['total_cb_count'] = df['structure'].apply(lambda s: s.count(')'))/107
    
    df['pair_rates'] = (df['total_ob_count'] + df['total_cb_count'])/df['total_dot_count']
    
    df['S_percent'] = df['sequence'].apply(lambda s: s.count('S'))/107
    df['M_percent'] = df['sequence'].apply(lambda s: s.count('M'))/107
    df['I_percent'] = df['sequence'].apply(lambda s: s.count('I'))/107
    df['X_percent'] = df['sequence'].apply(lambda s: s.count('X'))/107
    df['B_percent'] = df['sequence'].apply(lambda s: s.count('B'))/107
    df['H_percent'] = df['sequence'].apply(lambda s: s.count('H'))/107
    
    return df
train = featurize(train)
test = featurize(test)
train['reactivity_error'] = train['reactivity_error'].apply(lambda x: np.mean(x))
train['deg_error_Mg_pH10'] = train['deg_error_Mg_pH10'].apply(lambda x: np.mean(x))
train['deg_error_Mg_50C'] = train['deg_error_Mg_50C'].apply(lambda x: np.mean(x))
required_mean = train['reactivity_error'][train['reactivity_error'] <= 1].mean()
train['reactivity_error'][train['reactivity_error'] > 1] = required_mean

required_mean = train['deg_error_Mg_pH10'][train['deg_error_Mg_pH10'] <= 1].mean()
train['deg_error_Mg_pH10'][train['deg_error_Mg_pH10'] > 1] = required_mean

required_mean = train['deg_error_Mg_50C'][train['deg_error_Mg_50C'] <= 1].mean()
train['deg_error_Mg_50C'][train['deg_error_Mg_50C'] > 1] = required_mean
train['reactivity_error'].describe()
train['mean_reactivity'] = train['reactivity'].apply(lambda x: np.mean(x))# + train['reactivity_error']
train['mean_deg_Mg_pH10'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))# + train['deg_error_Mg_pH10']
train['mean_deg_Mg_50C'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))# + train['deg_error_Mg_50C']
# Expand Sequence Features
for n in range(107):
    train[f'sequence_{n}'] = train['sequence'].apply(lambda x: x[n]).astype('category')
    test[f'sequence_{n}'] = test['sequence'].apply(lambda x: x[n]).astype('category')
# Expand Structure Features
for n in range(107):
    train[f'structure_{n}'] = train['structure'].apply(lambda x: x[n]).astype('category')
    test[f'structure_{n}'] = test['structure'].apply(lambda x: x[n]).astype('category')
# Expand predicted_loop_type Features
for n in range(107):
    train[f'predicted_loop_type_{n}'] = train['predicted_loop_type'].apply(lambda x: x[n]).astype('category')
    test[f'predicted_loop_type_{n}'] = test['predicted_loop_type'].apply(lambda x: x[n]).astype('category')
train = train[train.SN_filter == 1]
train
SEQUENCE_COLS = [c for c in train.columns if 'sequence_' in c]
STRUCTURE_COLS = [c for c in train.columns if 'structure_' in c]
PREDICTED_LOOP_COLS = [c for c in train.columns if 'predicted_loop_type_' in c]
OTHERS = ['A_percent','G_percent','C_percent','U_percent', 'pair_rates',
          'S_percent','B_percent','X_percent','H_percent','I_percent','M_percent']
MY_COLS = SEQUENCE_COLS + STRUCTURE_COLS + PREDICTED_LOOP_COLS + OTHERS
oof_error = 0
for target in ['reactivity','deg_Mg_pH10','deg_Mg_50C']:

    X = train[MY_COLS]
    y = train[f'mean_{target}']
    X_test = test[MY_COLS]
    
    N_SPLITS = 7
    target_error = 0
    
    test[f'mean_{target}_pred'] = 0
    
    for fn, (trn_idx, val_idx) in enumerate(KFold(n_splits = N_SPLITS, shuffle = True).split(X)):
        print('Fold: ', fn+1)
        X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        reg = lgb.LGBMRegressor()
        reg.fit(X_train, y_train)
        pred = reg.predict(X_val)
        loss = np.sqrt(mean_squared_error(y_val,pred))
        total_error += loss/N_SPLITS
        test[f'mean_{target}_pred'] += reg.predict(X_test)/N_SPLITS
    
    
    oof_error += total_error
    
print("mean columnwise root mean squared error:",oof_error/3)
test
ss['id'] = 'id_' + ss['id_seqpos'].str.split('_', expand=True)[1]

# Merge my predicted average values
ss_new = ss. \
    drop(['reactivity','deg_Mg_pH10','deg_Mg_50C'], axis=1) \
    .merge(test[['id',
               'mean_reactivity_pred',
               'mean_deg_Mg_pH10_pred',
               'mean_deg_Mg_50C_pred']] \
               .rename(columns={'mean_reactivity_pred' : 'reactivity',
                                'mean_deg_Mg_pH10_pred': 'deg_Mg_pH10',
                                'mean_deg_Mg_50C_pred' : 'deg_Mg_50C'}
                      ),
         on='id',
        validate='m:1')
ss_new[ss.columns]
# Make Submission
ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
ss_new[ss.columns].to_csv('submission_lgbm_v1.csv', index=False)
ss
