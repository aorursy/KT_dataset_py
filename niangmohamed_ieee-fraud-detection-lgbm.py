# General imports

import numpy as np

import pandas as pd

import os, sys, gc, warnings, random, datetime



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%matplotlib inline



from sklearn import metrics

from sklearn.model_selection import train_test_split, KFold, GroupKFold

from sklearn.preprocessing import LabelEncoder



import lightgbm as lgb

from tqdm import tqdm



import math

warnings.filterwarnings('ignore')



os.chdir('/kaggle/input/ieeefewithpreprocessing') # Set working directory

print(os.listdir('/kaggle/input/ieeefewithpreprocessing'))
# Helpers



# Seeder

# :seed to make all processes deterministic     # type: int

def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

# Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
# Data Load



train_df = reduce_mem_usage(pd.read_pickle('train_df.pkl'))

test_df = reduce_mem_usage(pd.read_pickle('test_df.pkl'))



remove_features = pd.read_pickle('remove_features.pkl')

remove_features = list(remove_features['features_to_remove'].values)



print('Load Data')

print('Shape control:', train_df.shape, test_df.shape)
# Vars

SEED = 42

seed_everything(SEED)

target = 'isFraud'

N_SPLITS = 5

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
# Model params

lgb_params = {

                    'objective':'binary',

                    'boosting_type':'gbdt',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.01,

                    'num_leaves': 2**8,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.7,

                    'subsample_freq':1,

                    'subsample':0.7,

                    'n_estimators':20000,

                    'max_bin':255,

                    'verbose':-1,

                    'seed': SEED,

                    'early_stopping_rounds':100, 

                } 
# Final features list

features_columns = [col for col in list(train_df) if col not in remove_features]
# Model Train

   

folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)



X,y = train_df[features_columns], train_df[target]    

P = test_df[features_columns]  



predictions = np.zeros(len(test_df))

oof = np.zeros(len(train_df))
del train_df, test_df, features_columns

gc.collect()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

    print('Fold:',fold_+1)

    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]

    vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]

            

    print(len(tr_x),len(vl_x))

    tr_data = lgb.Dataset(tr_x, label=tr_y)

    vl_data = lgb.Dataset(vl_x, label=vl_y)  



    estimator = lgb.train(

        lgb_params,

        tr_data,

        valid_sets = [tr_data, vl_data],

        verbose_eval = 1000,

    )   

        

    pp_p = estimator.predict(P)

    predictions += pp_p/N_SPLITS

        

    oof_preds = estimator.predict(vl_x)

    oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())

        

    del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data

    gc.collect()
print('OOF AUC:', metrics.roc_auc_score(y, oof))
print(metrics.confusion_matrix(y, oof.round()))
print(metrics.classification_report(y, oof.round()))
feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:50])

plt.title('LGBM with cross validation Most Important Features')

plt.tight_layout()

plt.show()
submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')

submission.isFraud = predictions

submission.head()
plt.hist(submission.isFraud,bins=100)

plt.ylim((0,5000))

plt.title('LGBM with cross validation submission')

plt.show()
submission.to_csv('/kaggle/working/lgbm_cv_submission.csv')