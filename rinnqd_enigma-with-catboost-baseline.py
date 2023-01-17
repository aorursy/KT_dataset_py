import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



from tqdm import tqdm_notebook

import lightgbm as lgb

from catboost import Pool, CatBoostClassifier



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline

import seaborn as sns







plt.style.use('seaborn')

sns.set(font_scale=1)



import gc



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample = pd.read_csv('../input/sample_submission.csv')
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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)

sample = reduce_mem_usage(sample)
gc.collect()
train.head()
cols = train.columns[1:-1]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5168)

# oof = df_train[['ID_code', 'target']]

# oof['predict'] = 0

# predictions = test[['ID_code']]

# feature_importance_df = pd.DataFrame()

# val_aucs = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):

    X_train, y_train = train.iloc[trn_idx][cols], train.iloc[trn_idx]['target']

    X_valid, y_valid = train.iloc[val_idx][cols], train.iloc[val_idx]['target']

    break

    

    

    

clf = CatBoostClassifier(loss_function = "Logloss", eval_metric = "AUC",random_seed=123,use_best_model=True,

                          learning_rate=0.1,  iterations=15000,verbose=100,

                           bootstrap_type= "Poisson", 

                           task_type="GPU", 

#                              l2_leaf_reg= 16.5056753964314982, depth= 3.0,

#                              fold_len_multiplier= 2.9772639036842174, 

#                              scale_pos_weight= 3.542962442406767, 

#                              fold_permutation_block_size=16.0, subsample= 0.46893530376570957

#                              fold_len_multiplier=3.2685541035861747, 

#                              scale_pos_weight= 2.6496926337120916, 

#                              fold_permutation_block_size= 6.0, 

                          )

print("Model training")

clf.fit(X_train, y_train,  eval_set=(X_valid, y_valid), early_stopping_rounds=2000,verbose=100)

predict = clf.predict_proba(test[cols])
sample.target = predict[:,1]
sample
from IPython.display import FileLink

def create_submission(submission_file, submission_name):

    submission_file.to_csv(submission_name+".csv",index=False)

    return FileLink(submission_name+".csv")
create_submission(sample, "sub_c_15k_simple")