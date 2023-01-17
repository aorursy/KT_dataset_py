import pandas as pd

import numpy as np

np.set_printoptions(precision=4)

import catboost

from catboost import datasets

from catboost import *

print("catboost version:", catboost.__version__)



# Read data from Amazon.com_Employee Access Challenge

train_df = pd.read_csv('../input/amazoncom-employee-access-challenge/train.csv')

test_df = pd.read_csv('../input/amazoncom-employee-access-challenge/test.csv')



y = train_df.ACTION

X = train_df.drop('ACTION', axis=1)

cat_features = list(range(0, X.shape[1]))



# train/valid split

train_count = int(X.shape[0] * 0.8)

X_train = X.iloc[:train_count,:]

y_train = y[:train_count]

X_validation = X.iloc[train_count:, :]

y_validation = y[train_count:]



# Use widely default params to show the difference

params = {'learning_rate': 0.05, 'iterations': 500, 'random_seed': 3,

          'custom_loss': ['Accuracy']}



mod1 = CatBoostClassifier(**params, task_type='GPU')

mod2 = CatBoostClassifier(**params, task_type='CPU')



args = (X_train, y_train)

kwargs = {'eval_set': (X_validation, y_validation), 'cat_features': cat_features, 'verbose': 100}



print("train on GPU (mod1)...")

mod1.fit(*args, **kwargs)



print("train on CPU (mod2)...")

mod2.fit(*args, **kwargs)
# Compare all params that are different

params_gpu = mod1.get_all_params()

params_cpu = mod2.get_all_params()



for k in set(params_cpu.keys())|set(params_gpu.keys()):

    val_gpu = params_gpu[k] if k in params_gpu.keys() else 'None'

    val_cpu = params_cpu[k] if k in params_cpu.keys() else 'None'

    if val_cpu == val_gpu: continue

    print(f'{k:<30}  {str(val_cpu):<40}  {str(val_gpu):<40}')
params = {'learning_rate': 0.05, 'iterations': 500, 'random_seed': 3,

          'custom_loss': ['Accuracy']}



# Try to use the same params for both CPU/GPU

params['bootstrap_type'] = 'MVS'

params['boosting_type'] = 'Plain'         # GPU: much worse (because dataset is small)

params['boosting_type'] = 'Ordered'       # CPU: no difference

params['model_shrink_mode'] = 'Constant'  # default, ignored by GPU

params['model_shrink_rate'] = 0           # default for mode=Constant (should not shrink at all), ignored by GPU

params['sampling_frequency'] = 'PerTree'  # doc error: default is 'PerTree' not 'PerTreeLevel'

params['posterior_sampling'] = False      # same as None? ignored by GPU

params['bagging_temperature'] = 1         # same as None

params['border_count'] = 254              # no impact, CPU default: 254, GPU default: 128

params['penalties_coefficient'] = 1       # same as None

params['fold_permutation_block'] = 64     # 0 ignored by GPU

params['subsample'] = 0.8                 # no impact



# ctr: FeatureFreq is not implemented on CPU, Counter not on GPU, use only Border for both.

params['simple_ctr'] = ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1']

params['combinations_ctr'] = ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1']



params['ctr_history_unit'] = 'Sample'     # undocumented, ignored by CPU

#params['fold_size_loss_normalization'] = False  # unexpected keyword argument by GPU

#params['min_fold_size'] = 100             # undocumented, unexpected keyword argument by GPU

#params['observations_to_bootstrap'] = 'TestOnly'  # undocumented, unexpected keyword argument for GPU



# Same default for both (but mentioned in Issue report)

params['leaf_estimation_method'] = 'Newton'                # same default for both

params['leaf_estimation_iterations'] = 10                  # same default for both

params['leaf_estimation_backtracking'] = 'AnyImprovement'  # same default for both



mod1 = CatBoostClassifier(task_type='GPU', **params)

mod2 = CatBoostClassifier(task_type='CPU', **params)



args = (X_train, y_train)

kwargs = {'eval_set': (X_validation, y_validation), 'cat_features': cat_features, 'verbose': 100}



print("train on GPU (mod1)...")

mod1.fit(*args, **kwargs)  # 1 min setup time



print("train on CPU (mod2)...")

mod2.fit(*args, **kwargs)