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
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from tqdm import tqdm
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
# ## Remove these since they seem to have very low importances -- for baseline establishment purposes -- perhaps add them back later
# low_imp_feats = [#'IsBeta',
# # 'AutoSampleOptIn',
# # 'IsBeta',
#  #'AutoSampleOptIn',
# # 'OsVer',
#  'Census_IsPortableOperatingSystem',
#  #'OsVer',
#  'Census_IsFlightsDisabled',
# # 'Census_IsPortableOperatingSystem',
# # 'Census_IsFlightsDisabled',
# # 'HasTpm',
#  'Census_IsPenCapable',
#  'Census_IsFlightingInternal',
# # 'HasTpm',
#  'Census_ProcessorManufacturerIdentifier',
#  'UacLuaenable',
#  'Census_IsWIMBootEnabled',
# # 'UacLuaenable',
#  'Census_IsWIMBootEnabled',
# # 'Census_IsPenCapable',
#  'Census_IsAlwaysOnAlwaysConnectedCapable',
#  'Census_IsTouchEnabled',
#  'Census_ProcessorManufacturerIdentifier',
#  'Census_HasOpticalDiskDrive',
#  'Census_IsFlightingInternal']
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
#categorical_columns = [c for c,v in dtypes.items() if v not in numerics]
categorical_columns = [
    'EngineVersion',
    'AppVersion',
    'AvSigVersion',
    'Platform',
    'Processor',
    'OsVer',
    'OsPlatformSubRelease',
    'OsBuildLab',
    'SkuEdition',
    'PuaMode',
    'SmartScreen',
    'Census_MDC2FormFactor',
    'Census_DeviceFamily',
    'Census_ProcessorClass',
    'Census_PrimaryDiskTypeName',
 ]
%%time
nrows = 2000000
retained_columns = numerical_columns + categorical_columns
train = pd.read_csv('../input/reduced-train-test-msft-malware-comp/train_reduced.csv',
                    nrows = nrows,
                    usecols = retained_columns)
#_______________________________________________________________
retained_columns += ['MachineIdentifier']
retained_columns.remove('HasDetections')
test = pd.read_csv('../input/reduced-train-test-msft-malware-comp/test_reduced.csv',
                   usecols = retained_columns)
true_numerical_columns = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges'
]

binary_variables = [c for c in train.columns if train[c].nunique() == 2]

categorical_columns = [c for c in train.columns 
                       if (c not in true_numerical_columns) & (c not in binary_variables)]
indexer = {}
for col in tqdm(categorical_columns):
    if train[col].dtype in numerics: continue
    _, indexer[col] = pd.factorize(train[col])
    
for col in tqdm(categorical_columns):
    if train[col].dtype in numerics: continue
    train[col] = indexer[col].get_indexer(train[col])
    test[col] = indexer[col].get_indexer(test[col])
target = train['HasDetections']
del train['HasDetections']
# cols_to_keep = list(set(train.columns) & set(test.columns))
# train = train[cols_to_keep]
# test = test[cols_to_keep]
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
# Let's try a CatBoost just for 3 iterations

max_iter=3

features = train.columns

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof_cb = np.zeros(len(train))
predictions_cb = np.zeros(len(test))
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print('-')
    print("Fold {}".format(fold_ + 1))
    X_train, y_train = train[features].iloc[trn_idx], target.iloc[trn_idx]
    X_valid, y_valid = train[features].iloc[val_idx], target.iloc[val_idx]
    
    
    model = cb.CatBoostClassifier(learning_rate = 0.25,
        iterations = 10000,
        eval_metric = 'AUC',
        allow_writing_files = False,
        od_type = 'Iter',
        bagging_temperature = 0.3,
        random_strength = 0.1,
        l2_leaf_reg = 0.1,
        depth = 8,
        od_wait = 20,
        silent = True)
    
            
    # Fit
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    oof_cb[val_idx] = model.predict(X_valid)
    predictions_cb += model.predict(test[features]) / min(folds.n_splits, max_iter)    
    
    score[fold_] = metrics.roc_auc_score(target.iloc[val_idx], oof_cb[val_idx])
    if fold_ == max_iter - 1: break
        
if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(metrics.roc_auc_score(target, oof_cb)))
else:
     print("CV score: {:<8.5f}".format(sum(score) / max_iter))
sub_df = pd.DataFrame({"MachineIdentifier": test["MachineIdentifier"].values})
sub_df["HasDetections"] = predictions_cb

sub_df.to_csv("submit.csv", index=False)
