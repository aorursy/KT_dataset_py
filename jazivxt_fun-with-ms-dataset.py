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
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import time, datetime
from sklearn import *

train = pd.read_csv('../input/train.csv', iterator=True, chunksize=1_500_000, dtype=dtypes)
test = pd.read_csv('../input/test.csv', iterator=True, chunksize=1_000_000, dtype=dtypes)
gf_defaults = {'col': [], 'ocol':[], 'dcol' : ['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsBuildLab', 'Census_OSVersion']}
one_hot = {}

def get_features(df, gf_train=False):
    global one_hot
    global gf_defaults
    
    for c in gf_defaults['dcol']:
        for i in range(5):
            df[c + str(i)] = df[c].map(lambda x: str(x).split('.')[i] if len(str(x).split('.'))>i else -1)

    col = [c for c in df.columns if c not in ['MachineIdentifier', 'HasDetections']]
    if gf_train:
        for c in col:
            if df[c].dtype == 'O' or df[c].dtype.name == 'category':
                gf_defaults['ocol'].append(c)
            else:
                gf_defaults['col'].append(c)
        one_hot = {c: list(df[c].value_counts().index) for c in gf_defaults['ocol']}

    #train and test
    for c in one_hot:
        if len(one_hot[c])>1 and len(one_hot[c]) < 20:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
                if gf_train:
                    gf_defaults['col'].append(c+'_oh_' + str(val))
    return df[gf_defaults['col']+['MachineIdentifier', 'HasDetections']]
col = gf_defaults['col']
model = []
params = {'objective':'binary', "boosting": "gbdt", 'learning_rate': 0.02, 'max_depth': -1, 
         "feature_fraction": 0.8, "bagging_freq": 1, "bagging_fraction": 0.8 , "bagging_seed": 11,
         "metric": 'auc', "lambda_l1": 0.1, 'num_leaves': 60, 'min_data_in_leaf': 60, "verbosity": -1, "random_state": 3}
online_start = True
for df in train:
    if online_start:
        df = get_features(df, True)
        x1, x2, y1, y2 = model_selection.train_test_split(df[col], df['HasDetections'], test_size=0.2, random_state=25)
        model = lgb.train(params, lgb.Dataset(x1, y1), 2500,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=200)
        model.save_model('lgb.model')
    else:
        df = get_features(df)
        x1, x2, y1, y2 = model_selection.train_test_split(df[col], df['HasDetections'], test_size=0.2, random_state=25)
        model = lgb.train(params, lgb.Dataset(x1, y1), 2500,  lgb.Dataset(x2, y2), verbose_eval=100, early_stopping_rounds=200, init_model='lgb.model')
        model.save_model('lgb.model')
    online_start = False
    print('training...')
predictions = []
for df in test:
    df['HasDetections'] = 0.0
    df = get_features(df)
    df['HasDetections'] = model.predict(df[col], num_iteration=model.best_iteration + 50)
    predictions.append(df[['MachineIdentifier', 'HasDetections']].values)
    print('testing...')
sub = np.concatenate(predictions)
sub = pd.DataFrame(sub, columns = ['MachineIdentifier', 'HasDetections'])
sub.to_csv('submission.csv', index=False)