!pip install fastai==0.7.0 --no-deps
import os

import numpy as np

import pandas as pd

import lightgbm

import warnings

warnings.filterwarnings('ignore')

from IPython.display import FileLink

from sklearn.metrics import roc_auc_score

from fastai.structured import proc_df, train_cats

from sklearn.model_selection import train_test_split



print(os.listdir("../input"))
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
%time df_raw = pd.read_csv('../input/train.csv', dtype=dtypes)
imp_col = ['AvSigVersion', 'CountryIdentifier', 'MachineIdentifier','Census_SystemVolumeTotalCapacity', 'Census_FirmwareVersionIdentifier',

           'CityIdentifier', 'Census_OEMModelIdentifier','AVProductStatesIdentifier', 'LocaleEnglishNameIdentifier','Census_OSVersion', 

          'AppVersion', 'Census_OSBuildRevision', 'Census_OSUILocaleIdentifier', 'Census_PrimaryDiskTotalCapacity','OsBuildLab', 

           'SmartScreen', 'EngineVersion', 'Census_TotalPhysicalRAM', 'Census_ChassisTypeName', 'Census_OSEdition', 'Census_MDC2FormFactor',

          'Census_OSBranch', 'Census_InternalBatteryNumberOfCharges', 'Census_OSSkuName', 'Census_ActivationChannel', 

           'Census_OSInstallTypeName', 'HasDetections']

imp_col_test = ['AvSigVersion', 'CountryIdentifier', 'MachineIdentifier','Census_SystemVolumeTotalCapacity', 'Census_FirmwareVersionIdentifier',

           'CityIdentifier', 'Census_OEMModelIdentifier','AVProductStatesIdentifier', 'LocaleEnglishNameIdentifier','Census_OSVersion', 

          'AppVersion', 'Census_OSBuildRevision', 'Census_OSUILocaleIdentifier', 'Census_PrimaryDiskTotalCapacity','OsBuildLab', 

           'SmartScreen', 'EngineVersion', 'Census_TotalPhysicalRAM', 'Census_ChassisTypeName', 'Census_OSEdition', 'Census_MDC2FormFactor',

          'Census_OSBranch', 'Census_InternalBatteryNumberOfCharges', 'Census_OSSkuName', 'Census_ActivationChannel', 

           'Census_OSInstallTypeName']

imp_col2 = ['AvSigVersion', 'CountryIdentifier', 'MachineIdentifier','Census_SystemVolumeTotalCapacity', 'Census_FirmwareVersionIdentifier',

           'CityIdentifier', 'Census_OEMModelIdentifier','AVProductStatesIdentifier', 'LocaleEnglishNameIdentifier','Census_OSVersion', 

          'AppVersion', 'Census_OSBuildRevision', 'Census_OSUILocaleIdentifier', 'Census_PrimaryDiskTotalCapacity','OsBuildLab', 

           'SmartScreen', 'EngineVersion', 'Census_TotalPhysicalRAM', 'Census_ChassisTypeName', 'Census_OSEdition', 'Census_MDC2FormFactor',

          'Census_OSBranch', 'Census_InternalBatteryNumberOfCharges', 'Census_OSSkuName', 'Census_ActivationChannel_Volume:GVLK', 

           'Census_OSInstallTypeName_UUPUpgrade']
df = df_raw[imp_col]

train_cats(df)

%time X, y, nas = proc_df(df, 'HasDetections', max_n_cat=10)
X = X[imp_col2]; X.shape
X_trn, X_vld, y_trn, y_vld = train_test_split(X, y, test_size =0.0)
params = {}

params['learning_rate'] = 0.3

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 2**12-1

params['min_data'] = 50

params['max_depth'] = -1

train_data = lightgbm.Dataset(X_trn, label=y_trn)
%time clf = lightgbm.train(params, train_data, 100)
%time df_test = pd.read_csv('../input/test.csv', dtype=dtypes)
del(df_raw, df)
del(X, X_trn, X_vld, y_trn, y_vld)
test = df_test[imp_col_test]
train_cats(test)

%time X_test, _, nas = proc_df(test, na_dict=nas, max_n_cat=10)
X = X_test[imp_col2]; X.shape
preds= clf.predict(X)
submission_data = test[['MachineIdentifier']]
submission_data['HasDetections'] = preds
submission_data.to_csv('sub.csv', index=False)
submission_data.head()
FileLink('sub.csv')