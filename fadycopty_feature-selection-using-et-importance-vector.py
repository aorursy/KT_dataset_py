import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import KFold

import warnings

import gc

import time

import sys

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import metrics

# Plotly library

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', 500)



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import StratifiedKFold

from scipy.stats import uniform

from scipy.stats import randint



from hyperopt import STATUS_OK

from hyperopt import hp

from hyperopt import tpe

from hyperopt import Trials

from hyperopt import fmin
!mkdir output
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
drop = ['Census_ProcessorClass',

 'Census_IsWIMBootEnabled',

 'IsBeta',

 'Census_IsFlightsDisabled',

 'Census_IsFlightingInternal',

 'AutoSampleOptIn',

 'Census_ThresholdOptIn',

 'SMode',

 'Census_IsPortableOperatingSystem',

 'PuaMode',

 'Census_DeviceFamily',

 'UacLuaenable',

 'Census_IsVirtualDevice',

 'Platform',

 'Census_OSSkuName',

 'Census_OSInstallLanguageIdentifier',

 'Processor']



nrows = 20000



train = pd.read_csv('../input/train.csv',

                    nrows = nrows,                

                    dtype = dtypes)

train.shape

train.drop(drop, axis=1, inplace=True)
train.shape
target = train['HasDetections']

del train['HasDetections']

train.shape
target.shape
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.impute import SimpleImputer



X = pd.get_dummies(train)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, target)



my_imputer = SimpleImputer()

X_train = my_imputer.fit_transform(X_train)



X_train.shape
clf = ExtraTreesClassifier(n_estimators=5)

clf = clf.fit(X_train, y_train)

clf.feature_importances_  



model = SelectFromModel(clf, prefit=True)

train_new = model.transform(X_train)

train_new.shape               

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=70)

rf = rf.fit(train_new, y_train)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, rf.predict_proba(train_new)[:,1])
X_test_impute = my_imputer.transform(X_test)

X_test_reduce = model.transform(X_test_impute)
roc_auc_score(y_test, rf.predict_proba(X_test_reduce)[:,1])