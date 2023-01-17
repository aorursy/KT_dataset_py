# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
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



dtypes

import gc

gc.enable()



print('Load Train Start.\n')

train = pd.read_csv('/kaggle/input/microsoft-malware-prediction/train.csv', dtype=dtypes, low_memory=True)

print('Load Train Done.\n')



print('Load Text Start.\n')

test  = pd.read_csv('/kaggle/input/microsoft-malware-prediction/test.csv',  dtype=dtypes, low_memory=True)

test['MachineIdentifier']  = test.index.astype('uint32')

print('Load Tst Done.\n')



gc.collect()
train.head()
train['MachineIdentifier'] = train.index.astype('uint32')

train.info()
train.head()
train.tail()
train.columns
train.columns.tolist()[1:-1]
# all features to category

print('change all features to category.')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



for usecol in train.columns.tolist()[1:-1]:

    print("start loop: " + usecol)

    train[usecol] = train[usecol].astype('str')

    test[usecol] = test[usecol].astype('str')

    

    #Fit LabelEncoder for train and test datasets

    le = LabelEncoder().fit(np.unique(train[usecol].unique().tolist() + test[usecol].unique().tolist()))

#     print("Label Encoder ")



    # Transform labelEncoder

    train[usecol] = le.transform(train[usecol])+1

    test[usecol]  = le.transform(test[usecol])+1



    # aggregate MachineID as count for train and test data

    agg_tr = (train

              .groupby([usecol])

              .aggregate({'MachineIdentifier':'count'})

              .reset_index()

              .rename({'MachineIdentifier':'Train'}, axis=1))

    agg_te = (test

              .groupby([usecol])

              .aggregate({'MachineIdentifier':'count'})

              .reset_index()

              .rename({'MachineIdentifier':'Test'}, axis=1))

    

    # any dropped values will changes to 0

    agg = pd.merge(agg_tr, agg_te, on=usecol, how='outer').replace(np.nan, 0)

    

    #

    #

    #Select values with more than 1000 observations

    agg = agg[(agg['Train'] > 1000)].reset_index(drop=True)

    agg['Total'] = agg['Train'] + agg['Test']

    

    #

    #

    # aggregate date where divided value between 0.2 and 0.8

    agg = agg[(agg['Train'] / agg['Total'] > 0.2) & (agg['Train'] / agg['Total'] < 0.8)]

    

    # make a new coulumn for new data

    agg[usecol+'Copy'] = agg[usecol]

    agg[usecol+'Copy']

    #Drop unbalanced values

    train[usecol] = (pd.merge(train[[usecol]], 

                              agg[[usecol, usecol+'Copy']], 

                              on=usecol, how='left')[usecol+'Copy']

                     .replace(np.nan, 0).astype('int').astype('category'))



    test[usecol]  = (pd.merge(test[[usecol]], 

                              agg[[usecol, usecol+'Copy']], 

                              on=usecol, how='left')[usecol+'Copy']

                     .replace(np.nan, 0).astype('int').astype('category'))



    del le, agg_tr, agg_te, agg, usecol

    gc.collect()

train.head()
train.head(20)
train.tail()
test.tail()


y_train = train['HasDetections']



del train['HasDetections'], train['MachineIdentifier'], test['MachineIdentifier']

gc.collect()



y_train
# test_ids
train.info()
#Fit OneHotEncoder

# ohe = OneHotEncoder(categories='auto', sparse=True, dtype='uint8').fit(train)



#Transform data using small groups to reduce memory usage

from scipy.sparse import vstack, csr_matrix, save_npz, load_npz

from sklearn.model_selection import StratifiedKFold



skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

skf.get_n_splits(train.index, y_train)



test_result = np.zeros(len(test))





import lightgbm as lgb

print('Load Train Start.\n')

sample = pd.read_csv('/kaggle/input/microsoft-malware-prediction/sample_submission.csv')

print('Load Train Done.\n')

sample.head(20)


# for train_index, test_index in skf.split(train.index, y_train):    

#     print("start class for train index")

#     X_fit, X_val = train[train.index.isin(train_index)],   train[train.index.isin(test_index)]

#     y_fit, y_val = y_train[y_train.index.isin(train_index)], y_train[y_train.index.isin(test_index)]

#     model = lgb.LGBMClassifier(max_depth=-1,

#                                n_estimators=30000,

#                                learning_rate=0.1,

#                                colsample_bytree=0.2,

#                                objective='binary', 

#                                n_jobs=-1)

#     print("fit")

#     model.fit(X_fit, y_fit, eval_metric='auc', eval_set=[(X_val, y_val)], verbose=100, early_stopping_rounds=100)



#     test_result += model.predict_proba(test)[:,1]



#     del X_fit, X_val, y_fit, y_val

#     gc.collect()



#     print("end class ")



# # to_submit = pd.read_csv('/kaggle/input/microsoft-malware-prediction/sample_submission.csv')

# to_submit = sample

# to_submit['HasDetections'] = test_result / 6

# to_submit.to_csv('result.csv', index=False)

# to_submit = pd.read_csv('/kaggle/input/microsoft-malware-prediction/sample_submission.csv')

# to_submit['HasDetections'] = test_result / 6

# to_submit.to_csv('result.csv', index=False)
