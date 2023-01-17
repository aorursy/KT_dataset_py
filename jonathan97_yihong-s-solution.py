import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import os

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
%%time

train_df = pd.read_csv('../input/train.csv', dtype=dtypes)
%%time

test_df = pd.read_csv('../input/train.csv', dtype=dtypes)
train_df.head()
train_df.shape
to_drop = ['MachineIdentifier']
def DataCleaning(df):

    stats = []

    for col in df.columns:

        stats.append((col, 

                      df[col].nunique(), 

                      df[col].isnull().sum() / train_df.shape[0],

                      df[col].value_counts(normalize=True).values[0],

                      df[col].dtype))



    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Skewness', 'type'])

    return stats_df
stats_df = DataCleaning(train_df)

stats_df.sort_values('Percentage of missing values', ascending=False)
stats_df.sort_values('Skewness', ascending=False)
to_drop.extend(stats_df.loc[stats_df['Percentage of missing values'] > .95]['Feature'].tolist())

to_drop.extend(stats_df.loc[stats_df['Skewness'] > .9]['Feature'].tolist())

to_drop
train_df.drop(to_drop, axis=1, inplace=True)
stats_df = DataCleaning(train_df)

stats_df.sort_values('Percentage of missing values', ascending=False)
pd.options.display.max_rows = 99

train_df.Census_InternalBatteryType.value_counts()
trans_dict = {

    '˙˙˙': 'unknown', 'unkn': 'unknown', np.nan: 'unknown'

}

train_df.replace({'Census_InternalBatteryType': trans_dict}, inplace=True)
train_df.Census_InternalBatteryType.isnull().sum()
train_df.SmartScreen.value_counts()
trans_dict = {

    'off': 'Off', '&#x02;': '2', '&#x01;': '1', 'on': 'On', 'requireadmin': 'RequireAdmin', 'OFF': 'Off', 

    'Promt': 'Prompt', 'requireAdmin': 'RequireAdmin', 'prompt': 'Prompt', 'warn': 'Warn', 

    '00000000': '0', '&#x03;': '3', np.nan: 'NoExist'

}

train_df.replace({'SmartScreen': trans_dict}, inplace=True)
train_df.SmartScreen.isnull().sum()
train_df.OrganizationIdentifier.value_counts()
train_df.replace({'np': {np.nan: 0}}, inplace=True)
train_df.OrganizationIdentifier.isnull().sum()
train_df.shape
train_df.dropna(inplace=True)

train_df.shape
cols = train_df.columns.tolist()

plt.figure(figsize=(30,30))

sns.heatmap(train_df[cols].corr().abs(), cmap='RdBu_r', annot=True, center=0.0)

plt.show()
corr_matrix = train_df.corr()



def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i + 1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations:")

print(get_top_abs_correlations(corr_matrix, 20))
to_drop = []

if train_df.Census_OSInstallLanguageIdentifier.nunique() > train_df.Census_OSUILocaleIdentifier.nunique():

    to_drop.append('Census_OSInstallLanguageIdentifier')

else:

    to_drop.append('Census_OSUILocaleIdentifier')

if train_df.Census_InternalPrimaryDisplayResolutionHorizontal.nunique() > train_df.Census_InternalPrimaryDisplayResolutionVertical.nunique():

    to_drop.append('Census_InternalPrimaryDisplayResolutionHorizontal')

else:

    to_drop.append('Census_InternalPrimaryDisplayResolutionVertical')

if train_df.OsBuild.nunique() > train_df.Census_OSBuildNumber.nunique():

    to_drop.append('OsBuild')

else:

    to_drop.append('Census_OSBuildNumber')

if train_df.Census_ProcessorManufacturerIdentifier.nunique() > train_df.Census_ProcessorModelIdentifier.nunique():

    to_drop.append('Census_ProcessorManufacturerIdentifier')

else:

    to_drop.append('Census_ProcessorModelIdentifier')

if train_df.AVProductStatesIdentifier.nunique() > train_df.Census_ProcessorModelIdentifier.nunique():

    to_drop.append('AVProductStatesIdentifier')

else:

    to_drop.append('Census_ProcessorModelIdentifier')
if train_df.Census_ProcessorManufacturerIdentifier.nunique() > train_df.Census_ProcessorModelIdentifier.nunique():

    to_drop.append('Census_ProcessorManufacturerIdentifier')

else:

    to_drop.append('Census_ProcessorModelIdentifier')
if train_df.Census_ProcessorManufacturerIdentifier.nunique() > train_df.Census_ProcessorModelIdentifier.nunique():

    to_drop.append('Census_ProcessorManufacturerIdentifier')

else:

    to_drop.append('Census_ProcessorModelIdentifier')
if train_df.Census_ProcessorManufacturerIdentifier.nunique() > train_df.Census_ProcessorModelIdentifier.nunique():

    to_drop.append('Census_ProcessorManufacturerIdentifier')

else:

    to_drop.append('Census_ProcessorModelIdentifier')
train_df.drop(to_drop, axis=1, inplace=True)

train_df.shape
sample = train_df.sample(50000, random_state=42)

X = sample.drop('HasDetections', axis=1).values

y = sample['HasDetections'].values
from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras.callbacks import LearningRateScheduler

from keras.optimizers import Adam



model = Sequential()

model.add(Dense(100,input_dim=X.shape[1]))

model.add(Dropout(0.4))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dense(100))

model.add(Dropout(0.4))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])
from sklearn.model_selection import StratifiedKFold



skf = StratifiedKFold(n_splits=5)



for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train, epochs=20, batch_size=32)

    score = model.evaluate(X_test, y_test, batch_size=32)

    print(score)