import warnings

warnings.simplefilter('ignore')

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import gc

import multiprocessing

import time

from time import strftime

import seaborn as sns

import datetime

pd.set_option('display.max_columns', 83)

pd.set_option('display.max_rows', 83)

plt.style.use('seaborn')

for package in [pd, np, sns]:

    print(package.__name__, 'version:', package.__version__)

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

#         'OsBuild':                                              'int16',

        'OsSuite':                                              'int16',

#         'OsPlatformSubRelease':                                 'category',

        'OsBuildLab':                                           'category',

        'SkuEdition':                                           'category',

#         'IsProtected':                                          'float16',

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

#         'Census_OSArchitecture':                                'category',

        'Census_OSBranch':                                      'category',

        'Census_OSBuildNumber':                                 'int16',

        'Census_OSBuildRevision':                               'int32',

        'Census_OSEdition':                                     'category',

#         'Census_OSSkuName':                                     'category',

        'Census_OSInstallTypeName':                             'category',

        'Census_OSInstallLanguageIdentifier':                   'float16',

#         'Census_OSUILocaleIdentifier':                          'int16',

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

def load_dataframe(dataset):

    usecols = dtypes.keys()

    if dataset == 'test':

        usecols = [col for col in dtypes.keys() if col != 'HasDetections']

    df = pd.read_csv(f'../input/{dataset}.csv', dtype=dtypes, usecols=usecols)

    return df

with multiprocessing.Pool() as pool:

    train, test = pool.map(load_dataframe, ["train", "test"])
def plot_features_interaction_binary(df, first_feature, second_feature, first_feature_values, second_feature_values, target):

    """

    This function plots two features interaction along with target value for a binary classification task

    Args:

        df (pandas.DataFrame): dataset

        first_feature (str): name of the first feature

        second_feature (str): name of the second feature

        first_feature_values (int): number of values out of the first feature to plot

        second_feature_values (int): number of values out of the second feature to plot

        target (str): name of the target feature

    """

    # If first_feature_values argument is not set we are using all possible values of the first_feature

    if not first_feature_values:

        first_feature_values = df[first_feature].value_counts(dropna=False).shape[0]

    # If second_feature_values argument is not set we are using all possible values of the second_feature

    if not second_feature_values:

        second_feature_values = df[second_feature].value_counts(dropna=False).shape[0]

    # If first_feature_values argument exceeds the number of different features in first_feature we are resetting it

    first_feature_values = min(first_feature_values, df[first_feature].value_counts(dropna=False).shape[0])

    # If second_feature_values argument exceeds the number of different features in second_feature we are resetting it

    second_feature_values = min(second_feature_values, df[second_feature].value_counts(dropna=False).shape[0])

    

    # One barplot per row

    # Number of rows equals to a number of features for the first_feature

    fig, axes = plt.subplots(nrows=first_feature_values, ncols=1, figsize=(14, first_feature_values * 6))

    fig.subplots_adjust(hspace=1)

        

    for i in range(first_feature_values):

        # Handling NaN values

        if pd.isna(train[first_feature].value_counts(dropna=False).index[i]):

            features_interaction_df = df.loc[df[first_feature].isnull(), second_feature].value_counts(True, dropna=False).head(second_feature_values)

        else:

            features_interaction_df = df.loc[df[first_feature] == df[first_feature].value_counts(dropna=False).index[i], second_feature].value_counts(True, dropna=False).head(second_feature_values)

        features_interaction_df.plot(kind='bar', ax=axes[i], fontsize=14, rot=45).set_xlabel(second_feature, fontsize=14);

        for j in range(min(second_feature_values, features_interaction_df.shape[0])):

            try:

                # Again handling NaN values this time for both features

                # I'm pretty sure it might be done in more elegant way

                if pd.isna(train[first_feature].value_counts(dropna=False).index[i]) and not pd.isna(features_interaction_df.index[j]):

                    detection_rate = df.loc[(df[first_feature].isnull()) & (df[second_feature] == features_interaction_df.index[j]), target].value_counts(True)[1]

                elif not pd.isna(train[first_feature].value_counts(dropna=False).index[i]) and pd.isna(features_interaction_df.index[j]):

                    detection_rate = df.loc[(df[first_feature] == df[first_feature].value_counts(dropna=False).index[i]) & (df[second_feature].isnull()), target].value_counts(True)[1]

                elif pd.isna(train[first_feature].value_counts(dropna=False).index[i]) and pd.isna(features_interaction_df.index[j]):

                    detection_rate = df.loc[(df[first_feature].isnull()) & (df[second_feature].isnull()), target].value_counts(True)[1]

                else:

                    detection_rate = df.loc[(df[first_feature] == df[first_feature].value_counts(dropna=False).index[i]) & (df[second_feature] == features_interaction_df.index[j]), target].value_counts(True)[1]

            except:

                detection_rate = 0

            axes[i].plot(j, detection_rate, marker='.', color="black", markersize=22)

            axes[i].text(j + 0.1, y=detection_rate, s="%.2f" % detection_rate, fontsize=16, fontweight='bold')

        axes[i].set_title(first_feature + ': ' + str(df[first_feature].value_counts(dropna=False).index[i]) + ' - {0} values ({1:.2f}% of total)'.format(df[first_feature].value_counts(dropna=False).values[i], (df[first_feature].value_counts(dropna=False).values[i] / train[first_feature].value_counts(dropna=False).values.sum()) * 100), fontsize=18);
plot_features_interaction_binary(train, 'SmartScreen', 'Census_OSInstallTypeName', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Platform', 'SmartScreen', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'OsVer', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Processor', 'SmartScreen', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'Census_OEMModelIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'Wdft_IsGamer', 'SmartScreen', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'Census_FirmwareVersionIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'CityIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'AVProductStatesIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'Census_ProcessorModelIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'AppVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'EngineVersion', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'AVProductsInstalled', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'LocaleEnglishNameIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'SmartScreen', 'GeoNameIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'IsBeta', 'SmartScreen', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'AppVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Wdft_IsGamer', 'AVProductStatesIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Census_OSInstallTypeName', 'AVProductStatesIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'EngineVersion', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductsInstalled', 'AVProductStatesIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'Platform', 'AVProductStatesIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'Processor', 'AVProductStatesIdentifier', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'OsVer', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'Census_OEMModelIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'Census_FirmwareVersionIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'CityIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'Census_ProcessorModelIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'AppVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'LocaleEnglishNameIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AVProductStatesIdentifier', 'GeoNameIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'IsBeta', 'AVProductStatesIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'Census_OSInstallTypeName', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Platform', 'AppVersion',10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'OsVer', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Processor', 'AppVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'Census_OEMModelIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Wdft_IsGamer', 'AppVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'Census_FirmwareVersionIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'CityIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'AVProductStatesIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'Census_ProcessorModelIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'EngineVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'AVProductsInstalled', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'LocaleEnglishNameIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'GeoNameIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'Census_InternalPrimaryDisplayResolutionHorizontal', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'Census_TotalPhysicalRAM', 10, 15, 'HasDetections')
plot_features_interaction_binary(train, 'AppVersion', 'OsSuite', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'IsBeta', 'AppVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'Census_OSInstallTypeName', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'Platform', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'OsVer', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Processor', 'EngineVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'Census_OEMModelIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'Wdft_IsGamer', 'EngineVersion', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'Census_FirmwareVersionIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'CityIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'AVProductStatesIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'Census_ProcessorModelIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'AVProductsInstalled', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'LocaleEnglishNameIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'GeoNameIdentifier', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'Census_InternalPrimaryDisplayResolutionHorizontal', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'Census_TotalPhysicalRAM', 10, 10, 'HasDetections')
plot_features_interaction_binary(train, 'EngineVersion', 'OsSuite', 10, 10, 'HasDetections')