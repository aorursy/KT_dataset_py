# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Now load the training set
print('Loading the training set:')
#https://www.kaggle.com/theoviel/load-the-totality-of-the-data
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
training_set = pd.read_csv('../input/train.csv', dtype=dtypes)
print('Traiaing set loaded')
print(training_set.shape)
columns_to_remove = []
for col_name in training_set.columns.values:
    if col_name == 'HasDetections' or col_name == 'MachineIdentifier':
        continue
    unique_values = training_set[col_name].value_counts(dropna=False)
    msg = 'column ' + col_name + ' have ' + str(len(unique_values)) + ' unique values. The bigger category has ' + str(100 * unique_values.values[0] / training_set.shape[0]) + ' percent of the data'
    if unique_values.values[0] / training_set.shape[0] > 0.9:
        msg = msg + " - removed"
        del training_set[col_name]  
        columns_to_remove.append(col_name)
    print(msg)

print('')
print('Untill now ' + str(len(columns_to_remove)) + ' colums removed')
print(training_set.shape)
p = training_set['HasDetections'].mean() # p is the probability that a computer is detected
print('The detection rate is ' + str(p))
significantTh = training_set.shape[0] * 0.1
for col_name in training_set.columns.values:
    if col_name == 'HasDetections' or col_name == 'MachineIdentifier':
        continue
    unique_values = training_set[col_name].value_counts(dropna=False)
    nSignificant = 0
    for i in range(len(unique_values)):
        val = unique_values.index[i]
        n = unique_values.values[i]
        # Calculate the probability P(detect | col_name=val)
        currTable = training_set[training_set[col_name] == val]
        pCurr = currTable['HasDetections'].mean()
        # We are in binomial distribution where E(X)=p and std(X)=sqrt(p*(1-p)/n). We want abs(p-pCurr) > std
        std = np.sqrt(p * (1 - p) / n)
        if abs(p-pCurr) > 3*std:
            nSignificant = nSignificant + n
            if nSignificant > significantTh:
                # We have significant parameter. No need to check more
                break
    if nSignificant < significantTh:
        columns_to_remove.append(col_name)
        del training_set[col_name]
        print('Only ' + str(100*nSignificant/training_set.shape[0]) + '% of the values in column ' + col_name +
              ' is significant - Remove the column')
print('')
print('Untill now ' + str(len(columns_to_remove)) + ' colums removed')
print(training_set.shape)
choosen_cols = training_set.columns.values
print(choosen_cols)