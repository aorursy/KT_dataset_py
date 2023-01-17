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
continuous_columns = [ # All the columns which have a real continuous data
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges',
    'Census_OSBuildNumber',
    'Census_OSBuildRevision',
    'Census_ThresholdOptIn',
    'OsBuild'
]
version_columns = [ # All the columns which have a version data e.g. 1.1.12603.0
    'EngineVersion',
    'AppVersion',
    'AvSigVersion',
    'OsVer',
    'Census_OSVersion'
]
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
for col_name in training_set.columns.values:
    if col_name == 'HasDetections' or col_name == 'MachineIdentifier' or col_name in continuous_columns or col_name in version_columns:
        continue
    unique_values = training_set[col_name].value_counts(dropna=False)
    if len(unique_values)>500:
        # print('Column ' + col_name + ' has ' + str(len(unique_values)) - " unique values - remove")
        del training_set[col_name]  
        columns_to_remove.append(col_name)
print('')
print('Untill now ' + str(len(columns_to_remove)) + ' colums removed')
print(training_set.shape)
choosen_cols = training_set.columns.values
print(choosen_cols)
def create_conversion_dict():
    conv_dict = dict()
    for col_name in training_set.columns.values:
        if col_name == 'HasDetections' or col_name == 'MachineIdentifier' or col_name in continuous_columns or col_name in version_columns:
            continue
        unique_values = training_set[col_name].value_counts(dropna=False)
        if training_set[col_name].dtypes == 'int8' and len(unique_values) == 2:
            continue # No need to convert it to ints because it has only true and false
        val_and_n = []
        for category in unique_values.index:
            curr_table = training_set['HasDetections'][training_set[col_name] == category]
            detected_ratio = curr_table.mean()
            val_and_n.append((str(category), detected_ratio))
        val_and_n.sort(key=lambda r: r[1])
        val_to_index = {val_and_n[idx][0]: idx for idx in range(len(val_and_n))}
        conv_dict[col_name] = val_to_index
    return conv_dict

print('Creating conversion dictionary')
categoryToInt = create_conversion_dict()
print('Done')
def apply_version_conv(version_str, loc):
    if version_str == '':
        return np.nan
    tokens = version_str.split('.')
    if not tokens[loc].isdigit():
        return np.nan
    return float(tokens[loc])


def apply_conv(conv_dict, v):
    if v in conv_dict:
        return conv_dict[v]
    return np.nan

def convert_categories(data, conv_dict):
    for col_name in data.columns.values:
        if col_name in conv_dict.keys():
            tmp_col = data[col_name].apply(lambda v: apply_conv(conv_dict[col_name], str(v))).astype(np.float32)
            data[col_name] = tmp_col

    for col_name in version_columns:
        if col_name not in data.columns.values:
            continue
        for loc in range(4):
            tmp_col = data[col_name].apply(lambda v: apply_version_conv(str(v), loc)).astype(np.float32)
            curr_col_name = col_name + '_' + str(loc)
            data[curr_col_name] = tmp_col
        del data[col_name]

    return data


print('Cronverting the training set')
training_set = convert_categories(training_set, categoryToInt)
print('Done')
print(training_set.dtypes)
print(training_set.head(10))
print('Saving the trainig data')
training_set.to_csv('training_decisionTrees.csv', index=False, float_format='%g')
print('Done')
del training_set
test_set = pd.read_csv('../input/test.csv', dtype=dtypes)
print('Test set loaded')
print(test_set.shape)
print('Removing columns')
for col in columns_to_remove:
    del test_set[col]
print(test_set.shape)
print('Convert the test set')
test_set = convert_categories(test_set, categoryToInt)
print(test_set.dtypes)
print('Saving the test data')
test_set.to_csv('test_decisionTrees.csv', index=False, float_format='%g')
print('Done')