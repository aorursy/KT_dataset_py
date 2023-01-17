# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
df = pd.read_csv('/kaggle/input/microsoft-malware-prediction/train.csv',dtype =dtypes )
df_test = test_set = df.sample(frac = 0.3,random_state= 5)

training_index = [x for x in df.index if x not in test_set.index]

data = training_set = df.iloc[training_index,1:-1]

training_y = df.iloc[training_index,-1]

data['HasDetections'] = training_y
data.head()
data.shape
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from time import time

import scipy as sp

%matplotlib inline

sns.set_style("whitegrid")

pd.options.display.max_columns = None
data.info()
null_df = data.isna()
print('PERCENTAGE OF MISSING VALUES:')

percent_of_null = {}

for column in null_df.columns:

    false_ratio = null_df[column].value_counts(normalize=True).loc[False]

    percent_of_null[column] = (1-false_ratio)*100
pomv = pd.DataFrame(data = percent_of_null.keys(), columns=['column'])

pomv['percentage'] = percent_of_null.values()

pomv.sort_values('percentage', ascending=False).head(10)
desc_df = data.describe()

desc_df
# finding the inter-quartile range for each column



intqtl_range = {col: desc_df[col].loc['75%'] - desc_df[col].loc['25%'] for col in desc_df.columns}



# Finding the upper and lower bound for each column



upper_bound = {col : desc_df[col].loc['75%'] + (intqtl_range[col]*1.5) for col in desc_df.columns}

lower_bound = {col : desc_df[col].loc['25%'] - (intqtl_range[col]*1.5) for col in desc_df.columns}



# Getting the number of instances with outliers for each column 

outlier_count = {col : len(data[(data[col]>upper_bound[col])|(data[col]<lower_bound[col])].index) for col in desc_df.columns}
#Creating a datafram for the columns with outliers



outlier = pd.DataFrame(np.array(list(outlier_count.items())),columns=['column','no_of_outliers'])

total = df.shape[0]

outlier['no_of_outliers'] = outlier['no_of_outliers'].apply(lambda val: int(val))

outlier['percentage_of_outliers'] = outlier['no_of_outliers'].apply(lambda val: (val/total)*100)
first_20_outlier = outlier.sort_values('percentage_of_outliers',ascending=False).head(20)

first_20_outlier
plt.figure(figsize=(18,12))

sns.barplot(first_20_outlier.percentage_of_outliers,first_20_outlier.column, orient='h')

plt.tight_layout()
plt.figure(figsize=(10,8))

sns.distplot(outlier.percentage_of_outliers,bins=10)
#from pprint import pprint

outlier[outlier.percentage_of_outliers>10].sort_values('percentage_of_outliers', ascending=False)
sns.countplot(data.HasDetections)
hsdf = pd.DataFrame(data.HasDetections.value_counts().values,

             columns=['count'])

hsdf['probability'] = data.HasDetections.value_counts(normalize=True)

hsdf
group = data.groupby('HasDetections')
HasDgroup = group.get_group(1)
detect_counts = HasDgroup.ProductName.value_counts()

total_counts = data.ProductName.value_counts()
fault_ratio_df = pd.DataFrame({'ProductName':detect_counts.index,'no_of_detection':detect_counts.values})

fault_ratio_df['total_count_of_product'] = total_counts.values

fault_ratio_df['probability'] = fault_ratio_df.no_of_detection/fault_ratio_df.total_count_of_product



fault_ratio_df.sort_values('probability',ascending=False)
detect_counts = HasDgroup.Processor.value_counts()

total_counts = df.Processor.value_counts()
fault_ratio_df = pd.DataFrame({'Processor':detect_counts.index,'no_of_detection':detect_counts.values})

fault_ratio_df['total_count_of_product'] = total_counts.values

fault_ratio_df['ratio'] = fault_ratio_df.no_of_detection/fault_ratio_df.total_count_of_product



fault_ratio_df.sort_values('ratio',ascending=False)
detect_counts = HasDgroup.Platform.value_counts()

total_counts = df.Platform.value_counts()
fault_ratio_df = pd.DataFrame({'Platform':detect_counts.index,'no_of_detection':detect_counts.values})

fault_ratio_df['total_count_of_product'] = total_counts.values

fault_ratio_df['ratio'] = fault_ratio_df.no_of_detection/fault_ratio_df.total_count_of_product



fault_ratio_df.sort_values('ratio',ascending=False)
HasDgroup.head()
plt.figure(figsize=(18,12))

sns.kdeplot(data.Census_SystemVolumeTotalCapacity, shade=True)
plt.figure(figsize=(16,12))

sns.FacetGrid(data,hue='HasDetections',height=16).map(sns.kdeplot,

                                                   'Census_SystemVolumeTotalCapacity',shade=True ).add_legend()
plt.figure(figsize=(18,12))

sns.kdeplot(data.Census_TotalPhysicalRAM,shade=True )
plt.figure(figsize=(16,12))

sns.FacetGrid(data,hue='HasDetections',height=16).map(sns.kdeplot,

                                                   'Census_TotalPhysicalRAM',shade=True ).add_legend()
plt.figure(figsize=(16,12))

sns.scatterplot(data.Census_SystemVolumeTotalCapacity,df.Census_TotalPhysicalRAM, hue=df.HasDetections)
data.head()
corr_matrix = data.corr()
most_corr = corr_matrix[corr_matrix.HasDetections.abs()>0.005]

cols = most_corr.index

most_corr = most_corr[cols]

most_corr
plt.figure(figsize=(14,8))

corrs = most_corr.HasDetections.values[:-1]

col = most_corr.HasDetections.index[:-1]

sns.barplot(corrs,col,label=corrs, palette='viridis')

corr_df = pd.DataFrame(corr_matrix.HasDetections.values,index=corr_matrix.HasDetections.index,

                       columns=['correlation'])



corr_df.sort_values('correlation',ascending=False).head(11).iloc[1:]
corr_df.sort_values('correlation',ascending=False).tail(11).iloc[:-1]
plt.figure(figsize=(20,14))

sns.heatmap(most_corr,cmap='winter')
#sns.countplot(df.AppVersion, hue=df.HasDetections)

data.AppVersion.nunique()
plt.figure(figsize=(14,8))

sns.countplot(data.Census_MDC2FormFactor,hue=data.HasDetections)

plt.tight_layout()
plt.figure(figsize=(18,12))

data.CountryIdentifier.value_counts()
data.head()
sns.countplot(data.IsProtected, hue=data.HasDetections)
sns.countplot(data.Firewall, hue=data.HasDetections)