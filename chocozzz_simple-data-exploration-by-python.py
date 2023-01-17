import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import gc

import os
import string
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

import time
dtypes = {
        'MachineIdentifier':                                    'object',
        'ProductName':                                          'object',
        'EngineVersion':                                        'object',
        'AppVersion':                                           'object',
        'AvSigVersion':                                         'object',
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
        'Platform':                                             'object',
        'Processor':                                            'object',
        'OsVer':                                                'object',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'object',
        'OsBuildLab':                                           'object',
        'SkuEdition':                                           'object',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'object',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'object',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'object',
        'Census_DeviceFamily':                                  'object',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'object',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'object',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'object',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'object',
        'Census_InternalBatteryType':                           'object',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'object',
        'Census_OSArchitecture':                                'object',
        'Census_OSBranch':                                      'object',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'object',
        'Census_OSSkuName':                                     'object',
        'Census_OSInstallTypeName':                             'object',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'object',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'object',
        'Census_ActivationChannel':                             'object',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'object',
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
%time train = pd.read_csv("../input/train.csv", dtype=dtypes)
%time test = pd.read_csv("../input/test.csv", dtype=dtypes)
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
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train.head()
train.columns
print(train.shape,test.shape)
temp = train["HasDetections"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (14,6))
plt.title('HasDetections 0 vs 1')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()
temp = train["ProductName"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (14,6))
plt.title('Count of ProductName')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()
temp
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
# from plotly import tools
# import plotly.tools as tls
# import squarify
# from mpl_toolkits.basemap import Basemap
# from numpy import array
# from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

# from sklearn import preprocessing
# # Supress unnecessary warnings so that presentation looks clean
# import warnings
# warnings.filterwarnings("ignore")

# # Print all rows and columns
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
temp = train["EngineVersion"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Engine Version')
temp = train["AppVersion"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of App Version')
temp = train["AvSigVersion"].value_counts()
temp
temp = train["IsBeta"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (14,6))
plt.title('IsBeta 0 vs 1')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()
temp = train["RtpStateBitfield"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of RtpStateBitfield')
temp = train["IsSxsPassiveMode"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of IsSxsPassiveMode')
temp = train["DefaultBrowsersIdentifier"].value_counts()
temp
temp = train["AVProductStatesIdentifier"].value_counts()
temp
temp = train["AVProductsInstalled"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of AVProductsInstalled')
temp = train["AVProductsInstalled"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of AVProductsInstalled')
temp = train["HasTpm"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of HasTpm')
#histogram
f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(train['CountryIdentifier'])
temp = train["OrganizationIdentifier"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of OrganizationIdentifier')
temp = train["GeoNameIdentifier"].value_counts()
temp
temp = train["LocaleEnglishNameIdentifier"].value_counts()
temp

temp = train["Platform"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Platform')
temp = train["Processor"].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Processor')
temp = train["OsVer"].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of OsVer')
temp = train["OsBuild"].value_counts()
temp
temp = train["OsSuite"].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of OsSuite')
temp = train['OsPlatformSubRelease'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of OsPlatformSubRelease')
temp = train['OsBuildLab'].value_counts()

temp
temp = train['SkuEdition'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of SkuEdition')
temp = train['IsProtected'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of IsProtected')
temp = train['AutoSampleOptIn'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of AutoSampleOptIn')
temp = train['PuaMode'].value_counts()
temp
temp = train['PuaMode'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of PuaMode')
temp = train['SMode'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of SMode')
temp = train['IeVerIdentifier'].value_counts()

temp
temp = train['SmartScreen'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of SmartScreen')
temp = train['Firewall'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Firewall')
temp = train['UacLuaenable'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of UacLuaenable')
temp = train['Census_MDC2FormFactor'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_MDC2FormFactor')
temp = train['Census_DeviceFamily'].value_counts()

df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_DeviceFamily')
temp = train['Census_OEMNameIdentifier'].value_counts()
temp
temp = train['Census_OEMModelIdentifier'].value_counts()
temp
temp = train['Census_ProcessorCoreCount'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_ProcessorCoreCount')
temp = train['Census_ProcessorManufacturerIdentifier'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_ProcessorManufacturerIdentifier')
temp = train['Census_ProcessorModelIdentifier'].value_counts()
temp
temp = train['Census_ProcessorClass'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_ProcessorClass')
train['Census_PrimaryDiskTotalCapacity'].describe()
temp = train['Census_PrimaryDiskTypeName'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_PrimaryDiskTypeName')
train['Census_SystemVolumeTotalCapacity'].describe()
temp = train['Census_HasOpticalDiskDrive'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_HasOpticalDiskDrive')
train['Census_TotalPhysicalRAM'].describe()
temp = train['Census_ChassisTypeName'].value_counts()
temp
train['Census_InternalPrimaryDiagonalDisplaySizeInInches'].describe()
train['Census_InternalPrimaryDisplayResolutionHorizontal'].describe()
train['Census_InternalPrimaryDisplayResolutionVertical'].describe()
temp = train['Census_PowerPlatformRoleName'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_PowerPlatformRoleName')
temp = train['Census_InternalBatteryType'].value_counts()
temp
train['Census_InternalBatteryNumberOfCharges'].describe()
temp = train['Census_OSVersion'].value_counts()
temp
temp = train['Census_OSArchitecture'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_OSArchitecture')
temp = train['Census_OSBranch'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_OSBranch')
temp = train['Census_OSBuildNumber'].value_counts()
temp
temp = train['Census_OSBuildRevision'].value_counts()
temp
temp = train['Census_OSEdition'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_OSEdition')
temp = train['Census_OSSkuName'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_OSSkuName')
temp = train['Census_OSInstallTypeName'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_OSInstallTypeName')
temp = train['Census_OSInstallLanguageIdentifier'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_OSInstallLanguageIdentifier')
temp = train['Census_OSUILocaleIdentifier'].value_counts()
temp
temp = train['Census_OSWUAutoUpdateOptionsName'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_OSWUAutoUpdateOptionsName')
temp = train['Census_IsPortableOperatingSystem'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsPortableOperatingSystem')
temp = train['Census_GenuineStateName'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_GenuineStateName')
temp = train['Census_ActivationChannel'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_ActivationChannel')
temp = train['Census_IsFlightingInternal'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsFlightingInternal')
temp = train['Census_IsFlightsDisabled'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsFlightsDisabled')
temp = train['Census_FlightRing'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_FlightRing')
temp = train['Census_ThresholdOptIn'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_ThresholdOptIn')
temp = train['Census_FirmwareManufacturerIdentifier'].value_counts()
temp
temp = train['Census_FirmwareVersionIdentifier'].value_counts()
temp
temp = train['Census_IsSecureBootEnabled'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsSecureBootEnabled')
temp = train['Census_IsWIMBootEnabled'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsWIMBootEnabled')
temp = train['Census_IsWIMBootEnabled'].value_counts()
temp
temp = train['Census_IsVirtualDevice'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsVirtualDevice')
temp = train['Census_IsTouchEnabled'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsTouchEnabled')
temp = train['Census_IsPenCapable'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsPenCapable')
temp = train['Census_IsPenCapable'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsPenCapable')
temp = train['Census_IsAlwaysOnAlwaysConnectedCapable'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Census_IsAlwaysOnAlwaysConnectedCapable')
temp = train['Wdft_IsGamer'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Wdft_IsGamer')
temp = train['Wdft_RegionIdentifier'].value_counts()


df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Count of Wdft_RegionIdentifier')
temp = train["EngineVersion"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["EngineVersion"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["EngineVersion"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Type of EngineVersion is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["EngineVersion"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["EngineVersion"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["EngineVersion"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Type of EngineVersion is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["AppVersion"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["AppVersion"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["AppVersion"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Type of AppVersion is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["RtpStateBitfield"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["RtpStateBitfield"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["RtpStateBitfield"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of RtpStateBitfield is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["IsSxsPassiveMode"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["IsSxsPassiveMode"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["IsSxsPassiveMode"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of IsSxsPassiveMode is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["AVProductsInstalled"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["AVProductsInstalled"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["AVProductsInstalled"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of AVProductsInstalled is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["OrganizationIdentifier"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["OrganizationIdentifier"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["OrganizationIdentifier"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of OrganizationIdentifier is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Platform"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Platform"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Platform"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Platform is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Processor"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Processor"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Processor"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Processor is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["OsVer"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["OsVer"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["OsVer"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of OsVer is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["OsSuite"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["OsSuite"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["OsSuite"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of OsSuite is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["OsPlatformSubRelease"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["OsPlatformSubRelease"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["OsPlatformSubRelease"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Type of OsPlatformSubRelease is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["SkuEdition"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["SkuEdition"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["SkuEdition"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Type of SkuEdition is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["IsProtected"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["IsProtected"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["IsProtected"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of IsProtected is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["SmartScreen"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["SmartScreen"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["SmartScreen"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of SmartScreen is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["UacLuaenable"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["UacLuaenable"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["UacLuaenable"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of UacLuaenable is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_MDC2FormFactor"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_MDC2FormFactor"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_MDC2FormFactor"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_MDC2FormFactor is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_DeviceFamily"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_DeviceFamily"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_DeviceFamily"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_DeviceFamily is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_ProcessorCoreCount"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_ProcessorCoreCount"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_ProcessorCoreCount"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of Census_ProcessorCoreCount is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_ProcessorManufacturerIdentifier"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_ProcessorManufacturerIdentifier"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_ProcessorManufacturerIdentifier"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of Census_ProcessorManufacturerIdentifier is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_ProcessorClass"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_ProcessorClass"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_ProcessorClass"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_ProcessorClass is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_PrimaryDiskTypeName"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_PrimaryDiskTypeName"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_PrimaryDiskTypeName"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_PrimaryDiskTypeName is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_HasOpticalDiskDrive"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_HasOpticalDiskDrive"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_HasOpticalDiskDrive"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of Census_HasOpticalDiskDrive is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_PowerPlatformRoleName"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_PowerPlatformRoleName"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_PowerPlatformRoleName"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Values of Census_PowerPlatformRoleName is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_OSArchitecture"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_OSArchitecture"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_OSArchitecture"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_OSArchitecture is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_OSBranch"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_OSBranch"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_OSBranch"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_OSBranch is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_OSEdition"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_OSEdition"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_OSEdition"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_OSEdition is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_OSSkuName"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_OSSkuName"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_OSSkuName"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_OSSkuName is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_OSInstallTypeName"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_OSInstallTypeName"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_OSInstallTypeName"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_OSInstallTypeName is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_OSWUAutoUpdateOptionsName"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_OSWUAutoUpdateOptionsName"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_OSWUAutoUpdateOptionsName"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_OSWUAutoUpdateOptionsName is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_GenuineStateName"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_GenuineStateName"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_GenuineStateName"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_GenuineStateName is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_ActivationChannel"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_ActivationChannel"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_ActivationChannel"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_ActivationChannel is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
temp = train["Census_FlightRing"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(train["HasDetections"][train["Census_FlightRing"]==val] == 1))
    temp_y0.append(np.sum(train["HasDetections"][train["Census_FlightRing"]==val] == 0))    
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='Detected'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100, 
    name='Not Detected'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Types of Census_FlightRing is HasDetections or not in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)