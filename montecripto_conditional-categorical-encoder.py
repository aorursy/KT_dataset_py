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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from collections import Counter,defaultdict
dtypes = {
    'MachineIdentifier': 'object',
    'ProductName': 'category',
    'EngineVersion': 'category',
    'AppVersion': 'category',
    'AvSigVersion': 'category',
    'IsBeta': 'int8',
    'RtpStateBitfield': 'float16',
    'IsSxsPassiveMode': 'int8',
    'DefaultBrowsersIdentifier': 'float16',
    'AVProductStatesIdentifier': 'float32',
    'AVProductsInstalled': 'float16',
    'AVProductsEnabled': 'float16',
    'HasTpm': 'int8',
    'CountryIdentifier': 'int16',
    'CityIdentifier': 'float32',
    'OrganizationIdentifier': 'float16',
    'GeoNameIdentifier': 'float16',
    'LocaleEnglishNameIdentifier': 'int8',
    'Platform': 'category',
    'Processor': 'category',
    'OsVer': 'category',
    'OsBuild': 'int16',
    'OsSuite': 'int16',
    'OsPlatformSubRelease': 'category',
    'OsBuildLab': 'category',
    'SkuEdition': 'category',
    'IsProtected': 'float16',
    'AutoSampleOptIn': 'int8',
    'PuaMode': 'category',
    'SMode': 'float16',
    'IeVerIdentifier': 'float16',
    'SmartScreen': 'category',
    'Firewall': 'float16',
    'UacLuaenable': 'float32',
    'Census_MDC2FormFactor': 'category',
    'Census_DeviceFamily': 'category',
    'Census_OEMNameIdentifier': 'float16',
    'Census_OEMModelIdentifier': 'float32',
    'Census_ProcessorCoreCount': 'float16',
    'Census_ProcessorManufacturerIdentifier': 'float16',
    'Census_ProcessorModelIdentifier': 'float16',
    'Census_ProcessorClass': 'category',
    'Census_PrimaryDiskTotalCapacity': 'float32',
    'Census_PrimaryDiskTypeName': 'category',
    'Census_SystemVolumeTotalCapacity': 'float32',
    'Census_HasOpticalDiskDrive': 'int8',
    'Census_TotalPhysicalRAM': 'float32',
    'Census_ChassisTypeName': 'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float16',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'float16',
    'Census_InternalPrimaryDisplayResolutionVertical': 'float16',
    'Census_PowerPlatformRoleName': 'category',
    'Census_InternalBatteryType': 'category',
    'Census_InternalBatteryNumberOfCharges': 'float32',
    'Census_OSVersion': 'category',
    'Census_OSArchitecture': 'category',
    'Census_OSBranch': 'category',
    'Census_OSBuildNumber': 'int16',
    'Census_OSBuildRevision': 'int32',
    'Census_OSEdition': 'category',
    'Census_OSSkuName': 'category',
    'Census_OSInstallTypeName': 'category',
    'Census_OSInstallLanguageIdentifier': 'float16',
    'Census_OSUILocaleIdentifier': 'int16',
    'Census_OSWUAutoUpdateOptionsName': 'category',
    'Census_IsPortableOperatingSystem': 'int8',
    'Census_GenuineStateName': 'category',
    'Census_ActivationChannel': 'category',
    'Census_IsFlightingInternal': 'float16',
    'Census_IsFlightsDisabled': 'float16',
    'Census_FlightRing': 'category',
    'Census_ThresholdOptIn': 'float16',
    'Census_FirmwareManufacturerIdentifier': 'float16',
    'Census_FirmwareVersionIdentifier': 'float32',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsWIMBootEnabled': 'float16',
    'Census_IsVirtualDevice': 'float16',
    'Census_IsTouchEnabled': 'int8',
    'Census_IsPenCapable': 'int8',
    'Census_IsAlwaysOnAlwaysConnectedCapable': 'float16',
    'Wdft_IsGamer': 'float16',
    'Wdft_RegionIdentifier': 'float16',
    'HasDetections': 'float32'
}
class ConditionalCategoricalEnconder:
    def __init__(self):
        self.data_dict=dict()
        self.categorical_columns = None
        self.df_input = None
        self.target = None        
    
    def fit_column(self,col):
        df = pd.DataFrame(self.df_input[col])
        
        #base count P(X = x)
        counts = Counter(df[col])        
        df[col+"_count"] = df[col].map(counts)        
        
        
        # joint count  P(Traget=1 and X = x )
        counts_yes = Counter(df[self.target==1][col])        
        df[col+"_count_yes"] = df[col].map(counts_yes)                                           
        
        # conditional probability  P(Target=Yes | X=x) = P(Traget=1 and X = x ) / P(X = x)
        df[col+"_prob"] = (df[col+"_count_yes"].values.astype(np.float32) / df[col+"_count"].values.astype(np.float32))        
        
        col_dict = defaultdict(lambda: 0) # if in test set there is not match then p=0
        col_dict.update({k:v for k,v in zip(df[col].values,df[col+"_prob"])})        
        self.data_dict[col]=col_dict
        
    
    def fit(self,X,y,categorical_columns):
        self.data_dict=dict()
        self.categorical_columns = categorical_columns
        self.df_input = X
        self.target = y
        
        for col in tqdm(categorical_columns,desc='Fitting columns...'):        
            self.fit_column(col);
        
        self.df_input = None
        self.target = None
        return self
    
    
    def transform_col(self,col):
        self.df_input[col] = pd.to_numeric(self.df_input[col].map(self.data_dict[col])).astype(np.float16)
    
    def transform(self,X,drop_cols = False):
        self.df_input = X.copy(deep=True)
        for col in tqdm(self.categorical_columns,desc='Transforming columns...'):        
            self.transform_col(col);
        
        if drop_cols:
            self.df_input = self.df_input.drop(categorical_columns, axis=1)
        
        return self.df_input    
ID = 'MachineIdentifier'
TARGET = 'HasDetections'

train_features = list(dtypes.keys())
train_features.remove(ID)
train_features.remove(TARGET)
nrows = 100000
train_loader = pd.read_csv('../input/train.csv', dtype=dtypes,nrows =nrows)
test_loader = pd.read_csv('../input/test.csv', dtype=dtypes,nrows =nrows)
train = train_loader.drop([ID, TARGET], axis=1)
train_labels = train_loader[TARGET].values
train_ids = train_loader[ID].values
print('\n Shape of raw train data:', train.shape)
[col for col,val in train.dtypes.to_dict().items() if val.name=='category']
test = test_loader.drop([ID], axis=1)
test_ids = test_loader[ID].values
print(' Shape of raw test data:', test.shape)
[col for col,val in test.dtypes.to_dict().items() if val.name=='category']
categorical_columns = [col for col,val in test.dtypes.to_dict().items() if val.name=='category']
ec = ConditionalCategoricalEnconder()
ec = ec.fit(X=train,y=train_labels,categorical_columns=categorical_columns)
train = ec.transform(train)
test = ec.transform(test)
train[categorical_columns].head()
test[categorical_columns].head()