import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/train.csv',nrows=500000,low_memory=True)
# Not the complete dataset, but the number ratios should be representative of all the data
df_train.head()
df_train = df_train.select_dtypes('object')
df_train.shape
df_train.columns
df_train['MachineIdentifier'][:5] # Can we glean info from this?
print(df_train['ProductName'].isna().sum(),df_train['EngineVersion'].isna().sum(),\
      df_train['AppVersion'].isna().sum(),df_train['AvSigVersion'].isna().sum())
print(len(df_train['ProductName'].value_counts().unique()),len(df_train['EngineVersion'].value_counts().unique()),\
     len(df_train['AppVersion'].value_counts().unique()),len(df_train['AvSigVersion'].value_counts().unique()))
df_train['ProductName'].value_counts()
df_train['EngineVersion'].value_counts()[:10]
df_train['AppVersion'].value_counts()[:10]
df_train['AvSigVersion'].value_counts()[:10]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,20))
sns.countplot(x = 'ProductName',
              data = df_train,
              order = df_train['ProductName'].value_counts().index,ax=ax1)
sns.countplot(x = 'EngineVersion',
              data = df_train,
              order = df_train['EngineVersion'].value_counts().iloc[:5].index,ax=ax2)
sns.countplot(x = 'AppVersion',
              data = df_train,
              order = df_train['AppVersion'].value_counts().iloc[:5].index,ax=ax3)
sns.countplot(x = 'AvSigVersion',
              data = df_train,
              order = df_train['AvSigVersion'].value_counts().iloc[:5].index,ax=ax4);
df_train['Platform'].value_counts() # Huh, whatever happened to Windows 95
df_train['Processor'].value_counts()
df_train['OsVer'].value_counts()[:10]
df_train['OsPlatformSubRelease'].value_counts()
df_train['OsBuildLab'].value_counts()[:10]
df_train['SkuEdition'].value_counts()
df_train['PuaMode'].value_counts() # The rest are probably nans => PuaMode = off
df_train['PuaMode'].isna().sum()
df_train['SmartScreen'].value_counts()
df_train['SmartScreen'].isna().sum() # Perhaps we should impute "ExistsNotSet" for these
print(df_train['Platform'].isna().sum(),df_train['Processor'].isna().sum(),df_train['SkuEdition'].isna().sum())
print(df_train['OsPlatformSubRelease'].isna().sum(),df_train['OsBuildLab'].isna().sum(),df_train['OsVer'].isna().sum())
df_train[df_train['OsBuildLab'].isnull()]
print(len(df_train['Platform'].value_counts().unique()),len(df_train['Processor'].value_counts().unique()),\
     len(df_train['OsVer'].value_counts().unique()),len(df_train['OsPlatformSubRelease'].value_counts().unique()),\
     len(df_train['OsBuildLab'].value_counts().unique()),len(df_train['SkuEdition'].value_counts().unique()),\
     len(df_train['PuaMode'].value_counts().unique()),len(df_train['SmartScreen'].value_counts().unique()))
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, figsize=(20,20))
sns.countplot(x = 'Platform',
              data = df_train,
              order = df_train['Platform'].value_counts().index,ax=ax1)
sns.countplot(x = 'Processor',
              data = df_train,
              order = df_train['Processor'].value_counts().index,ax=ax2)
sns.countplot(x = 'OsVer',
              data = df_train,
              order = df_train['OsVer'].value_counts().iloc[:5].index,ax=ax3)
sns.countplot(x = 'OsPlatformSubRelease',
              data = df_train,
              order = df_train['OsPlatformSubRelease'].value_counts().iloc[:5].index,ax=ax4);
sns.countplot(x = 'OsBuildLab',
              data = df_train,
              order = df_train['OsBuildLab'].value_counts().iloc[:3].index,ax=ax5);
sns.countplot(x = 'SkuEdition',
              data = df_train,
              order = df_train['SkuEdition'].value_counts().iloc[:5].index,ax=ax6);
ax5.set_xticklabels(['17134.1.amd64fre','16299.431.amd64fre','16299.15.amd64fre'], rotation=-30);
sns.countplot(x = 'PuaMode',
              data = df_train,
              order = df_train['PuaMode'].value_counts().index,ax=ax7);
sns.countplot(x = 'SmartScreen',
              data = df_train,
              order = df_train['SmartScreen'].value_counts().iloc[:5].index,ax=ax8);
ax8.set_xticklabels( df_train['SmartScreen'].value_counts().iloc[:5].index,rotation=-30);
df_train['Census_MDC2FormFactor'].value_counts()
df_train['Census_DeviceFamily'].value_counts()
df_train['Census_ProcessorClass'].value_counts() # So lots of nans -- probably for the newer models
df_train['Census_PrimaryDiskTypeName'].value_counts() # Unknown and Unspecified are probably the same?
df_train['Census_ChassisTypeName'].value_counts()[:10]
df_train['Census_PowerPlatformRoleName'].value_counts()
df_train['Census_InternalBatteryType'].value_counts()[:10]
df_train['Census_OSVersion'].value_counts()[:5]
cols = ['Census_MDC2FormFactor', 'Census_DeviceFamily',
       'Census_ProcessorClass', 'Census_PrimaryDiskTypeName',
       'Census_ChassisTypeName', 'Census_PowerPlatformRoleName',
       'Census_InternalBatteryType', 'Census_OSVersion']
for col in cols:
    print(col, "::::: NAN values: {}".format(df_train[col].isna().sum()),'::::: Unique vals: {}'.format(len(df_train[col].value_counts().unique())))
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, figsize=(20,20))
sns.countplot(x = 'Census_MDC2FormFactor',
              data = df_train,
              order = df_train['Census_MDC2FormFactor'].value_counts().iloc[:5].index,ax=ax1)
ax1.set_xticklabels(df_train['Census_MDC2FormFactor'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_DeviceFamily',
              data = df_train,
              order = df_train['Census_DeviceFamily'].value_counts().index,ax=ax2)
sns.countplot(x = 'Census_ProcessorClass',
              data = df_train,
              order = df_train['Census_ProcessorClass'].value_counts().iloc[:5].index,ax=ax3)
sns.countplot(x = 'Census_PrimaryDiskTypeName',
              data = df_train,
              order = df_train['Census_PrimaryDiskTypeName'].value_counts().iloc[:5].index,ax=ax4);
sns.countplot(x = 'Census_ChassisTypeName',
              data = df_train,
              order = df_train['Census_ChassisTypeName'].value_counts().iloc[:5].index,ax=ax5);
sns.countplot(x = 'Census_PowerPlatformRoleName',
              data = df_train,
              order = df_train['Census_PowerPlatformRoleName'].value_counts().iloc[:5].index,ax=ax6);
ax6.set_xticklabels(df_train['Census_PowerPlatformRoleName'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_InternalBatteryType',
              data = df_train,
              order = df_train['Census_InternalBatteryType'].value_counts().iloc[:5].index,ax=ax7);
sns.countplot(x = 'Census_OSVersion',
              data = df_train,
              order = df_train['Census_OSVersion'].value_counts().iloc[:5].index,ax=ax8);
ax8.set_xticklabels( df_train['Census_OSVersion'].value_counts().iloc[:5].index,rotation=-30);
df_train['Census_OSArchitecture'].value_counts()
df_train['Census_OSBranch'].value_counts()[:5]
df_train['Census_OSEdition'].value_counts()[:5]
df_train['Census_OSSkuName'].value_counts()[:5]
df_train['Census_OSInstallTypeName'].value_counts()
df_train['Census_OSWUAutoUpdateOptionsName'].value_counts()
df_train['Census_GenuineStateName'].value_counts() # Probably important in predicting Malware infection probability
df_train['Census_ActivationChannel'].value_counts()
df_train['Census_FlightRing'].value_counts()
cols = ['Census_OSArchitecture', 'Census_OSBranch', 'Census_OSEdition',
       'Census_OSSkuName', 'Census_OSInstallTypeName',
       'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName',
       'Census_ActivationChannel', 'Census_FlightRing']
for col in cols:
    print(col, "::::: NAN values: {}".format(df_train[col].isna().sum()),'::::: Unique vals: {}'.format(len(df_train[col].value_counts().unique())))
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(20,20))
sns.countplot(x = 'Census_OSArchitecture',
              data = df_train,
              order = df_train['Census_OSArchitecture'].value_counts().iloc[:5].index,ax=ax1)
# ax1.set_xticklabels(df_train['Census_MDC2FormFacto'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSBranch',
              data = df_train,
              order = df_train['Census_OSBranch'].value_counts().iloc[:5].index,ax=ax2)
ax2.set_xticklabels(df_train['Census_OSBranch'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSEdition',
              data = df_train,
              order = df_train['Census_OSEdition'].value_counts().iloc[:5].index,ax=ax3)
ax3.set_xticklabels(df_train['Census_OSEdition'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSSkuName',
              data = df_train,
              order = df_train['Census_OSSkuName'].value_counts().iloc[:5].index,ax=ax4);
ax4.set_xticklabels(df_train['Census_OSSkuName'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_OSInstallTypeName',
              data = df_train,
              order = df_train['Census_OSInstallTypeName'].value_counts().iloc[:5].index,ax=ax5);
sns.countplot(x = 'Census_OSWUAutoUpdateOptionsName',
              data = df_train,
              order = df_train['Census_OSWUAutoUpdateOptionsName'].value_counts().iloc[:5].index,ax=ax6);
ax6.set_xticklabels(df_train['Census_OSWUAutoUpdateOptionsName'].value_counts().iloc[:5], rotation=-30);
sns.countplot(x = 'Census_GenuineStateName',
              data = df_train,
              order = df_train['Census_GenuineStateName'].value_counts().iloc[:5].index,ax=ax7);
sns.countplot(x = 'Census_ActivationChannel',
              data = df_train,
              order = df_train['Census_ActivationChannel'].value_counts().iloc[:6].index,ax=ax8);
ax8.set_xticklabels( df_train['Census_ActivationChannel'].value_counts().iloc[:6].index,rotation=-30);
sns.countplot(x = 'Census_FlightRing',
              data = df_train,
              order = df_train['Census_FlightRing'].value_counts().iloc[:5].index,ax=ax9);
