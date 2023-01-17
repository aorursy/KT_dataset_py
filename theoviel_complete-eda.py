import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import operator 

sns.set_style('whitegrid')
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
        'HasDetections':                                        'int8',}
train_df = pd.read_csv('../input/train.csv', dtype=dtypes, nrows=1000000)
train_df.info()
train_df.head()
plt.figure(figsize=(15,10))
sns.countplot(x='ProductName', hue='HasDetections', data=train_df)
plt.yscale('log')
plt.title('Which Defenders are infected', size=15)
plt.show()
df = train_df[train_df["IsBeta"] == 1]

plt.figure(figsize=(15,10))
sns.countplot(x='ProductName', hue='HasDetections', data=df)
plt.title('Beta defenders', size=15)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='RtpStateBitfield', hue='HasDetections', data=train_df)
plt.yscale('log')
plt.title('RtpStateBitfield influence', size=15)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='IsSxsPassiveMode', hue='HasDetections', data=train_df)
plt.yscale('log')
plt.title('Which Defenders are infected', size=15)
plt.show()
print("Number of browser ids :", len(df['DefaultBrowsersIdentifier'].unique()))
print("Number of antivirus ids :", len(df['AVProductStatesIdentifier'].unique()))
plt.figure(figsize=(15,10))
sns.countplot(x='AVProductsInstalled', hue='HasDetections', data=train_df)
plt.title('The more AntiVirus installed...', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='AVProductsEnabled', hue='HasDetections', data=train_df)
plt.title('Enable your AntiVirus ?', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='IsProtected', hue='HasDetections', data=train_df)
plt.title('Protected devices', size=15)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='HasTpm', hue='HasDetections', data=train_df)
plt.title('TPM influence', size=15)
plt.yscale('log')
plt.show()
print("Number of country ids :", len(train_df['CountryIdentifier'].unique()))
print("Number of city ids :", len(train_df['CityIdentifier'].unique()))
print("Number of organization ids :", len(train_df['OrganizationIdentifier'].unique()))
print("Number of region ids :", len(train_df['GeoNameIdentifier'].unique()))
print("Number of locale english name ids :", len(train_df['LocaleEnglishNameIdentifier'].unique()))
print("Number of region ids (wdft) :", len(train_df['Wdft_RegionIdentifier'].unique()))
ratios = {}

for c in train_df['CountryIdentifier'].unique():
    df = train_df[train_df['CountryIdentifier'] == c]
    ratios[c] = sum(df['HasDetections']) / len(df)
data = pd.DataFrame({"Country Id": list(ratios.keys()), "Infection Ratio": list(ratios.values())}).sample(50).sort_values(by='Infection Ratio')
order = list(data['Country Id'])[::-1]

plt.figure(figsize=(15,10))
sns.barplot(x="Country Id", y="Infection Ratio", data=data, order=order)
plt.title('Proportion of infected samples for some countries', size=15)
plt.xticks(rotation=-45)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Platform', hue='HasDetections', data=train_df)
plt.title('Impact of the Operation System', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Census_DeviceFamily', hue='HasDetections', data=train_df)
plt.title('UacLua Influence', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Platform', hue='Processor', data=train_df)
plt.title('Architectures of the different OS', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Processor', hue='HasDetections', data=train_df)
plt.title('x64 vs x86', size=15)
plt.yscale('log')
plt.show()
print("Number of os versions :", len(train_df['OsVer'].unique()))
print("Number of os builds :", len(train_df['OsBuild'].unique()))
print("Number of os suites :", len(train_df['OsSuite'].unique()))
print("Number of os platform subreleases :", len(train_df['OsPlatformSubRelease'].unique()))
print("Number of os build labs :", len(train_df['OsBuildLab'].unique()))
order = ['10.0.0.0', '10.0.0.1', '10.0.1.0', '10.0.1.44', '10.0.2.0', '10.0.21.0', '10.0.3.0', 
         '10.0.3.80', '10.0.32.0', '10.0.32.72', '10.0.4.0', '10.0.5.0', '10.0.5.18', '10.0.7.0', '10.0.80.0', 
         '6.1.0.0', '6.1.1.0', '6.1.3.0', '6.3.0.0', '6.3.1.0', '6.3.3.0', '6.3.4.0']

plt.figure(figsize=(15,10))
sns.countplot(x='OsVer', hue='Platform', data=train_df, order=order)
plt.title('Version of the OS', size=15)
plt.yscale('log')
plt.xticks(rotation=-45)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='OsSuite', hue='Platform', data=train_df)
plt.title('Suites of different OS', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='OsSuite', hue='HasDetections', data=train_df)
plt.title('Influence of the Suite', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='OsPlatformSubRelease', hue='Platform', data=train_df)
plt.title('Subreleases for different OS', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='OsPlatformSubRelease', hue='HasDetections', data=train_df)
plt.title('Influence of the os subreleases', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='SkuEdition', hue='HasDetections', data=train_df)
plt.title('Different SKUs', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='AutoSampleOptIn', hue='HasDetections', data=train_df)
plt.title('Influence of auto sample', size=15)
plt.yscale('log')
plt.show()
train_df['PuaMode'] = train_df['PuaMode'].cat.add_categories(['off'])
train_df['PuaMode'] = train_df[['PuaMode']].fillna('off')
plt.figure(figsize=(15,10))
sns.countplot(x='PuaMode', hue='HasDetections', data=train_df)
plt.title('Pua Mode influence', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='SMode', hue='HasDetections', data=train_df)
plt.title('Store Mode influence', size=15)
plt.yscale('log')
plt.show()
print("Number of ie versions :", len(train_df['IeVerIdentifier'].unique()))
plt.figure(figsize=(15,10))
sns.countplot(x='SmartScreen', hue='HasDetections', data=train_df)
plt.title('Smart Screen influence', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Firewall', hue='HasDetections', data=train_df)
plt.title('Firewall Influence', size=15)
plt.yscale('log')
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='UacLuaenable', hue='HasDetections', data=train_df)
plt.title('UacLua Influence', size=15)
plt.yscale('log')
plt.show()
print("Number of names ids :", len(train_df['Census_OEMNameIdentifier'].unique()))
print("Number of model ids :", len(train_df['Census_OEMModelIdentifier'].unique()))
plt.figure(figsize=(15,10))
sns.countplot(x='Census_MDC2FormFactor', hue='HasDetections', data=train_df)
plt.title('UacLua Influence', size=15)
plt.yscale('log')
plt.show()
print("Number of processor manufacturer ids :", len(train_df['Census_ProcessorManufacturerIdentifier'].unique()))
print("Number of processor model ids :", len(train_df['Census_ProcessorModelIdentifier'].unique()))
plt.figure(figsize=(15,10))
sns.countplot(x='Census_ProcessorClass', hue='HasDetections', data=train_df, order=['low', 'mid', 'high'])
plt.title('Different precessor tiers', size=15)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Census_PrimaryDiskTypeName', hue='HasDetections', data=train_df)
plt.title('Type of primary disk influence', size=15)
plt.show()
ratios = {}
volumes = [50000, 100000, 500000, 1000000, 1000000000]

for i, vol in enumerate(volumes[1:]):
    df = train_df[train_df['Census_PrimaryDiskTotalCapacity'] <= vol]
    df = df[df['Census_PrimaryDiskTotalCapacity'] >= volumes[i]]
    ratios[vol] = sum(df['HasDetections']) / len(df)
    
data = pd.DataFrame({"Disk Volume": list(ratios.keys()), "Infection Ratio": list(ratios.values())}).sort_values(by="Disk Volume")
plt.figure(figsize=(15,10))
sns.barplot(x="Disk Volume", y="Infection Ratio", data=data)
plt.title('Proportion of infected samples for different disk sizes', size=15)
plt.xticks(range(0, 4), ["<100 000", "[100 000; 500 000]", "[500 000; 1 000 000]", ">1 000 000"], rotation=-85)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Census_HasOpticalDiskDrive', hue='HasDetections', data=train_df)
plt.title('Type of primary disk influence', size=15)
plt.show()
ratios = {}
volumes = [0, 2000, 4000, 6000, 8000, 12000, 16000, 1000000]

for i, vol in enumerate(volumes[1:]):
    df = train_df[train_df['Census_TotalPhysicalRAM'] <= vol]
    df = df[df['Census_TotalPhysicalRAM'] >= volumes[i]]
    ratios[vol] = sum(df['HasDetections']) / len(df)
    
data = pd.DataFrame({"ram": list(ratios.keys()), "Infection Ratio": list(ratios.values())}).sort_values(by="ram")
plt.figure(figsize=(15,10))
sns.barplot(x="ram", y="Infection Ratio", data=data)
plt.title('Proportion of infected samples for different RAM sizes', size=15)
plt.xticks(range(0, 8), ["<2Gb", "[2Gb; 4Gb]", "[4Gb; 6Gb]", "[6Gb; 8Gb]", "[8Gb; 12Gb]", "[12Gb; 16Gb]", "> 16Gb",], rotation=-85)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(x='Wdft_IsGamer', hue='HasDetections', data=train_df)
plt.title("Gamers' Malwares", size=15)
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(train_df['HasDetections'])
plt.show()