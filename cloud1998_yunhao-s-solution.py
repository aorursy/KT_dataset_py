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
train_df.replace({'OrganizationIdentifier': {np.nan: 0}}, inplace=True)
train_df.OrganizationIdentifier.isnull().sum()
train_df.shape
train_df.dropna(inplace=True)

train_df.shape
cols = train_df.columns.tolist()

plt.figure(figsize=(30,30))

sns.heatmap(train_df[cols].corr().abs(), cmap='RdBu_r', annot=True, center=0.0)

plt.show()
'EngineVersion' in train_df
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

print(get_top_abs_correlations(corr_matrix, 10))
train_df.shape
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
train_df.info()
import math



# CHECK FOR NAN

def nan_check(x):

    if isinstance(x,float):

        if math.isnan(x):

            return True

    return False



# FREQUENCY ENCODING

def encode_FE(df,col,verbose=1):

    d = df[col].value_counts(dropna=False)

    n = col+"_FE"

    df[n] = df[col].map(d)/d.max()

    if verbose==1:

        print('FE encoded',col)

    return [n]



# ONE-HOT-ENCODE ALL CATEGORY VALUES THAT COMPRISE MORE THAN

# "FILTER" PERCENT OF TOTAL DATA AND HAS SIGNIFICANCE GREATER THAN "ZVALUE"

def encode_OHE(df, col, filter, zvalue, tar='HasDetections', m=0.5, verbose=1):

    cv = df[col].value_counts(dropna=False)

    cvd = cv.to_dict()

    vals = len(cv)

    th = filter * len(df)

    sd = zvalue * 0.5/ math.sqrt(th)

    #print(sd)

    n = []; ct = 0; d = {}

    for x in cv.index:

        try:

            if cv[x]<th: break

            sd = zvalue * 0.5/ math.sqrt(cv[x])

        except:

            if cvd[x]<th: break

            sd = zvalue * 0.5/ math.sqrt(cvd[x])

        if nan_check(x): r = df[df[col].isna()][tar].mean()

        else: r = df[df[col]==x][tar].mean()

        if abs(r-m)>sd:

            nm = col+'_BE_'+str(x)

            if nan_check(x): df[nm] = (df[col].isna()).astype('int8')

            else: df[nm] = (df[col]==x).astype('int8')

            n.append(nm)

            d[x] = 1

        ct += 1

        if (ct+1)>=vals: break

    if verbose==1:

        print('OHE encoded',col,'- Created',len(d),'booleans')

    return [n,d]



# ONE-HOT-ENCODING from dictionary

def encode_OHE_test(df,col,dt):

    n = []

    for x in dt: 

        n += encode_BE(df,col,x)

    return n



# BOOLEAN ENCODING

def encode_BE(df,col,val):

    n = col+"_BE_"+str(val)

    if nan_check(val):

        df[n] = df[col].isna()

    else:

        df[n] = df[col]==val

    df[n] = df[n].astype('int8')

    return [n]
# LOAD AND FREQUENCY-ENCODE

FE = ['EngineVersion','AppVersion','AvSigVersion','Census_OSVersion']

# LOAD AND ONE-HOT-ENCODE

OHE = [ 'OsPlatformSubRelease','OsBuildLab','SkuEdition','SmartScreen','Census_MDC2FormFactor','Census_PrimaryDiskTypeName','Census_ChassisTypeName','Census_PowerPlatformRoleName',

      'Census_InternalBatteryType','Census_OSBranch','Census_OSEdition','Census_OSSkuName','Census_OSInstallTypeName','Census_OSWUAutoUpdateOptionsName','Census_GenuineStateName',

      'Census_ActivationChannel']
samples = train_df.sample(50000,random_state=42)
cols = []; dd = []



# ENCODE NEW

for x in FE:

    cols += encode_FE(samples,x)

for x in OHE:

    tmp = encode_OHE(samples,x,0.005,5)

    cols += tmp[0]; dd.append(tmp[1])

print('Encoded',len(cols),'new variables')



# REMOVE OLD

for x in FE+OHE:

    del samples[x]

print('Removed original',len(FE+OHE),'variables')
samples
from sklearn.model_selection import KFold
kf = KFold(n_splits=5,random_state=5,shuffle=True)
samples = samples.reset_index().drop(columns='index')
import xgboost as xgb

from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score
samples
auc = []

for train_index,test_index in kf.split(samples):

    X_train = samples.iloc[train_index].drop(columns='HasDetections')    

    y_train = samples.iloc[train_index].HasDetections

    X_test = samples.iloc[test_index].drop(columns='HasDetections')    

    y_test = samples.iloc[test_index].HasDetections        

    xgb = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=180, silent=True)

    xgb.fit(X_train,y_train)

    predictions = xgb.predict(X_test)

    auc.append(roc_auc_score(y_test,predictions))

print (auc)
print(np.mean(auc))