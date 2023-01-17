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
!pip install git+https://github.com/aerdem4/lofo-importance


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

%time

train_df = pd.read_csv('../input/train.csv',nrows=1000000,dtype=dtypes,low_memory=False)

print('Loaded traininig data..')
train_df.head()
from sklearn.model_selection import KFold



sample_df = train_df.sample(frac=0.01, random_state=0)

sample_df.sort_values("AvSigVersion", inplace=True)



cv = KFold(n_splits=4, shuffle=False, random_state=0)



sample_df.shape
from lofo.lofo_importance import LOFOImportance, plot_importance



target = "HasDetections"

features = [col for col in train_df.columns if col != target]



lofo = LOFOImportance(sample_df, features, target, cv=cv, scoring="roc_auc")



importance_df = lofo.get_importance()

importance_df.head()
importance_df.tail()
%matplotlib inline



plot_importance(importance_df, figsize=(12, 20))
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
%time

total = train_df.isnull().sum().sort_values(ascending = False)

percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data.head()
missing_top_10= missing_train_data.nlargest(10,'Percent') 
plt.figure(figsize = (15,10))

sns.barplot(x=missing_top_10.index,y='Percent',data= missing_top_10)

plt.xticks(rotation=90)

plt.title("Top 10 Missing Features")

plt.show()
y=train_df['HasDetections']

X_train= train_df.drop(['HasDetections'],axis=1)
X_train= train_df.drop(['MachineIdentifier'],axis=1)

X_train.head()
X_train[X_train.select_dtypes(['object']).columns] = X_train.select_dtypes(['object']).apply(lambda x: x.astype('category'))
X_train.dtypes
X_train[X_train.select_dtypes(['category']).columns] = X_train.select_dtypes(['category']).apply(lambda x: x.cat.codes)
importance_df
to_keep_cols = X_train[["SmartScreen","OsBuildLab","Census_ChassisTypeName","AVProductsInstalled","AppVersion","Wdft_IsGamer","AvSigVersion",

"Processor","Census_OSInstallLanguageIdentifier","Census_OSEdition",

                       'Census_InternalBatteryNumberOfCharges',"Census_OSBuildNumber","Census_FirmwareVersionIdentifier","Census_HasOpticalDiskDrive"

                       ]]
to_keep_cols['HasDetections'] = y
to_keep_cols.head()
corr = to_keep_cols.corr()



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 12))



# Create heatmap

heatmap = sns.heatmap(corr,annot=True)

plt.show()
to_keep_cols.nunique()
to_keep_cols.isnull().sum()
cols = ["AVProductsInstalled", "Wdft_IsGamer","Census_OSInstallLanguageIdentifier","Census_InternalBatteryNumberOfCharges","Census_FirmwareVersionIdentifier"]

to_keep_cols[cols]=to_keep_cols[cols].fillna(to_keep_cols.mean().iloc[0])
import seaborn as sns

sns.countplot(x='HasDetections',data=to_keep_cols)

plt.show()
import itertools

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
y = to_keep_cols['HasDetections']

X = to_keep_cols.drop(['HasDetections'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(random_state=42, max_depth=80,max_features=3,min_samples_leaf=5,min_samples_split= 8,n_estimators= 300)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



# prediction on test set

y_pred=clf.predict_proba(X_test)[:,1]

y_pred
clf.feature_importances_
import pandas as pd

%matplotlib inline

plt.rcParams['figure.figsize'] = (20,10)



#do code to support model

#"data" is the X dataframe and model is the SKlearn object



feats = {} # a dict to hold feature_name: feature_importance

for feature, importance in zip(to_keep_cols.columns, clf.feature_importances_):

    feats[feature] = importance #add the name/value pair 



importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})

importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(max_depth=5,

                                   num_leaves=20,

                                   n_estimators=30000,

                                   learning_rate=0.05,

                                   colsample_bytree=0.28,

                                   objective='binary',

                                   boosting_type='gbdt',

                                   metric='auc',

                                   nthread=-1,

                                   bagging_freq = 1,

                                   lambda_l1 = 0.1, 

                                   lambda_l2 = 0.1,

                                   n_jobs=-1)
lgb_model.fit(X_train, y_train, eval_metric='auc', 

                  eval_set=[(X_test, y_test)], 

                  verbose=100, early_stopping_rounds=500)
test_df=pd.read_csv('../input/test.csv',dtype=dtypes)
X_test= test_df.drop(['MachineIdentifier','PuaMode','Census_ProcessorClass','Census_InternalBatteryType',],axis=1)
X_test.head()
X_test[X_test.select_dtypes(['object']).columns] = X_test.select_dtypes(['object']).apply(lambda x: x.astype('category'))

X_test[X_test.select_dtypes(['category']).columns] = X_test.select_dtypes(['category']).apply(lambda x: x.cat.codes)
test_X = X_test[["SmartScreen","OsBuildLab","Census_ChassisTypeName","AVProductsInstalled","AppVersion","Wdft_IsGamer","AvSigVersion",

"Processor","Census_OSInstallLanguageIdentifier","Census_OSEdition",

                       'Census_InternalBatteryNumberOfCharges',"Census_OSBuildNumber","Census_FirmwareVersionIdentifier","Census_HasOpticalDiskDrive"

                       ]]
cols = ["AVProductsInstalled", "Wdft_IsGamer","Census_OSInstallLanguageIdentifier","Census_InternalBatteryNumberOfCharges","Census_FirmwareVersionIdentifier"]

test_X[cols]=test_X[cols].fillna(test_X.mean().iloc[0])
y_pred_submission = lgb_model.predict_proba(test_X)[:,1]
sub_df = pd.DataFrame({"MachineIdentifier": test_df["MachineIdentifier"].values})

sub_df["HasDetections"] = y_pred_submission

sub_df[:10]
sub_df.to_csv("submit.csv", index=False)