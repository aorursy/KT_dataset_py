%matplotlib inline

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import scipy
import re
!ls ../input
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

%time train = pd.read_csv('../input/train.csv', nrows=1000000, usecols=dtypes.keys(), low_memory=False)

display(train.describe(include='all').T)
col = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'OsBuildLab', 'Census_OSVersion']
for c in col:
    for i in range(6):
        train[c + str(i)] = train[c].map(lambda x: re.split('\.|-', str(x))[i] if len(re.split('\.|-', str(x))) > i else -1)
        try:
            train[c + str(i)] = pd.to_numeric(train[c + str(i)])
        except:
            print(f'{c + str(i)} cannot be casted to number')
train['HasExistsNotSet'] = train['SmartScreen'] == 'ExistsNotSet'
def split_train_val_set(X, Y, n):
    if n < 1: n=int(len(X.index) * n)
    return X.iloc[:n], X.iloc[n:], Y.iloc[:n], Y.iloc[n:]
for col, val in train.items():
    if pd.api.types.is_string_dtype(val): 
        train[col] = val.astype('category').cat.as_ordered()
        train[col] = train[col].cat.codes
    elif pd.api.types.is_numeric_dtype(val) and val.isnull().sum() > 0:
        train[col] = val.fillna(val.median())

X, Y = train.drop('HasDetections', axis=1), train['HasDetections']
#X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
X_train, X_val, Y_train, Y_val = split_train_val_set(X, Y, n=0.1)
X_train.head(5)
def print_score(m):
    res = [roc_auc_score(m.predict(X_train), Y_train), roc_auc_score(m.predict(X_val), Y_val), 
           m.score(X_train, Y_train), m.score(X_val, Y_val)
          ]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples: forest.check_random_state(rs).randint(0, n_samples, n))
def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
set_rf_samples(50000)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=1, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=10, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
fi = pd.DataFrame({'feature': X_train.columns, 'importance': m.feature_importances_}).sort_values(by='importance', ascending=False)
fi = fi.reset_index()
fi
def plot_feature_importance(fi):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(24,8))
    ax1.plot(np.arange(0, len(fi.index)), fi['importance'])
    label_nrs = np.arange(0, len(fi.index), 5 )
    ax1.set_xticks(label_nrs)
    ax1.set_xticklabels(fi['feature'][label_nrs], rotation=90)
    
    num_bar = min(len(fi.index), 30)
    ax2.barh(np.arange(0, num_bar), fi['importance'][:num_bar], align='center', alpha=0.5)
    ax2.set_yticks(np.arange(0, num_bar))
    ax2.set_yticklabels(fi['feature'][:num_bar])

plot_feature_importance(fi)
to_keep = fi.loc[fi['importance']>0.005, 'feature']
len(to_keep)
X_keep = X.copy()[to_keep]
X_keep.sample(5)
X_train, X_val, Y_train, Y_val = split_train_val_set(X_keep, Y, n=0.1)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
fi = pd.DataFrame({'feature': X_train.columns, 'importance': m.feature_importances_}).sort_values(by='importance', ascending=False)
fi = fi.reset_index()
fi
plot_feature_importance(fi)
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(X_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=X_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
def get_val_score(X_tr, X_v, Y_tr, Y_v):
    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=False)
    m.fit(X_tr, Y_tr)
    scores = [roc_auc_score(m.predict(X_tr), Y_tr), roc_auc_score(m.predict(X_v), Y_v), m.score(X_tr, Y_tr), m.score(X_v, Y_v)]
    print(scores)
    
    return scores
sim_cols = []
num_cols = X_keep.shape[-1]

for row in z[z[:, 2] < 0.1, :2]:
    while np.any(row >= num_cols):
        vals_remove = row[row >= num_cols]
        
        row = np.append(row, z[int(row[row >= num_cols][0]) - num_cols, :2])
        mask = np.isin(row, vals_remove)
        row = row[~mask]
    row = row.astype(int)
    
    sim_cols.append(list(X_keep.columns[row]))
print(sim_cols)
for i, val in enumerate(sim_cols):
    for j in sim_cols:
        if not np.array_equal(val, j):
            if np.all(np.isin(val, j)):
                sim_cols.pop(i)
                break
print(sim_cols)
to_keep = []
get_val_score(X_train, X_val, Y_train, Y_val)
for row in sim_cols:
    scores = []
    for c in row:
        print(c)
        scores.append(get_val_score(X_train.drop(c, axis=1), X_val.drop(c, axis=1), Y_train, Y_val)[1])
    to_keep.append(row[np.argmax(scores)])
to_keep
to_drop = [x for row in sim_cols for x in row if x not in to_keep]
X_train = X_train.drop(to_drop, axis=1)
X_val = X_val.drop(to_drop, axis=1)
m = RandomForestClassifier(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=False)
%time m.fit(X_train, Y_train)

print_score(m)
calc_field = [x for x in X_train.columns if x[-1].isdigit()]
include_cols = [x[:-1] if x[-1].isdigit() else x for x in X_train.columns if x != 'HasExistsNotSet']

cols_del = [x for x in include_cols if x not in X_train.columns]
print(cols_del)

include_cols += ['HasDetections']
print(include_cols)
print(calc_field)
import multiprocessing
import gc
reset_rf_samples()
gc.collect()
del train, X, Y
del X_train, X_val, Y_train, Y_val
del X_keep

del corr, hc, dendrogram
%who
def load_dataframe(dataset):
    cols = include_cols.copy()
    if dataset == 'test':
        cols.remove('HasDetections')
        
    df = pd.read_csv(f'../input/{dataset}.csv', dtype=dtypes, usecols=cols)
    return df
with multiprocessing.Pool(2) as pool: 
    train_df, test_df = pool.map(load_dataframe, ["train", "test"])
#calculate features
for f in calc_field:
    col = f[:-1]
    num = int(f[-1])
    train_df[f] = train_df[col].map(lambda x: re.split('\.|-', str(x))[num] if len(re.split('\.|-', str(x))) > num else -1)
    test_df[f] = test_df[col].map(lambda x: re.split('\.|-', str(x))[num] if len(re.split('\.|-', str(x))) > num else -1)
    
    try:
        train_df[f] = pd.to_numeric(train_df[f], downcast='integer')
        test_df[f] = pd.to_numeric(test_df[f], downcast='integer')
    except:
        train_df[f] = train_df[f].astype('category')
        test_df[f] = test_df[f].astype('category')
        
train_df['HasExistsNotSet'] = train_df['SmartScreen'] == 'ExistsNotSet'
test_df['HasExistsNotSet'] = test_df['SmartScreen'] == 'ExistsNotSet'
Y_train = train_df['HasDetections']
X_train = train_df.drop('HasDetections', axis=1)
X_test = test_df.copy()

cat_columns = X_train.select_dtypes(['category']).columns
num_columns = X_train.select_dtypes(['number', 'bool']).columns

cat_dict = {}
for c, val in X_train[cat_columns].items():
    cat_dict[c] = dict([(category, code) for code, category in enumerate(val.cat.categories)])

for c, val in cat_dict.items():
    X_train[c] = X_train[c].cat.codes
    X_test[c].cat.set_categories(train_df[c].cat.categories, inplace=True)
    X_test[c] = X_test[c].cat.codes

num_dict = {c: val.median() for c, val in train_df.loc[:, num_columns].items()}

X_train.loc[:, num_columns] = X_train.loc[:, num_columns].fillna(num_dict)
X_test.loc[:, num_columns] = X_test.loc[:, num_columns].fillna(num_dict)
gc.collect()
m = RandomForestClassifier(n_estimators=64, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=True)
%time m.fit(X_train, Y_train)

pred = m.predict_proba(X_test)
test_df['HasDetections'] = pred[:, -1]
test_df = test_df.loc[:, ['MachineIdentifier', 'HasDetections']]
test_df.to_csv('submissionv2.csv', index=False)
