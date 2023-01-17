import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')
from scipy.stats import mannwhitneyu
### loading

#df_train = pd.read_csv('application_train.csv')
#df_test = pd.read_csv('application_test.csv')

### converting categorical features to numeric (handmade OHE + the simplest target encoding)

#for c in df_train.columns[df_train.dtypes == 'object']:
#    c_train = set(df_train[c].unique())
#    c_test = set(df_test[c].unique())
#    diff = c_train ^ c_test
#    if len(diff) > 0:
#        print('feature ' + c + ' has different values: ', diff)

#df_train['CODE_GENDER'] = df_train['CODE_GENDER'] \
#                                                        .map(lambda x: x if x != 'XNA' else np.nan)
#df_train['NAME_INCOME_TYPE'] = df_train['NAME_INCOME_TYPE'] \
#                                                        .map(lambda x: x if x != 'Maternity leave' else np.nan)
#df_train['NAME_FAMILY_STATUS'] = df_train['NAME_FAMILY_STATUS'] \
#                                                        .map(lambda x: x if x != 'Unknown' else np.nan)

#for c in df_train.columns[df_train.dtypes == 'object']:
#    for val in df_train[c].unique():
#        df_train[str(c) + '_' + str(val)] = (df_train[c] == val).map(int)

#for c in df_test.columns[df_test.dtypes == 'object']:
#    for val in df_test[c].unique():
#        df_test[str(c) + '_' + str(val)] = (df_test[c] == val).map(int)

#for c in df_train.columns[df_train.dtypes == 'object']:
#    vc = df_train[c].dropna().unique()
#    d = {}
#    for v in vc:
#        target_0 = df_train[((df_train[c] == v) & (df_train['TARGET'] == 0))].shape[0]
#        target_1 = df_train[((df_train[c] == v) & (df_train['TARGET'] == 1))].shape[0]
#        d[v] = float(target_0) / (target_0 + target_1)
#    df_train[c] = df_train[c].map(d)
#    df_test[c] = df_test[c].map(d)

#to_del = []
#for c in df_train.drop('TARGET', axis = 1).columns:
#    if not c in df_test.columns:
#        to_del.append(c)
#df_train.drop(to_del, axis = 1, inplace = True)

### dropping features with zero variance

#features_with_zero_variance_train = df_train.columns[(df_train.std(axis = 0) == 0).values]
#features_with_zero_variance_test = df_test.columns[(df_test.std(axis = 0) == 0).values]
#print('features_with_zero_variance: ', len(features_with_zero_variance_train), len(features_with_zero_variance_test))
#df_train.drop(features_with_zero_variance_test, axis = 1, inplace = True)
#df_test.drop(features_with_zero_variance_test, axis = 1, inplace = True)

### dropping some outliers

#df_train['DAYS_EMPLOYED'] = df_train['DAYS_EMPLOYED'].map(lambda x: x if x <= 0 else np.nan)
#df_test['DAYS_EMPLOYED'] = df_test['DAYS_EMPLOYED'].map(lambda x: x if x <= 0 else np.nan)

#df_train['OWN_CAR_AGE'] = df_train['OWN_CAR_AGE'].map(lambda x: x if x <= 80 else np.nan)

#df_test['REGION_RATING_CLIENT_W_CITY'] = df_test['REGION_RATING_CLIENT_W_CITY'].map(lambda x: x if x >= 0 else np.nan)

#df_train['AMT_INCOME_TOTAL'] = df_train['AMT_INCOME_TOTAL'].map(lambda x: x if x <= 1e8 else np.nan)

#df_train['AMT_REQ_CREDIT_BUREAU_QRT'] = df_train['AMT_REQ_CREDIT_BUREAU_QRT'].map(lambda x: x if x <= 10 else np.nan)

#df_train['OBS_30_CNT_SOCIAL_CIRCLE'] = df_train['OBS_30_CNT_SOCIAL_CIRCLE'].map(lambda x: x if x <= 40 else np.nan)

### filling NaNs by means

#df_train.fillna(df_train.mean(axis = 0), inplace = True)
#df_test.fillna(df_test.mean(axis = 0), inplace = True)
def generate_features(df, ext_feature_name, is_train = True):
    columns_df = ['TARGET', 'SK_ID_CURR'] if is_train else ['SK_ID_CURR']
    df_new = df[columns_df]
    
    columns = df.drop(columns_df + [ext_feature_name], axis = 1).columns

    for c in columns:
        df_new[ext_feature_name + ' - ' + str(c)] = df[ext_feature_name] - df[c]

    for c in columns:
        df_new[ext_feature_name + ' * ' + str(c)] = df[ext_feature_name] * df[c]

    for c in columns:
        df_new[ext_feature_name + ' / ' + str(c)] = df[[ext_feature_name, c]] \
                                                        .apply(lambda x: 0 if x[1] == 0 else float(x[0]) / x[1], axis = 1)

    to_del = []
    for c in df_new.drop(columns_df, axis = 1).columns:
        st, pv = mannwhitneyu(df_new[c], df[ext_feature_name])
        if pv > .05:
            to_del.append(c)
            
    df_new.drop(to_del, axis = 1, inplace = True)
    
    features_with_zero_variance = df_new.columns[(df_new.std(axis = 0) == 0).values]
    if len(features_with_zero_variance) > 0:
        df_new.drop(features_with_zero_variance, axis = 1, inplace = True)
        
    return df_new
#df_EXT_SOURCE_1_train = generate_features(df_train, 'EXT_SOURCE_1', is_train = True)
#df_EXT_SOURCE_1_train.shape
#df_EXT_SOURCE_1_test = generate_features(df_test, 'EXT_SOURCE_1', is_train = False)
#df_EXT_SOURCE_1_test.shape
#common_features = list(np.intersect1d(df_EXT_SOURCE_1_train.drop('TARGET', axis = 1).columns, df_EXT_SOURCE_1_test.columns))

#df_EXT_SOURCE_1_train = df_EXT_SOURCE_1_train[['TARGET'] + common_features]
#df_EXT_SOURCE_1_test = df_EXT_SOURCE_1_test[common_features]

#print(df_EXT_SOURCE_1_train.shape, df_EXT_SOURCE_1_test.shape)
#df_EXT_SOURCE_2_train = generate_features(df_train, 'EXT_SOURCE_2', is_train = True)
#df_EXT_SOURCE_2_train.shape
#df_EXT_SOURCE_2_test = generate_features(df_test, 'EXT_SOURCE_2', is_train = False)
#df_EXT_SOURCE_2_test.shape
#common_features = list(np.intersect1d(df_EXT_SOURCE_2_train.drop('TARGET', axis = 1).columns, df_EXT_SOURCE_2_test.columns))

#df_EXT_SOURCE_2_train = df_EXT_SOURCE_2_train[['TARGET'] + common_features]
#df_EXT_SOURCE_2_test = df_EXT_SOURCE_2_test[common_features]

#print(df_EXT_SOURCE_2_train.shape, df_EXT_SOURCE_2_test.shape)
#df_EXT_SOURCE_3_train = generate_features(df_train, 'EXT_SOURCE_3', is_train = True)
#df_EXT_SOURCE_3_train.shape
#df_EXT_SOURCE_3_test = generate_features(df_test, 'EXT_SOURCE_3', is_train = False)
#df_EXT_SOURCE_3_test.shape
#common_features = list(np.intersect1d(df_EXT_SOURCE_3_train.drop('TARGET', axis = 1).columns, df_EXT_SOURCE_3_test.columns))

#df_EXT_SOURCE_3_train = df_EXT_SOURCE_3_train[['TARGET'] + common_features]
#df_EXT_SOURCE_3_test = df_EXT_SOURCE_3_test[common_features]

#print(df_EXT_SOURCE_3_train.shape, df_EXT_SOURCE_3_test.shape)
#df_train_new = pd.concat([df_EXT_SOURCE_1_train, 
#                          df_EXT_SOURCE_2_train.drop(['TARGET', 'SK_ID_CURR'], axis = 1),
#                          df_EXT_SOURCE_3_train.drop(['TARGET', 'SK_ID_CURR'], axis = 1)], axis = 1)
#df_train_new.shape
#df_test_new = pd.concat([df_EXT_SOURCE_1_test, 
#                         df_EXT_SOURCE_2_test.drop(['SK_ID_CURR'], axis = 1),
#                         df_EXT_SOURCE_3_test.drop(['SK_ID_CURR'], axis = 1)], axis = 1)
#df_test_new.shape
df_train_new = pd.read_csv('../input/hcb-2018-ext-sources-2000-train/HCB_2018_ext_sources_2000_train.csv')
df_train_new.shape
df_test_new = pd.read_csv('../input/hcb-2018-ext-sources-2000-test/HCB_2018_ext_sources_2000_test.csv')
df_test_new.shape
