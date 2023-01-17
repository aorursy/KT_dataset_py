# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc
def print_object_cols(df):

    for col in df.columns:

        if df[col].dtype == np.dtype('O'):

            print(col)

            print(df[col].unique())

            print('-----------------')
def parse_email(df, column_name):

    def parse(value):

        try:

            return value.lower().strip().split('.')[0]

        except Exception as e:

            return 'unknown'

    df.loc[:, column_name] = df[column_name].apply(lambda x: parse(x))

    return df
def convert_string_to_ints(df, column_name, conversion):

    df.loc[:, column_name] = df[column_name].apply(lambda x: 0 if np.nan is x else conversion[x])

    return df
def encode_char_cols_to_ints(df):

    col_name_mappings = {

        'M1': {'T': 1, 'F': 0},

        'M2': {'T': 1, 'F': 0},

        'M3': {'T': 1, 'F': 0},

        'M4': {'M0': 1, 'M1': 2, 'M2': 3},

        'M5': {'T': 1, 'F': 0},

        'M6': {'T': 1, 'F': 0},

        'M7': {'T': 1, 'F': 0},

        'M8': {'T': 1, 'F': 0},

        'M9': {'T': 1, 'F': 0}

    }



    for col_name in col_name_mappings.keys():

        df = convert_string_to_ints(df, col_name, col_name_mappings[col_name])

    

    return df
def parse_emails(df):

    parse_email_cols = ['P_emaildomain', 'R_emaildomain']

    for col in parse_email_cols:

        df = parse_email(df, col)

    return df
def dummify_df(df):

    drop_cols = []

    for cols in df.columns:

        if df[cols].dtype == np.dtype('O'):

            drop_cols.append(cols)

    df1 = pd.get_dummies(df[drop_cols])

    df.drop(drop_cols, axis=1, inplace=True)

    df = pd.concat([df1, df], axis=1)

    return df
def get_features(df):

    df = encode_char_cols_to_ints(df)

    df = parse_emails(df)

    df = dummify_df(df)

    return df
def align_cols(train_cols, test, target_cols):

    for col in train_cols:

        if col not in target_cols:

            if col not in test.columns:

                print(col)

                test.loc[:, col] = 0

    

    print(test.columns[~test.columns.isin(train_cols)])

    test_cols = train_cols

    for each_col in target_cols:

        test_cols.remove(each_col)

    return test[test_cols]
import re

reg_exp = re.compile(r'\d{1,2}\.?\d{0,2}')
def parse_device_info(df, column_name):

    def parse(value):

        try:

            value_splits = value.split()

            try:

                name = value_splits[0].lower().strip()

            except Exception as e:

                name = 'unknown'

            try:

                version1 = reg_exp.findall(value)[0]

                version = float(version1)

            except Exception as e:

                version = 0

        except Exception as e:

            name = 'unknown'

            version = 0

        return {column_name + '_name': name, column_name + '_version': version}

    

    temp_df = df[column_name].apply(lambda x: parse(x)).apply(pd.Series)

    df = pd.concat([df, temp_df], axis=1)

    df.drop(column_name, axis=1, inplace=True)

    return df
def parse_device_info_identity(df):

    device_cols = ['id_30', 'id_31']

    for col in device_cols:

        df = parse_device_info(df, col)

    return df
def parse_screen_ratio(df, column_name):

    def parse(value):

        try:

            w, h = list(map(lambda x: float(x), value.split('x')))

            ratio = w / h

        except Exception as e:

            w, h, ratio = 0, 0, 0

        return {column_name + '_w': w, column_name + '_h': h, column_name + '_ratio': ratio}

    

    df1 = df[column_name].apply(lambda x: parse(x)).apply(pd.Series)

    df = pd.concat([df, df1], axis=1)

    df.drop(column_name, axis=1, inplace=True)

    return df
def identity_encode_char_cols_to_ints(df):

    col_name_mappings = {

        'id_12': {'NotFound': 0, 'Found': 1},

        'id_15': {'Found': 1, 'Unknown': 0, 'New': -1},

        'id_16': {'Found': 1, 'NotFound': 0},

        'id_23': {'IP_PROXY:TRANSPARENT': 1, 'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 3},

        'id_27': {'Found': 1, 'NotFound': 0},

        'id_28': {'New': -1, 'Found': 1},

        'id_29': {'Found': 1, 'NotFound': 0},

        'id_34': {'match_status:2': 2, 'match_status:1': 1, 'match_status:-1': -1, 'match_status:0': 0},

        'id_35': {'T': 1, 'F': 0},

        'id_36': {'T': 1, 'F': 0},

        'id_37': {'T': 1, 'F': 0},

        'id_38': {'T': 1, 'F': 0},

        'DeviceType': {'mobile': 1, 'desktop': 0}

    }



    print(col_name_mappings.keys())

    for col_name in col_name_mappings.keys():

        print(col_name)

        df = convert_string_to_ints(df, col_name, col_name_mappings[col_name])

    

    return df
def replace_names(df, column_name, names_to_replace):

    df.loc[:, column_name] = df[column_name].apply(lambda x: names_to_replace[x] if x in names_to_replace.keys() else x)

    return df
def drop_cols(df, cols_to_drop):

    return df.drop(cols_to_drop, axis=1)
def process_identity_df(df):

    identity_encode_char_cols_to_ints(df)

    print('    get screen ratio...')

    df = parse_screen_ratio(df, 'id_33')

    print('    parsing device info identity...')

    df = parse_device_info_identity(df)

    names_to_replace = {

        'mozilla/firefox': 'firefox',

        'generic/android': 'android',

        'samsung/sm-g532m': 'samsung',

        'samsung/sm-g531h': 'samsung',

        'samsung/sch': 'samsung'

    }

    print('    replacing names...')

    df = replace_names(df, 'id_31_name', names_to_replace)

    print('    dropping cols...')

    df = drop_cols(df, ['DeviceInfo'])

    df = dummify_df(df)

    return df
def get_scaler():

    from sklearn.preprocessing import MinMaxScaler

    

    return MinMaxScaler(feature_range=(0, 1))
def normalize_data(scaler, df):

    scaler.fit(df.values)

    

    return scaler.transform(df.values)
data_dir_path = '/kaggle/input/ieee-fraud-detection'
print('Loading train data...')

train = pd.read_csv(os.path.join(data_dir_path, 'train_transaction.csv'))

print('Train data loaded...')
drop_cols_na = []

for col in train.columns:

    value_counts = train[col].isna().value_counts() / train.shape[0] * 100

    if True in value_counts.index:

        if value_counts[True] > 90:

            drop_cols_na.append(col)

            

print(drop_cols_na)
# train.drop(drop_cols_na, axis=1, inplace=True)
print('Getting train features...')

train_feats = get_features(train)
gc.collect()
print('Loading train identity data...')

train_identity = pd.read_csv(os.path.join(data_dir_path, 'train_identity.csv'))

ti = train_identity

print('Train identity data loaded...')
print('Process train identity data...')

ti = process_identity_df(ti)
print_object_cols(ti)
gc.collect()
# select_cols = [

#     'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10',

#     'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20',

#     'id_33_w', 'id_33_h', 'id_33_ratio',

#     'TransactionID'

# ]
ti.columns
scaler = get_scaler()
print_object_cols(train_feats)
print('Merging train and train identity data...')

train_final = train_feats.merge(ti, on='TransactionID', how='left', suffixes=('', '_IDENTITY'))

del train, train_feats, ti
print_object_cols(train_final)
# converted = train_final.loc[train_final.isFraud == 0]

# nonconverted = train_final.loc[train_final.isFraud == 1]



# train = pd.concat([nonconverted, converted.sample(n=20*nonconverted.shape[0])])
# train = pd.concat([nonconverted, converted.sample(n=nonconverted.shape[0])])

train = train_final
converted.shape, nonconverted.shape
train.shape, train_final.shape
print_object_cols(train)
print('Normalizing data...')

train_scaled = normalize_data(scaler, train.drop(['TransactionID', 'isFraud'], axis=1))

train_scaled_df = pd.DataFrame(train_scaled, columns=train.drop(['TransactionID', 'isFraud'], axis=1).columns)

train_scaled_df.loc[:, 'TransactionID'] = train['TransactionID']

train_scaled_df.loc[:, 'isFraud'] = train['isFraud']
gc.collect()
del train_final
import lightgbm as lgbm
print('Creating model...')

model = lgbm.LGBMClassifier(boosting_type='gbdt', learning_rate=0.01, reg_alpha=0.00314, reg_lambda=0.07, n_estimators=1000)
model
train_scaled_df.replace({np.inf: 0}, inplace=True)

train_scaled_df.fillna(0, inplace=True)
print('Training model...')

model.fit(train_scaled_df.drop(['TransactionID', 'isFraud'], axis=1).values, train_scaled_df.isFraud.values)
gc.collect()
train_cols = train_scaled_df.columns.tolist()

del train
gc.collect()
print('Loading test data...')

test = pd.read_csv(os.path.join(data_dir_path, 'test_transaction.csv'))

print('Test data loaded...')
print('Getting test features...')

test_feats = get_features(test)
print('Loading Test identity data...')

test_identity = pd.read_csv(os.path.join(data_dir_path, 'test_identity.csv'))

tei = test_identity

print('Test identity data loaded...')
print('Process test identity data...')

tei = process_identity_df(tei)
print('Merging Test identity data...')

test_final = test.merge(tei, on='TransactionID', how='left', suffixes=('', '_IDENTITY'))

del test, tei
test_cols_df = align_cols(train_cols, test_final, ['isFraud'])
test_cols_df.columns
test_cols_df.columns[~test_cols_df.columns.isin(train_cols)]
# test = scaler.transform(test_cols_df.drop(['TransactionID'], axis=1))
print('Doing predictions...')

all_preds = []

start_i = 0

end_i = 10000

for i in range(0, test_cols_df.shape[0], 10000):

    start_i = i

    end_i = min(start_i + 10000, test_cols_df.shape[0])

    preds = model.predict_proba(scaler.transform(test_cols_df.drop('TransactionID', axis=1).iloc[start_i: end_i].values))[:, 1]

    all_preds.extend(preds)

probs_df = pd.DataFrame(all_preds, columns=['isFraud'])

probs_df.loc[:, 'TransactionID'] = test_cols_df['TransactionID']

probs_df[['TransactionID', 'isFraud']].to_csv('test_submission.csv', index=False)

print('File saved...')