# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import PowerTransformer

from sklearn.decomposition import PCA
TARGET = 'isFraud'

PREPROCESS_BLACKLIST = {TARGET, 'TransactionID', 'TransactionDT'}

FORCE_KEEP = {'dist1'}

GROUP_CARDINALITY_MAX = 5

CATEGORICAL_FEATURES = {'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',

                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

                       'id_12', 'id_13', 'id_14', 'id_15', 'id_16',

       'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23',

       'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',

       'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37',

       'id_38', 'DeviceType', 'DeviceInfo'}



PCA_PREFIX = '_pc_'

PCA_N_COMPONENTS = 19

PCA_FEATURES = set([f'{PCA_PREFIX}{i}' for i in range(PCA_N_COMPONENTS)])

pca_input = set()

for i in range(1, 340):

    pca_input.add(f'V{i}')
# Characters such as empty strings '' or numpy.inf are considered NA values

pd.set_option('use_inf_as_na', True)

pd.set_option('display.max_columns', 500)
def is_categorical(df, col):

    return col in CATEGORICAL_FEATURES or pd.api.types.is_string_dtype(df[col])
%%time

folder_path = '../input/ieee-fraud-detection'

train_identity = pd.read_csv(f'{folder_path}/train_identity.csv')

train_transaction = pd.read_csv(f'{folder_path}/train_transaction.csv')

test_identity = pd.read_csv(f'{folder_path}/test_identity.csv')

test_transaction = pd.read_csv(f'{folder_path}/test_transaction.csv')



# let's combine the data and work with the whole dataset

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

del train_identity, train_transaction, test_identity, test_transaction
%%time

def handle_categorical_variables(df, columns):

    df[columns] = df[columns].astype(str)

    #df[columns] = df[columns].fillna(imput_value)

    #df[columns] = df[columns].replace('nan', imput)

    return df





cols = list(CATEGORICAL_FEATURES)

train = handle_categorical_variables(train, cols)

test = handle_categorical_variables(test, cols)
train.info()
test.info()
train.columns.values
train.head()
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

one_value_cols == one_value_cols_test
def combine_address_columns(df):

    cols = ['addr1', 'addr2']

    df['addr'] = df[cols].apply(','.join, axis=1)

    df = df.drop(columns=cols)

    return df





train = combine_address_columns(train)

test = combine_address_columns(test)
def _missing_data(df):

    total = train.isnull().sum().sort_values(ascending=False)

    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data[missing_data['Total'] > 0]





missing = _missing_data(train)

missing.head()
tmp = missing.filter(items=['dist1'], axis=0)

tmp.head()
cols = missing[missing['Percent'] > 0.5].index.values

cols = [col for col in cols if col not in FORCE_KEEP]

print(f'drop {len(cols)} columns={cols}')
train = train.drop(columns=cols)

test = test.drop(columns=cols)

print(f'train={train.shape}, test={test.shape}')
def imput_mode(col, train, test):

    imput = train[col].mode()[0]

    train[col].fillna(imput, inplace = True)

    test[col].fillna(imput, inplace = True)

    return imput





def imput_median(col, train, test):

    imput = train[col].median()

    train[col].fillna(imput, inplace = True)

    test[col].fillna(imput, inplace = True)

    return imput

    



cols = [col for col in train if col not in PREPROCESS_BLACKLIST and not is_categorical(train, col)]

for i, col in enumerate(cols):

    imput = imput_median(col=col, train=train, test=test)

    print(f'{i + 1}. "{col}" imput median={imput}')
# sanity check to ensure no missing data

missing = _missing_data(train)

print(f'[{len(missing) == 0}] train has no missing data')
missing = _missing_data(test)

print(f'[{len(missing) == 0}] test has no missing data')
train[TARGET].value_counts()
train['TransactionDT'].describe().apply(lambda x: format(x, 'f'))
test['TransactionDT'].describe().apply(lambda x: format(x, 'f'))
CATEGORICAL_FEATURES.add('addr')

CATEGORICAL_FEATURES.remove('addr1')

CATEGORICAL_FEATURES.remove('addr2')

cmap = {}

for cf in CATEGORICAL_FEATURES:

    cmap[cf] = len(train[cf].unique())



print(f'cmap={cmap}')
groups = []

for k, v in cmap.items():

    if v <= GROUP_CARDINALITY_MAX:  # max theshold for number of distinct values

        groups.append([k])

        

print(f'{len(groups)} groups={groups}')
%%time

def ratio_to_group(train, test, target_columns, group_columns, statistic):

    for t in target_columns:

        d = train.groupby(group_columns)[t].transform(statistic)

        col = f'{t}_to_{"_".join(group_columns)}_{statistic}'

        train[col] = train[t] / d

        test[col] = test[t] / d

        train[col] = train[col].fillna(0)

        test[col] = test[col].fillna(0)

    return train, test



cols = ['TransactionAmt', 'dist1']

statistics = ['mean', 'std']

for g in groups:

    for s in statistics:

        train, test = ratio_to_group(train, test, target_columns=cols, group_columns=g, statistic=s)





cols = train.columns.values

print(f'{len(cols)} columns={cols}')
def encode(df, col, encoder):

    df[col] = df[col].map(encoder).fillna(0)

    assert df[col].isnull().sum() == 0



def freq_encode(col):

    encoder = dict(train[col].value_counts(normalize=True))

    encode(train, col, encoder)

    encode(test, col, encoder)





cols = [col for col in train if col not in PREPROCESS_BLACKLIST and is_categorical(train, col)]

fe_cols = set(cols)

print(f'Frequency encode {len(cols)} columns={cols}')
%%time

for col in cols:

    freq_encode(col)



train.head()
%%time

cols = set(train.columns.values) - PREPROCESS_BLACKLIST - fe_cols - PCA_FEATURES

cols = list(cols)

print(f'transform {len(cols)} columns={cols}')

pt = PowerTransformer()

pt.fit(train[cols]) 

train[cols] = pt.transform(train[cols])

test[cols] = pt.transform(test[cols])

# imput zero for any numerical errors

train = train.fillna(0)

test = test.fillna(0)

train.head()
%%time

def _pca_features(dfs, cols, n_components, prefix):

    pca = PCA(n_components=n_components)

    pca.fit(dfs[0][cols])

    res = []

    for df in dfs:

        pcs = pd.DataFrame(pca.transform(df[cols]))

        pcs.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

        df = pd.concat([df, pcs], axis=1)

        #df.drop(columns=cols, inplace=True)

        res.append(df)

    return res





cols = pca_input & set(train.columns.values)

cols = list(cols)

print(f'PCA {len(cols)} columns={cols}')

dfs = _pca_features(dfs=[train, test], cols=cols, n_components=PCA_N_COMPONENTS, prefix=PCA_PREFIX)

train, test = dfs[0], dfs[1]

train.head()
test.head()
train = train.sort_values(by=['TransactionDT'])

#val_size = int(0.01 * len(train))

#val = train[-val_size:]

#val['isFraud'].value_counts()

#del val
cols = ['TransactionID', 'TransactionDT']

train = train.drop(columns=cols)

test = test.drop(columns=cols)

cols = train.columns.values

print(f'{len(cols)} columns={cols}')
cdt_map = {TARGET: 'uint8'}

for col in test.columns:

    cdt_map[col] = 'float32'



train = train.astype(cdt_map)

del cdt_map[TARGET]

test = test.astype(cdt_map)

train.head()
train.info()
test.info()
train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)

print(os.listdir("."))