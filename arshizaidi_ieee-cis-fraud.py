# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv", index_col='TransactionID')

train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv", index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

del train_transaction, train_identity
test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv", index_col='TransactionID')

test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv", index_col='TransactionID')
test_identity.head()
test_transaction .head()
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

import itertools

from sklearn.preprocessing import LabelEncoder
def encode_categorial_features_fit(df, columns_to_encode):

    encoders = {}

    for c in columns_to_encode:

        if c in df.columns:

            encoder = LabelEncoder()

            encoder.fit(df[c].astype(str).values)

            encoders[c] = encoder

    return encoders



def encode_categorial_features_transform(df, encoders):

    out = pd.DataFrame(index=df.index)

    for c in encoders.keys():

        if c in df.columns:

            out[c] = encoders[c].transform(df[c].astype(str).values)

    return out



categorial_features_columns = [

    'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',

    'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31',

    'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',

    'DeviceType', 'DeviceInfo', 'ProductCD', 'P_emaildomain', 'R_emaildomain',

    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

    'addr1', 'addr2',

    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',

    'P_emaildomain_vendor', 'P_emaildomain_suffix', 'P_emaildomain_us',

    'R_emaildomain_vendor', 'R_emaildomain_suffix', 'R_emaildomain_us'

]





categorial_features_encoders = encode_categorial_features_fit(train, categorial_features_columns)

temp = encode_categorial_features_transform(train, categorial_features_encoders)

columns_to_drop = list(set(categorial_features_columns) & set(train.columns))

train = train.drop(columns_to_drop, axis=1).merge(temp, how='left', left_index=True, right_index=True)

del temp
def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:

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

    return df

train = reduce_mem_usage(train)
train_y = train['isFraud'].copy()

train_x = train.drop('isFraud', axis=1)

del train
def test(x_train, y_train):

    

    params = {

        'objective': 'binary',

        'metric': 'auc',

        'is_unbalance': False,

        'boost_from_average': True,

        'num_threads': 4,

        

        'num_leaves': 200,

        'min_data_in_leaf': 20,

        'max_depth': 30

    }

    

    scores = []

    

    cv = KFold(n_splits=5)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        

        train_x_train = train_x.iloc[train_idx]

        train_y_train = train_y.iloc[train_idx]

        train_x_valid = train_x.iloc[valid_idx]

        train_y_valid = train_y.iloc[valid_idx]

        

        lgb_train = lgb.Dataset(data=train_x_train.astype('float32'), label=train_y_train.astype('float32'))

        lgb_valid = lgb.Dataset(data=train_x_valid.astype('float32'), label=train_y_valid.astype('float32'))

        

        lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_valid, verbose_eval=100)

        y = lgb_model.predict(train_x_valid.astype('float32'), num_iteration=lgb_model.best_iteration)

        

        score = roc_auc_score(train_y_valid.astype('float32'), y)

        print('Fold score:', score)

        scores.append(score)

    

    average_score = sum(scores) / len(scores)

    print('Average score:', average_score)

    return average_score
baseline_score = test(train_x, train_y)



features_to_test = ['id_02', 'id_20', 'D8', 'D11', 'DeviceInfo']



new_features = {}



for features in itertools.combinations(features_to_test, 2):

    new_feature = features[0] + '__' + features[1]

    print('Test feature:', new_feature, '/ interaction of', features[0], 'and', features[1])

    

    temp = train_x.copy()

    temp[new_feature] = temp[features[0]].astype(str) + '_' + temp[features[1]].astype(str)

    temp[new_feature] = LabelEncoder().fit_transform(temp[new_feature].values)

    

    score = test(temp, train_y)

    print('Score =', score)

    

    new_features[new_feature] = score