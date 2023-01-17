import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

from sklearn import preprocessing

from sklearn.preprocessing import Imputer



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train_i = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_t = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')



test_i = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

test_t = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
train_i['is_in_identity'] = 1



train = pd.merge(

    train_t,

    train_i,

    on = 'TransactionID',

    how = 'left'

)



train['is_in_identity'] = train['is_in_identity'].fillna(value = 0)



test_i['is_in_identity'] = 1



test = pd.merge(

    test_t,

    test_i,

    on = 'TransactionID',

    how = 'left'

)



test['is_in_identity'] = test['is_in_identity'].fillna(value = 0)





del train_i, train_t, test_i, test_t
print(train.shape)

print(test.shape)
o_types = train.dtypes.to_frame().reset_index()

o_types.columns = ['col', 'type']

o_types = o_types[o_types['type'] == 'object']

o_types_counts = train[o_types['col']].nunique().to_frame().reset_index()

o_types_counts.columns = ['col', 'cnt']

o_types_one_hot = o_types_counts['col'][o_types_counts.cnt <= 10]

o_types_label_encode = o_types_counts['col'][o_types_counts.cnt > 10]
train_one_hot = pd.get_dummies(train[o_types_one_hot])



train = pd.concat(

    [

        train,

        train_one_hot

    ],

    axis = 1

)



train = train.drop(columns = train[o_types_one_hot])



del train_one_hot
test_one_hot = pd.get_dummies(test[o_types_one_hot])



test = pd.concat(

    [

        test,

        test_one_hot

    ],

    axis = 1

)



test = test.drop(columns = test[o_types_one_hot])



del test_one_hot
print(train.shape)

print(test.shape)
P_emaildomain_encoder =  preprocessing.LabelEncoder()

id_30_encoder =  preprocessing.LabelEncoder()

id_33_encoder =  preprocessing.LabelEncoder()

DeviceInfo_encoder =  preprocessing.LabelEncoder()



train['P_emaildomain'].fillna(value = '0', inplace = True)

train['id_30'].fillna(value = '0', inplace = True)

train['id_33'].fillna(value = '0', inplace = True)

train['DeviceInfo'].fillna(value = '0', inplace = True)



test['P_emaildomain'].fillna(value = '0', inplace = True)

test['id_30'].fillna(value = '0', inplace = True)

test['id_33'].fillna(value = '0', inplace = True)

test['DeviceInfo'].fillna(value = '0', inplace = True)



P_emaildomain_encoder.fit(list(train['P_emaildomain']) + list(test['P_emaildomain']))

id_30_encoder.fit(list(train['id_30']) + list(test['id_30']))

id_33_encoder.fit(list(train['id_33']) + list(test['id_33']))

DeviceInfo_encoder.fit(list(train['DeviceInfo']) + list(test['DeviceInfo']))



train['P_emaildomain_encoder'] = P_emaildomain_encoder.transform(train['P_emaildomain'])

train['id_30_encoder'] = id_30_encoder.transform(train['id_30'])

train['id_33_encoder'] = id_33_encoder.transform(train['id_33'])

train['DeviceInfo_encoder'] = DeviceInfo_encoder.transform(train['DeviceInfo'])



test['P_emaildomain_encoder'] = P_emaildomain_encoder.transform(test['P_emaildomain'])

test['id_30_encoder'] = id_30_encoder.transform(test['id_30'])

test['id_33_encoder'] = id_33_encoder.transform(test['id_33'])

test['DeviceInfo_encoder'] = DeviceInfo_encoder.transform(test['DeviceInfo'])



train = train.drop(columns = train[o_types_label_encode])

test = test.drop(columns = test[o_types_label_encode])
print(train.shape)

print(test.shape)
imputer = Imputer(strategy = 'mean')

train = pd.DataFrame(

    data = imputer.fit_transform(train),

    columns = train.columns

)



test = pd.DataFrame(

    data = imputer.fit_transform(test),

    columns = test.columns

)
print(train.shape)

print(test.shape)
from datetime import timedelta

train['day_diff'] = ((train['TransactionDT'] - 86400) / (3600 * 24)).astype(int)

train['as_of_date'] = pd.to_datetime('2017-12-01') + pd.to_timedelta(train['day_diff'], unit = 'd')

train['month'] = train['as_of_date'].dt.month

train['day'] = train['as_of_date'].dt.day

train['weekofyear'] = train['as_of_date'].dt.weekofyear

train['weekday'] = train['as_of_date'].dt.weekday

train['dayofyear'] = train['as_of_date'].dt.dayofyear

train['week'] = train['as_of_date'].dt.week

train['dayofweek'] = train['as_of_date'].dt.dayofweek

train['dist_from_xmas'] = (pd.to_datetime('2017-12-25') - train['as_of_date']).dt.days



test['day_diff'] = ((test['TransactionDT'] - 86400) / (3600 * 24)).astype(int)

test['as_of_date'] = pd.to_datetime('2017-12-01') + pd.to_timedelta(test['day_diff'], unit = 'd')

test['month'] = test['as_of_date'].dt.month

test['day'] = test['as_of_date'].dt.day

test['weekofyear'] = test['as_of_date'].dt.weekofyear

test['weekday'] = test['as_of_date'].dt.weekday

test['dayofyear'] = test['as_of_date'].dt.dayofyear

test['week'] = test['as_of_date'].dt.week

test['dayofweek'] = test['as_of_date'].dt.dayofweek

test['dist_from_xmas'] = (pd.to_datetime('2017-12-25') - test['as_of_date']).dt.days
t = train.dtypes.to_frame().reset_index()

t.columns = ['col', 'type']



train = train.drop(columns = t['col'][(t.type == 'object') | (t.type == 'datetime64[ns]')], axis = 1)



te = test.dtypes.to_frame().reset_index()

te.columns = ['col', 'type']



test = test.drop(columns = te['col'][(te.type == 'object') | (te.type == 'datetime64[ns]')], axis = 1)
print(train.shape)

print(test.shape)
for feature in ['card1','card2','addr1']:

    for feature_2 in ['card1','card2','addr1']:

        if feature == feature_2:

            continue

        train[f'{feature}_{feature_2}'] = train[feature].astype(str) + '_' + train[feature_2].astype(str)

        test[f'{feature}_{feature_2}'] = test[feature].astype(str) + '_' + test[feature_2].astype(str)
#https://www.kaggle.com/nroman/recursive-feature-elimination

useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',

                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',

                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',

                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',

                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',

                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',

                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',

                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',

                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',

                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',

                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',

                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',

                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',

                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',

                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',

                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',

                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',

                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',

                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',

                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'device_name', 'device_version', 'OS_id_30', 'version_id_30',

                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id', 

                   'day_diff', 'month', 'day', 'day_of_week', 'dist_from_xmas','weekofyear','weekday','dayofyear','week',

                  'card1_card2', 'card1_addr1', 'card2_addr1',  'TransactionAmt_squared', 'TransactionAmt_log']
t_col = train.columns



use_feat = pd.Series(useful_features).to_frame().reset_index()

use_feat.columns = ['index', 'col']

t_df = t_col.to_frame().reset_index()

t_df.columns = ['index','col']



cols_to_keep = pd.merge(

    use_feat,

    t_df,

    on = 'col',

    how = 'inner'

)

 

cols = cols_to_keep['col'].append(pd.Series('isFraud'))

train = train[cols]
te_col = test.columns



use_feat_t = pd.Series(useful_features).to_frame().reset_index()

use_feat_t.columns = ['index', 'col']

t_df_t = te_col.to_frame().reset_index()

t_df_t.columns = ['index','col']



cols_to_keep_t = pd.merge(

    use_feat_t,

    t_df_t,

    on = 'col',

    how = 'inner'

)



test = test[cols_to_keep_t['col']]
card1_card2_encoder =  preprocessing.LabelEncoder()

card1_addr1_encoder =  preprocessing.LabelEncoder()

card2_addr1_encoder =  preprocessing.LabelEncoder()



train['card1_card2'].fillna(value = '0', inplace = True)

train['card1_addr1'].fillna(value = '0', inplace = True)

train['card2_addr1'].fillna(value = '0', inplace = True)



test['card1_card2'].fillna(value = '0', inplace = True)

test['card1_addr1'].fillna(value = '0', inplace = True)

test['card2_addr1'].fillna(value = '0', inplace = True)



card1_card2_encoder.fit(list(train['card1_card2']) + list(test['card1_card2']))

card1_addr1_encoder.fit(list(train['card1_addr1']) + list(test['card1_addr1']))

card2_addr1_encoder.fit(list(train['card2_addr1']) + list(test['card2_addr1']))



train['card1_card2_encoder'] = card1_card2_encoder.transform(train['card1_card2'])

train['card1_addr1_encoder'] = card1_addr1_encoder.transform(train['card1_addr1'])

train['card2_addr1_encoder'] = card2_addr1_encoder.transform(train['card2_addr1'])



test['card1_card2_encoder'] = card1_card2_encoder.transform(test['card1_card2'])

test['card1_addr1_encoder'] = card1_addr1_encoder.transform(test['card1_addr1'])

test['card2_addr1_encoder'] = card2_addr1_encoder.transform(test['card2_addr1'])



train = train.drop(columns = ['card1_card2', 'card1_addr1','card2_addr1'])

test = test.drop(columns = ['card1_card2', 'card1_addr1','card2_addr1'])
print(train.shape)

print(test.shape)
y = train['isFraud']

train, test = train.align(test, join = 'inner', axis = 1)
print(train.shape)

print(test.shape)
import numpy as np

train['TransactionAmt_squared'] = train['TransactionAmt'] ** 2 

test['TransactionAmt_squared'] = test['TransactionAmt'] ** 2 



train['TransactionAmt_log'] = np.log(train['TransactionAmt'])

test['TransactionAmt_log'] = np.log(test['TransactionAmt'])
import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



x = train

y = y



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)



param= {'num_leaves': 491,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.006883242363721497,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47,

         }
NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = x.columns

splits = folds.split(x, y)

test_predictions = np.zeros(test.shape[0])

y_oof = np.zeros(x.shape[0])

score = 0

feature_importances = pd.DataFrame()

feature_importances['feature'] = columns



for fold_n, (train_index, valid_index) in enumerate(splits):

    x_train, x_valid = x[columns].iloc[train_index], x[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    dtrain = lgb.Dataset(x_train, label=y_train)

    dvalid = lgb.Dataset(x_valid, label=y_valid)



    clf = lgb.train(param, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)

    

    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(x_valid, num_iteration=clf.best_iteration)

    y_oof[valid_index] = y_pred_valid

    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")

    

    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS

    test_predictions += clf.predict(test) / NFOLDS

    

    del x_train, x_valid, y_train, y_valid

    

print(f"\nMean AUC = {score}")

print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")
import matplotlib.pyplot as plt

import seaborn as sns



feature_importances['avg'] = (

    feature_importances['fold_1'] + 

    feature_importances['fold_2'] + 

    feature_importances['fold_3'] + 

    feature_importances['fold_4'] + 

    feature_importances['fold_5']

) / 5



plt.figure(figsize=(16, 16))

sns.barplot(

    data = feature_importances.sort_values(by = 'avg', ascending = False).head(50),

    x = 'avg',

    y = 'feature'

)
test_t = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



submit = pd.concat([test_t['TransactionID'].astype(int), pd.Series(test_predictions)], axis = 1)

submit.columns = ['TransactionID', 'isFraud']



submit.to_csv('submit.csv', index = False) 