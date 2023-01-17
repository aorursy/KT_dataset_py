!pip install catboost
import numpy as np

import pandas as pd

from tqdm import tqdm

import copy

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm
train = pd.read_csv('../input/zindiml//Train.csv')

test = pd.read_csv('../input/zindiml/Test.csv')

submission = pd.read_csv('../input/zindiml/SampleSubmission.csv')
train.head()
train.loc[train['marital_status']=='f', 'marital_status'] = 'F'
train['occupation_code'].unique()
X_train = []

X_train_columns = train.columns

c = 0

for v in train.values:

  info = v[:8]

  binary = v[8:]

  index = [k for k, i in enumerate(binary) if i == 1]

  for i in index:

    c+=1

    for k in range(len(binary)):

      if k == i:

        binary_transformed = list(copy.copy(binary))

        binary_transformed[i] = 0

        X_train.append(list(info) + binary_transformed + [X_train_columns[8+k]] + [c])



X_train = pd.DataFrame(X_train)

X_train.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',

       'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',

       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',

       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'product_pred', 'ID2']
X_test = []

true_values = []

c = 0

for v in test.values:

  c += 1

  info = v[:8]

  binary = v[8:]

  index = [k for k, i in enumerate(binary) if i == 1]

  X_test.append(list(info) + list(binary) + [c])

  for k in test.columns[8:][index]:

    true_values.append(v[0] + ' X ' + k)



X_test = pd.DataFrame(X_test)

X_test.columns = ['ID', 'join_date', 'sex', 'marital_status', 'birth_year', 'branch_code',

       'occupation_code', 'occupation_category_code', 'P5DA', 'RIBP', '8NN1',

       '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO',

       'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'ID2']
features_train = []

features_test = []

columns = []



append_features = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 

'N2MW', 'AHXO','BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 

'ECY3', 'ID', 'ID2', 'join_date', 'sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code',

'birth_year']

for v in append_features:

  features_train.append(X_train[v].values.reshape(-1, 1))

  features_test.append(X_test[v].values.reshape(-1, 1))

  columns.append(np.array([v]))



y_train = X_train[['product_pred']]
features_train = np.concatenate(features_train, axis=1)

features_test = np.concatenate(features_test, axis=1)

columns = np.concatenate(np.array(columns))



X_train = pd.DataFrame(features_train)

X_train.columns = columns

X_test = pd.DataFrame(features_test)

X_test.columns = columns
features_train
X_train
from datetime import date



X_train['day'] = X_train['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)

X_train['month'] = X_train['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)

X_train['year'] = X_train['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)

X_train['passed_years'] = date.today().year - pd.to_datetime(X_train['join_date']).dt.year

X_train.loc[:, 'dayofweek'] = pd.to_datetime(X_train['join_date']).dt.dayofweek





X_test.loc[:, 'dayofweek'] = pd.to_datetime(X_test['join_date']).dt.dayofweek

X_test['day'] = X_test['join_date'].apply(lambda x: int(x.split('/')[0]) if (x == x) else np.nan)

X_test['month'] = X_test['join_date'].apply(lambda x: int(x.split('/')[1]) if (x == x) else np.nan)

X_test['year'] = X_test['join_date'].apply(lambda x: int(x.split('/')[2]) if (x == x) else np.nan)

X_test['passed_years'] = date.today().year - pd.to_datetime(X_test['join_date']).dt.year



X_train['join_date'] = X_train['join_date'].fillna(X_train['join_date'].mode()[0])

st_date = pd.to_datetime(X_train['join_date']).min()

X_train['join_date'] = (pd.to_datetime(X_train['join_date']) - st_date).dt.days

X_train['join_date'] = X_train['join_date'].astype(int)



X_test['join_date'] = X_test['join_date'].fillna(X_test['join_date'].mode()[0])

X_test['join_date'] = (pd.to_datetime(X_test['join_date']) - st_date).dt.days

X_test['join_date'] = X_test['join_date'].astype(int)





# X_train['join_date'] = (X_train['join_date'] - np.mean(X_train['join_date']))/ np.std(X_train['join_date'])

# X_test['join_date'] = (X_test['join_date'] - np.mean(X_test['join_date']))/ np.std(X_test['join_date'])



X_train['date_diff'] = X_train['year'] - X_train['birth_year']

X_test['date_diff'] = X_test['year'] - X_test['birth_year']
X_train['day'] = X_train['day'].fillna(X_train['day'].mode()[0])

X_train['month'] = X_train['month'].fillna(X_train['month'].mode()[0])

X_train['year'] = X_train['year'].fillna(X_train['year'].mode()[0])

X_train['date_diff'] = X_train['date_diff'].fillna(X_train['date_diff'].mode()[0])

X_train['passed_years'] = X_train['passed_years'].fillna(X_train['passed_years'].mode()[0])

X_train['dayofweek'] = X_train['dayofweek'].fillna(X_train['dayofweek'].mode()[0])





X_test['day'] = X_test['day'].fillna(X_test['day'].mode()[0])

X_test['month'] = X_test['month'].fillna(X_test['month'].mode()[0])

X_test['year'] = X_test['year'].fillna(X_test['year'].mode()[0])

X_test['date_diff'] = X_test['date_diff'].fillna(X_test['date_diff'].mode()[0])

X_test['passed_years'] = X_test['passed_years'].fillna(X_test['passed_years'].mode()[0])

X_test['dayofweek'] = X_test['dayofweek'].fillna(X_test['dayofweek'].mode()[0])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data = X_train.append(X_test)



le.fit(y_train.iloc[:,0])

y_train = pd.DataFrame(le.transform(y_train.iloc[:,0]))

y_train.columns = ['target']
X_train.columns
X_train['birth_year'] = X_train['birth_year'].astype(int)

X_test['birth_year'] = X_test['birth_year'].astype(int)
# numeric_col = ['join_date','date_diff','birth_year','dayofweek']

# idx = []

# for col in numeric_col:

    

#     mean = X_train[col].mean()

#     std = X_train[col].std()



#     std3r = mean + 4*std

#     std3l = mean - 4*std



#     drop = X_train[col][(X_train[col]>std3r) | (X_train[col]<std3l)].index.values

#     idx.append(drop)

    

#     X_train = X_train.drop(drop).reset_index(drop=True)

#     y_train = y_train.drop(drop).reset_index(drop=True)    
# without doing dummies

plt.figure(figsize=(14,8))

plt.scatter(X_train['join_date'], y_train['target'])
indices = []



indices.append([c for c in X_train.loc[X_train['date_diff']<=15].index.values])

indices.append([c for c in X_train.loc[X_train['date_diff']>=80].index.values])

# indices.append([c for c in X_train.loc[X_train['birth_year']>2010].index.values])

idx = [item for sublist in indices for item in sublist]



idx
print(X_train.shape)

X_train.drop(idx, inplace=True)

y_train.drop(idx, inplace=True)

print(X_train.shape)
all_data = X_train.append(X_test)



# Removed join date from skewness

numeric_col = ['join_date','date_diff','birth_year','dayofweek']

skew = all_data[numeric_col].skew()

skew = skew[abs(skew) > 0.75]

skew


all_data['join_date'] = np.square(all_data['join_date'])

# all_data['birth_year_mean'] = np.log1p(all_data['birth_year_mean'])

# all_data['join_date'] = boxcox(all_data['join_date']+1)



X_train = all_data[:X_train.shape[0]]

X_test = all_data[-X_test.shape[0]:]
# ONEHOT

data = X_train.append(X_test)

data = pd.get_dummies(data, columns=['sex', 'marital_status', \

                                     'branch_code','occupation_code',\

                                     'occupation_category_code','month','year','passed_years'])

X_train = data[:X_train.shape[0]]

X_test = data[-X_test.shape[0]:]
data.columns.values
remove_features = []

for i in X_train.columns:

    if X_train[i].sum()==0:

        remove_features.append(i)
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier

from xgboost import XGBClassifier, XGBRegressor

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neural_network import MLPClassifier
len(X_train.columns)
cat_features = ['sex_F', 'sex_M',

       'marital_status_D', 'marital_status_F', 'marital_status_M',

       'marital_status_P', 'marital_status_R', 'marital_status_S',

       'marital_status_U', 'marital_status_W', 'branch_code_1X1H',

       'branch_code_30H5', 'branch_code_49BM', 'branch_code_748L',

       'branch_code_94KC', 'branch_code_9F9T', 'branch_code_BOAS',

       'branch_code_E5SW', 'branch_code_EU3L', 'branch_code_O4JC',

       'branch_code_O67J', 'branch_code_UAOD', 'branch_code_X23B',

       'branch_code_XX25', 'branch_code_ZFER', 'occupation_code_00MO',

       'occupation_code_0B60', 'occupation_code_0FOI',

       'occupation_code_0KID', 'occupation_code_0OJM',

       'occupation_code_0PO7', 'occupation_code_0S50',

       'occupation_code_0SH6', 'occupation_code_0VYC',

       'occupation_code_0ZND', 'occupation_code_1AN5',

       'occupation_code_1DT6', 'occupation_code_1H8Y',

       'occupation_code_1MB4', 'occupation_code_1MSV',

       'occupation_code_1NFK', 'occupation_code_1YKL',

       'occupation_code_2346', 'occupation_code_2686',

       'occupation_code_2A7I', 'occupation_code_2BE6',

       'occupation_code_2G86', 'occupation_code_2HLT',

       'occupation_code_2JHV', 'occupation_code_2MBB',

       'occupation_code_2R78', 'occupation_code_2US6',

       'occupation_code_2XZ1', 'occupation_code_2YAO',

       'occupation_code_31GG', 'occupation_code_31JW',

       'occupation_code_374O', 'occupation_code_3NHZ',

       'occupation_code_3X46', 'occupation_code_3YQ1',

       'occupation_code_44SU', 'occupation_code_4M0E',

       'occupation_code_4W0D', 'occupation_code_59QM',

       'occupation_code_5FPK', 'occupation_code_5JRZ',

       'occupation_code_5LNN', 'occupation_code_5OVC',

       'occupation_code_6E4H', 'occupation_code_6KYM',

       'occupation_code_6LKA', 'occupation_code_6PE7',

       'occupation_code_6SKY', 'occupation_code_6XXU',

       'occupation_code_6YZA', 'occupation_code_734F',

       'occupation_code_738L', 'occupation_code_73AC',

       'occupation_code_74BF', 'occupation_code_7G9M',

       'occupation_code_7KM4', 'occupation_code_7UDQ',

       'occupation_code_7UHW', 'occupation_code_7UWC',

       'occupation_code_820B', 'occupation_code_834U',

       'occupation_code_8CHJ', 'occupation_code_8HRZ',

       'occupation_code_8Y24', 'occupation_code_93OJ',

       'occupation_code_9B5B', 'occupation_code_9F96',

       'occupation_code_9FA1', 'occupation_code_9HD1',

       'occupation_code_9IM8', 'occupation_code_9IP9',

       'occupation_code_A4ZC', 'occupation_code_A793',

       'occupation_code_AIDS', 'occupation_code_AIIN',

       'occupation_code_APO0', 'occupation_code_AQIB',

       'occupation_code_B3QW', 'occupation_code_B8W8',

       'occupation_code_BER4', 'occupation_code_BFD1',

       'occupation_code_BIA0', 'occupation_code_BP09',

       'occupation_code_BPSA', 'occupation_code_BWBW',

       'occupation_code_BX9E', 'occupation_code_C1E3',

       'occupation_code_C8F6', 'occupation_code_CAAV',

       'occupation_code_CEL6', 'occupation_code_CV2C',

       'occupation_code_CYDC', 'occupation_code_DD8W',

       'occupation_code_DE5D', 'occupation_code_DHSN',

       'occupation_code_DPRV', 'occupation_code_DZRV',

       'occupation_code_E2MJ', 'occupation_code_E39I',

       'occupation_code_E5PF', 'occupation_code_EE5R',

       'occupation_code_F35Z', 'occupation_code_F57O',

       'occupation_code_FJBW', 'occupation_code_FLNZ',

       'occupation_code_FLXH', 'occupation_code_FSWO',

       'occupation_code_FSXG', 'occupation_code_GQ0N',

       'occupation_code_GVZ1', 'occupation_code_GWEP',

       'occupation_code_GZA8', 'occupation_code_H1K7',

       'occupation_code_HAXM', 'occupation_code_HJF4',

       'occupation_code_HSI5', 'occupation_code_HSVE',

       'occupation_code_HTQS', 'occupation_code_I2OD',

       'occupation_code_I31I', 'occupation_code_IE90',

       'occupation_code_IJ01', 'occupation_code_IMHI',

       'occupation_code_INEJ', 'occupation_code_IQFS',

       'occupation_code_IUT9', 'occupation_code_IX8T',

       'occupation_code_IZ77', 'occupation_code_J9SY',

       'occupation_code_JBJP', 'occupation_code_JHU5',

       'occupation_code_JI64', 'occupation_code_JN20',

       'occupation_code_JQH3', 'occupation_code_JS7M',

       'occupation_code_JSAX', 'occupation_code_JUIP',

       'occupation_code_K0DL', 'occupation_code_K5GV',

       'occupation_code_K5LB', 'occupation_code_KBWO',

       'occupation_code_KNVN', 'occupation_code_KPG9',

       'occupation_code_KUPK', 'occupation_code_L1P3',

       'occupation_code_L4PL', 'occupation_code_LAYD',

       'occupation_code_LGTN', 'occupation_code_LLLH',

       'occupation_code_LQ0W', 'occupation_code_M0WG',

       'occupation_code_MEFQ', 'occupation_code_MU16',

       'occupation_code_N2ZZ', 'occupation_code_N7K2',

       'occupation_code_NDL9', 'occupation_code_NFJH',

       'occupation_code_NO3L', 'occupation_code_NQW1',

       'occupation_code_NSJX', 'occupation_code_NX5Y',

       'occupation_code_OEH6', 'occupation_code_OME4',

       'occupation_code_ONY7', 'occupation_code_OPVX',

       'occupation_code_OQMY', 'occupation_code_OYQF',

       'occupation_code_P2K2', 'occupation_code_P4MD',

       'occupation_code_PJR4', 'occupation_code_PKW3',

       'occupation_code_PMAI', 'occupation_code_PPNK',

       'occupation_code_PSUY', 'occupation_code_PWCW',

       'occupation_code_Q0LY', 'occupation_code_Q231',

       'occupation_code_Q2L0', 'occupation_code_Q57T',

       'occupation_code_Q6J6', 'occupation_code_QJID',

       'occupation_code_QQUP', 'occupation_code_QQVA',

       'occupation_code_QS0L', 'occupation_code_QX54',

       'occupation_code_QZYX', 'occupation_code_R44Q',

       'occupation_code_R7GL', 'occupation_code_RE69',

       'occupation_code_RF6M', 'occupation_code_RH2K',

       'occupation_code_RM3L', 'occupation_code_RSN9',

       'occupation_code_RUFT', 'occupation_code_RXV3',

       'occupation_code_RY9B', 'occupation_code_S96O',

       'occupation_code_S9KU', 'occupation_code_SF1X',

       'occupation_code_SS6D', 'occupation_code_SST3',

       'occupation_code_SSTX', 'occupation_code_T6AB',

       'occupation_code_TUN1', 'occupation_code_U37O',

       'occupation_code_U9RX', 'occupation_code_UBBX',

       'occupation_code_UC7E', 'occupation_code_UJ5T',

       'occupation_code_URYD', 'occupation_code_UYDZ',

       'occupation_code_V4XX', 'occupation_code_VREH',

       'occupation_code_VVTC', 'occupation_code_VYSA',

       'occupation_code_VZN9', 'occupation_code_W1X2',

       'occupation_code_W3Y9', 'occupation_code_W3ZV',

       'occupation_code_WE0G', 'occupation_code_WE7U',

       'occupation_code_WIWP', 'occupation_code_WMTK',

       'occupation_code_WSID', 'occupation_code_WSRG',

       'occupation_code_WV7U', 'occupation_code_WVQF',

       'occupation_code_X1JO', 'occupation_code_XC1N',

       'occupation_code_XHJD', 'occupation_code_XVMH',

       'occupation_code_Y1WG', 'occupation_code_Y7G1',

       'occupation_code_YJXM', 'occupation_code_YMGT',

       'occupation_code_YX47', 'occupation_code_Z7PM',

       'occupation_code_ZA1S', 'occupation_code_ZCQR',

       'occupation_code_ZHC2', 'occupation_code_ZKQ3',

       'occupation_code_ZWPL', 'occupation_category_code_56SI',

       'occupation_category_code_90QI', 'occupation_category_code_AHH5',

       'occupation_category_code_JD7X', 'occupation_category_code_L44T',

       'occupation_category_code_T4MS', 'month_1.0', 'month_2.0',

       'month_3.0', 'month_4.0', 'month_5.0', 'month_6.0', 'month_7.0',

       'month_8.0', 'month_9.0', 'month_10.0', 'month_11.0', 'month_12.0',

       'year_2010.0', 'year_2011.0', 'year_2012.0', 'year_2013.0',

       'year_2014.0', 'year_2015.0', 'year_2016.0', 'year_2017.0',

       'year_2018.0', 'year_2019.0', 'year_2020.0', 'passed_years_0.0',

       'passed_years_1.0', 'passed_years_2.0', 'passed_years_3.0',

       'passed_years_4.0', 'passed_years_5.0', 'passed_years_6.0',

       'passed_years_7.0', 'passed_years_8.0', 'passed_years_9.0',

       'passed_years_10.0']
# remove_features = [str(i) for i in remove_features]
# cat_features = [c for c in cat_features if c not in remove_features]
# X_train,xeval,y_train,yeval = train_test_split(X_train,y_train,train_size=0.80,random_state=1236)
# X_train.drop(remove_features, axis=1, inplace=True)

# # xeval.drop(remove_features, axis=1, inplace=True)

# X_test.drop(remove_features, axis=1, inplace=True)
models = []





models.append(CatBoostClassifier(random_state=1, max_depth=3, task_type='GPU', iterations=1900, learning_rate=0.2))

models.append(CatBoostClassifier(random_state=2, max_depth=3, task_type='GPU', iterations=1900, learning_rate=0.2))

models.append(CatBoostClassifier(random_state=21, max_depth=4, task_type='GPU', iterations=1250,learning_rate=0.2))

models.append(CatBoostClassifier(random_state=22, max_depth=4, task_type='GPU', iterations=1250,learning_rate=0.2))

models.append(CatBoostClassifier(random_state=3, max_depth=7, task_type='GPU', iterations=750))

models.append(CatBoostClassifier(random_state=4, max_depth=7, task_type='GPU', iterations=750))

models.append(CatBoostClassifier(random_state=5, max_depth=9, task_type='GPU', iterations=750, l2_leaf_reg=0.3))

models.append(CatBoostClassifier(random_state=6, max_depth=9, task_type='GPU', iterations=750, l2_leaf_reg=0.3))



models_xg = []



models_xg.append(XGBClassifier(random_state=1, max_depth=3, tree_method='gpu_hist', n_estimators=150))

models_xg.append(XGBClassifier(random_state=2, max_depth=3, tree_method='gpu_hist', n_estimators=150))

models_xg.append(XGBClassifier(random_state=21, max_depth=4, tree_method='gpu_hist', n_estimators=120))

models_xg.append(XGBClassifier(random_state=22, max_depth=4, tree_method='gpu_hist', n_estimators=120))



models_xg.append(XGBClassifier(random_state=2, max_depth=5, tree_method='gpu_hist', n_estimators=100))

models_xg.append(XGBClassifier(random_state=2, max_depth=6, tree_method='gpu_hist', n_estimators=100))

models_xg.append(XGBClassifier(random_state=3, max_depth=7, tree_method='gpu_hist', n_estimators=90))

models_xg.append(XGBClassifier(random_state=5, max_depth=11, tree_method='gpu_hist', n_estimators=55, reg_lambda=0.3))

models_xg.append(XGBClassifier(random_state=6, max_depth=11, tree_method='gpu_hist', n_estimators=55, reg_lambda=0.3))

models_xg.append(XGBClassifier(random_state=7, max_depth=12, tree_method='gpu_hist', n_estimators=50, reg_lambda=0.7))

models_xg.append(XGBClassifier(random_state=8, max_depth=12, tree_method='gpu_hist', n_estimators=50, reg_lambda=0.7))

models_xg.append(XGBClassifier(random_state=9, max_depth=13, tree_method='gpu_hist', n_estimators=35, reg_lambda=0.9))

models_xg.append(XGBClassifier(random_state=10, max_depth=13, tree_method='gpu_hist', n_estimators=35, reg_lambda=0.9))

models_xg.append(XGBClassifier(random_state=11, max_depth=14, tree_method='gpu_hist', n_estimators=35, reg_lambda=0.9))

models_xg.append(XGBClassifier(random_state=12, max_depth=14, tree_method='gpu_hist', n_estimators=35, reg_lambda=0.9))

for i in range(len(models)):

    models[i].fit(X_train.drop(columns=['ID', 'ID2','day']), y_train,verbose=100,\

          cat_features=cat_features,\

#          eval_set=(xeval.drop(columns=['ID', 'ID2','day']),yeval),

#           plot=True

         )

    

#CHnaging these features to int

features = ['P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3', 'birth_year']

X_train[features] = X_train[features].astype(int)

X_test[features] = X_test[features].astype(int)

# xeval[features] = xeval[features].astype(int)





for i in tqdm(range(len(models_xg))):

    models_xg[i].fit(X_train.drop(columns=['ID', 'ID2','day']), y_train,verbose=100

#                     eval_set=(xeval.drop(columns=['ID', 'ID2','day']),yeval)

                     )



#3900
predicts = []

for i in tqdm(range(len(models))):

    predicts.append(models[i].predict_proba(X_test.drop(columns=['ID','ID2','day'], axis=1)))

for i in tqdm(range(len(models_xg))):

    predicts.append(models_xg[i].predict_proba(X_test.drop(columns=['ID','ID2','day'], axis=1))) 
# df_feature = pd.DataFrame(index=X_test.drop(columns=['ID','ID2','day']).columns)

# for i in tqdm(range(len(models))):

#     feature_importance = pd.Series(models[i].feature_importances_,index=X_test.drop(columns=['ID','ID2','day']).columns)

#     df_feature[i] = feature_importance



# for i in tqdm(range(len(models_xg))):    

#     feature_importance = pd.Series(models_xg[i].feature_importances_,index=X_test.drop(columns=['ID','ID2','day']).columns)

#     df_feature[i+6] = feature_importance

    

# df_feature

# # feature_importance = pd.Series(models[0].feature_importances_,index=X_test.drop(columns=['ID','ID2','day']).columns)

# # feature_importance

# # for i in range(len(df_feature)):

# # df_feature.plot(kind='barh',figsize=(100,100))

# df_feature['mean_importance'] = df_feature.mean(axis=1)

# df_feature.drop([c for c in df_feature.columns if c != 'mean_importance'], axis=1, inplace=True)

# df_feature = df_feature.sort_values(by=['mean_importance'], ascending=True)
# pd.set_option("display.max_rows", None, "display.max_columns", None) # to print full dataframe
# df_feature
# remove_features = df_feature[df_feature['mean_importance']<1.020845e-05].index.values

# remove_features
y_test = pd.DataFrame(np.mean(predicts, axis=0))

y_test.columns = le.inverse_transform(y_test.columns)
y_test
answer_mass = []

for i in range(X_test.shape[0]):

    id = X_test['ID'].iloc[i]

    

    for c in y_test.columns:

            answer_mass.append([id + ' X ' + c, y_test[c].iloc[i]])

            

df_answer = pd.DataFrame(answer_mass)

df_answer.columns = ['ID X PCODE', 'Label']

for i in range(df_answer.shape[0]):

    if df_answer['ID X PCODE'].iloc[i] in true_values:

        df_answer['Label'].iloc[i] = 1.0
df_answer.reset_index(drop=True, inplace=True)

df_answer.to_csv('submission1.csv', index=False)