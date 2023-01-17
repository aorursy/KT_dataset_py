# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
ppl = pd.read_csv('../input/people.csv', dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8},

                  parse_dates=['date'])

print(ppl.head())

print(ppl.columns.values)



data_train = pd.read_csv('../input/act_train.csv', dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8}, parse_dates=['date'])

data_test = pd.read_csv('../input/act_test.csv', dtype={'people_id': np.str, 'activity_id': np.str}, parse_dates=['date'])

print(data_train.head())

print('Train data shape: ' + format(data_train.shape))

print('Test data shape: ' + format(data_test.shape))

print('ppl shape: ' + format(ppl.shape))
def act_data_preproc(dat_orig):

    dataset = dat_orig

    for col in dataset.columns.values:

        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:

            if dataset[col].dtype == 'object':

                dataset[col] = dataset[col].fillna('type 0')

                dataset[col] = dataset[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)

            elif dataset[col].dtype == 'bool':

                dataset[col] = dataset[col].astype(np.int8)

    dataset['day'] = dataset['date'].dt.day.astype(np.int16)

    dataset['month'] = dataset['date'].dt.month.astype(np.int16)

    dataset['year'] = dataset['date'].dt.year.astype(np.int16)

    dataset['wkday'] = dataset['date'].dt.weekday.astype(np.int8)

    dataset = dataset.drop('date', axis=1)

    return dataset



data_train = act_data_preproc(data_train)

data_test = act_data_preproc(data_test)

ppl = act_data_preproc(ppl)



X_train = data_train.merge(ppl,on='people_id',how='left')

X_test = data_test.merge(ppl,on='people_id',how='left')

print(X_train.loc[:15, ['people_id', 'activity_id', 'group_1', 'wkday_y', 'char_38']])

print(X_test.loc[:15, ['people_id', 'activity_id', 'group_1', 'wkday_y', 'char_38']])

print('Train data shape: ' + format(X_train.shape))

print('Test data shape: ' + format(X_test.shape))

del data_train, data_test, ppl



whole = pd.concat([X_train.drop('outcome',axis=1),X_test],ignore_index=True)

for col in X_train.columns.values[X_train.columns.values!='outcome']:

    print(col + ' category number: ' + str(len(X_train[col].unique())) + ' / ' + str(len(whole[col].unique())))

del whole
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from scipy.sparse import hstack, coo_matrix

from sklearn.metrics import roc_auc_score, roc_curve



# def act_data_enc(X_train, X_test, categorical):

#     whole = pd.concat([X_train, X_test], ignore_index=True)

#     temp = whole.columns.isin(categorical)

#     print(sum(temp))

#     enc = OneHotEncoder(categorical_features=temp)

#     enc = enc.fit(whole)

#     del whole, temp

#     X_train_sparse = enc.transform(X_train)

#     X_test_sparse = enc.transform(X_test)

#     return X_train_sparse, X_test_sparse



# def act_data_rfc(Y_train, X_train, X_test, n_estimator=1000, max_dep=15):

#     tic = time.clock()

#     rfc = RandomForestClassifier(max_depth=max_dep, n_estimators=n_estimator, random_state=50, n_jobs=4)

#     rfc = rfc.fit(X_train, Y_train)

#     print(time.clock() - tic)

#     y_fitted = pd.DataFrame(rfc.predict(X_train))

#     roc_sc = roc_auc_score(Y_train, y_fitted)

#     y_pred = pd.DataFrame(rfc.predict(X_test))

#     # two columns corresponding to pred_prob of class 0 and 1

#     y_pred_proba = pd.DataFrame(rfc.predict_proba(X_test))

#     return y_fitted, roc_sc, y_pred, y_pred_proba



# X_train_ty1 = X_train.loc[X_train.activity_category == 1, :]

# X_train_ty2 = X_train.loc[X_train.activity_category != 1, :]

# X_test_ty1 = X_test.loc[X_test.activity_category == 1, :]

# X_test_ty2 = X_test.loc[X_test.activity_category != 1, :]

# Y_train_ty1 = X_train_ty1['outcome']

# Y_train_ty2 = X_train_ty2['outcome']

# activity_id_tp1 = list(X_test_ty1.activity_id)

# activity_id_tp2 = list(X_test_ty2.activity_id)



# catgc_ty1 = ['char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'day_x', 'month_x', 'year_x', 'wkday_x', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'day_y', 'month_y', 'year_y', 'wkday_y']

# catgc_ty2 = ['activity_category', 'char_10_x', 'day_x', 'month_x', 'year_x', 'wkday_x', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'day_y', 'month_y', 'year_y', 'wkday_y']

# categorical = ['activity_category', 'char_1_x', 'char_2_x', 'char_3_x', 'char_4_x', 'char_5_x', 'char_6_x', 'char_7_x', 'char_8_x', 'char_9_x', 'char_10_x', 'day_x', 'month_x', 'year_x', 'wkday_x', 'char_1_y', 'group_1', 'char_2_y', 'char_3_y', 'char_4_y', 'char_5_y', 'char_6_y', 'char_7_y', 'char_8_y', 'char_9_y', 'day_y', 'month_y', 'year_y', 'wkday_y']

# other = [col for col in X_train.columns.values if col not in categorical and col not in ['people_id', 'activity_id', 'outcome']]

# X_train_ty1 = X_train_ty1.loc[:,catgc_ty1+other]  # (157615, 56)

# X_train_ty2 = X_train_ty2.loc[:,catgc_ty2+other]  # (2039676, 49)

# X_test_ty1 = X_test_ty1.loc[:,catgc_ty1+other]  # (40092, 56)

# X_test_ty2 = X_test_ty2.loc[:,catgc_ty2+other]  # (458595, 49)

# X_train_tk1, X_test_tk1 = act_data_enc(X_train_ty1, X_test_ty1, catgc_ty1)  # coo_matrix: 157615x20135 / 5449839, 40092x20135 / 1379790

# X_train_tk2, X_test_tk2 = act_data_enc(X_train_ty2, X_test_ty2, catgc_ty2)  # coo_matrix: 2039676x39783 / 56651619, 458595x39783 / 12860350



# # rfc fitting and pred, roc calculation

# tic = time.clock()

# y_fitted_ty1, roc_sc_ty1, y_pred_ty1, y_predpr_ty1 = act_data_rfc(Y_train_ty1, X_train_tk1, X_test_tk1, n_estimator=1000, max_dep=15)  # 0.786395894714

# y_fitted_ty2, roc_sc_ty2, y_pred_ty2, y_predpr_ty2 = act_data_rfc(Y_train_ty2, X_train_tk2, X_test_tk2, n_estimator=100, max_dep=15)  # 0.771382492507

# print('act1: ' + str(roc_sc_ty1) + '\nact2: ' + str(roc_sc_ty2))

# y_pred_ty1.index = activity_id_tp1

# y_pred_ty1.columns = ['outcome']

# y_pred_ty2.index = activity_id_tp2

# y_pred_ty2.columns = ['outcome']

# y_pred = pd.concat([y_pred_ty1, y_pred_ty2], axis=0)

# y_pred = y_pred.reindex(X_test.activity_id.values)

# print(y_pred.shape)

# print(time.clock()-tic)
