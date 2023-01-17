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
import copy



from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor



from xgboost import XGBRegressor



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
def feature_engineering(df_ori):

    df_processed = copy.deepcopy(df_ori)

#     feature_list = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain',

#                     'id_12','id_15','id_16','id_23','id_16','id_27','id_28','id_29','id_30']

    feature_list = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain']

    for feature in feature_list:

        content_list = df_ori[feature].unique().tolist()

        content_list = [x for x in content_list if str(x) != 'nan']

        feature_replace_dict = {}

        for index in range(0, len(content_list)):

            feature_replace_dict[content_list[index]] = index

        df_processed[feature] = df_ori[feature].replace(feature_replace_dict)

    return df_processed
df_train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

df_train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

df_test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

df_test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
df_train_identity.describe()
df_train_transaction.describe()
print('\nData types.')

print(df_train_transaction.dtypes)
## Join Transaction and Identity data

# df_train = pd.merge(df_train_transaction, df_train_identity, on='TransactionID')

# df_test = pd.merge(df_test_transaction, df_test_identity, on='TransactionID', how='left')

df_train = df_train_transaction

df_test = df_test_transaction

print(len(df_test))

# print(len(df_train))

# df_train.head(5)
## Feature engineering one by one

df_train_processed = feature_engineering(df_train)



## Choose features

# feature_list = ['TransactionAmt','card1','card2','card3','card4','card5','card6','addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain','R_emaildomain',

#                 'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13', 

#                 'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14',

#                 'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14', 

#                 'id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09','id_10',

#                 'id_11','id_12','id_13','id_14','id_15','id_16','id_17','id_18','id_19','id_20',

#                 'id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_28','id_29','id_30']

feature_list = ['TransactionAmt','card1','card2','card3','card4','card5','card6','addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain','R_emaildomain',

                'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13', 

                'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14',

                'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14']

feature_with_label_list = feature_list+['isFraud']

df_feature_with_label = df_train_processed[feature_with_label_list]



## Replace NaN with 0

df_feature_with_label.fillna(0,inplace=True)



## Split train and test

msk = np.random.rand(len(df_feature_with_label)) < 0.8

X_train = df_feature_with_label[feature_list][msk]

X_test = df_feature_with_label[feature_list][~msk]

y_train = df_feature_with_label['isFraud'][msk]

y_test = df_feature_with_label['isFraud'][~msk]
print('X_train len:',len(X_train))

print('y_test len:',len(y_test))

## Label balance

print('# of 1 in y_train',y_train.sum())

# print('# of NaNs in X_train:', X_train.isna().sum())

# print('# of NaNs in y_train:', y_train.isna().sum())
rng = np.random.RandomState(1)



print("Training LogisticRegression...")

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)

y_pred_clf = clf.predict(X_test)

mse_clf = mean_squared_error(y_test, y_pred_clf)



# print("Training MLPRegressor...")

# mlp = MLPRegressor()

# mlp.fit(X_train, y_train)

# y_pred_mlp = mlp.predict(X_test)

# mse_mlp = mean_squared_error(y_test, y_pred_mlp)



# print("Training GradientBoostingRegressor...")

# gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4,learning_rate=0.1, 

#                                 loss='huber',random_state=1)

# gbr.fit(X_train, y_train)

# y_pred_gbr = gbr.predict(X_test)

# mse_gbr = mean_squared_error(y_test, y_pred_gbr)



# print("Training AdaBoostRegressor...")

# abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, random_state=rng)

# abr.fit(X_train, y_train)

# y_pred_abr = abr.predict(X_test)

# mse_abr = mean_squared_error(y_test, y_pred_abr)



# print("Training BaggingRegressor...")

# bgr = BaggingRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, random_state=rng)

# bgr.fit(X_train, y_train)

# y_pred_bgr = bgr.predict(X_test)

# mse_bgr = mean_squared_error(y_test, y_pred_bgr)



# print("Training RandomForestRegressor...")

# rfr = RandomForestRegressor(n_estimators=200, random_state=rng)

# rfr.fit(X_train, y_train)

# y_pred_rfr = rfr.predict(X_test)

# mse_rfr = mean_squared_error(y_test, y_pred_rfr)



print("Training XGBRegressor...")

xgb = XGBRegressor(max_depth=4, n_estimators=300, booster='dart')

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)





print("mse_clf:",mse_clf)

# print("mse_mlp:",mse_mlp)

# print("mse_gbr:",mse_gbr)

# print("mse_abr:",mse_abr)

# print("mse_bgr:",mse_bgr)

# print("mse_rfr:",mse_rfr)

# print("mse_rfr:",mse_sgd)

print("mse_xgb:",mse_xgb)
print(y_pred_xgb)
df_test_processed = feature_engineering(df_test)

real_X_test = df_test_processed[feature_list]

real_X_test.fillna(0,inplace=True)

## Feature Process



## Try different models

# real_pred = clf.predict(real_X_test)

# real_pred = mlp.predict(real_X_test)

# real_pred = gbr.predict(real_X_test)

# real_pred = abr.predict(real_X_test)

# real_pred = bgr.predict(real_X_test)

# real_pred = rfr.predict(real_X_test)

real_pred = xgb.predict(real_X_test)

print(real_pred)
df_output = pd.DataFrame(data=df_test_transaction.TransactionID,columns=['TransactionID'])

df_output.reset_index(inplace=True, drop=True)



print(len(df_output))

df_output.head(3)

df_output['isFraud'] = real_pred

df_output.reset_index(inplace=True, drop=True)
## Write out

df_output.to_csv('/kaggle/input/ieee-fraud-detection/submission.csv', index=False)

df_output.head(5)