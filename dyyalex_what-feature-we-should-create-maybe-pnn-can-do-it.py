import pandas as pd

import numpy as np

import pickle as pk

import torch



from sklearn.metrics import log_loss, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



!pip install deepctr_torch



from deepctr_torch.models import DeepFM, xDeepFM, AutoInt, PNN

from deepctr_torch.inputs import  SparseFeat, DenseFeat, get_feature_names



pd.set_option('display.max_columns', 500)



import warnings

warnings.filterwarnings('ignore')


with open('../input/ieeefraud/train.pickle', 'rb') as file:

    train =pk.load(file)

    

with open('../input/ieeefraud/test.pickle', 'rb') as file:

    test =pk.load(file)

    

file.close()
target = ['isFraud']



sparse_features = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',  'addr1', 

'addr2', 'P_emaildomain', 'R_emaildomain',  'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 

'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 

'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 

'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',]



notfeature = ["TransactionID"]



dense_features = [i for i in train.columns if i not in sparse_features+target+notfeature]



test['isFraud'] = 0

data = pd.concat([train, test], axis=0, sort=False)

print(train.shape, test.shape, data.shape)
# train.fillna(-10, inplace=True)

# test.fillna(-10, inplace=True)

data[sparse_features] = data[sparse_features].fillna("null",)

data[dense_features] = data[dense_features].fillna(-999,)
data[sparse_features] = data[sparse_features].astype("str")



# for feat in dense_features:

#     print("start pro ", feat)

#     if data[feat].dtype == 'O':

#         print("object -> float", feat)

#         data[feat] = data[feat].astype("float")
#del train, test
# 1.Label Encoding for sparse features,and do simple Transformation for dense features



for feat in sparse_features:

    # print("Process: ", feat)

    lbe = LabelEncoder()

    data[feat] = lbe.fit_transform(data[feat])

print("Process sparse finish ")



mms = MinMaxScaler(feature_range=(0, 1))

for feat in dense_features:

    print("Process: ", feat)

    data[feat] = mms.fit_transform(np.array(data[feat]).reshape(-1, 1))

print("Process dense finish ")



# 2.count #unique features for each sparse field,and record dense feature field name



fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())

                       for feat in sparse_features] + [DenseFeat(feat, 1,)

                      for feat in dense_features]



dnn_feature_columns = fixlen_feature_columns

linear_feature_columns = fixlen_feature_columns



fixlen_feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model



#train, test = train_test_split(data, test_size=0.2)

train = data[:590540]

test = data[590540:]

print(train.shape, test.shape, data.shape)



train_model_input = [train[name] for name in fixlen_feature_names]

test_model_input = [test[name] for name in fixlen_feature_names]
# import warnings

# warnings.filterwarnings('ignore')



bs = 1024

device = 'cpu'

use_cuda = True

if use_cuda and torch.cuda.is_available():

    print('cuda ready...')

    device = 'cuda:0'



model = AutoInt(dnn_feature_columns,task='binary',device=device)

model.compile("adam", "binary_crossentropy",

              metrics=['auc'], )



history = model.fit(train_model_input, train[target].values,

                    batch_size=bs, epochs=10, verbose=2, validation_split=0.2, shuffle=True)



AI_pred_trn = model.predict(train_model_input, batch_size=bs)

AI_pred_ans = model.predict(test_model_input, batch_size=bs)
# import warnings

# warnings.filterwarnings('ignore')





model = xDeepFM(linear_feature_columns,dnn_feature_columns,task='binary',device=device)

model.compile("adam", "binary_crossentropy",

              metrics=['auc'], )



history = model.fit(train_model_input, train[target].values,

                    batch_size=bs, epochs=10, verbose=2, validation_split=0.2, shuffle=True)



XD_pred_trn = model.predict(train_model_input, batch_size=bs)

XD_pred_ans = model.predict(test_model_input, batch_size=bs)
 

bs = 1024

model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary',device=device)

model.compile("adam", "binary_crossentropy", 

              metrics=['auc', 'logloss'], )



history = model.fit(train_model_input, train[target].values, 

                    batch_size=bs, epochs=10, verbose=2, validation_split=0.2, shuffle=True)



DF_pred_trn = model.predict(train_model_input, batch_size=bs)

DF_pred_ans = model.predict(test_model_input, batch_size=bs)



bs = 1024

model = PNN(dnn_feature_columns,task='binary',device=device)

model.compile("adam", "binary_crossentropy", 

              metrics=['auc', 'logloss'], )



history = model.fit(train_model_input, train[target].values, 

                    batch_size=bs, epochs=10, verbose=2, validation_split=0.2, shuffle=True)



PN_pred_trn = model.predict(train_model_input, batch_size=bs)

PN_pred_ans = model.predict(test_model_input, batch_size=bs)
trn_stack = train[['TransactionID','isFraud']]

test['isFraud'] = 0

tet_stack = test[["TransactionID", "isFraud"]]



trn_stack['AI_pred'] = AI_pred_trn

tet_stack['AI_pred'] = AI_pred_ans



trn_stack['XD_pred'] = XD_pred_trn

tet_stack['XD_pred'] = XD_pred_ans



trn_stack['DF_pred'] = DF_pred_trn

tet_stack['DF_pred'] = DF_pred_ans



trn_stack['PN_pred'] = PN_pred_trn

tet_stack['PN_pred'] = PN_pred_ans
train_fe = [i for i in trn_stack.columns if i not in ['TransactionID', 'isFraud']]

trn_stack.describe()
tet_stack.describe()
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedKFold

import lightgbm as lgb

import gc



def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=3):

    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True)

    # tr_df['VLABEL'] = 0



    X, y = tr_df[features_columns], tr_df[target]

    P, P_y = tt_df[features_columns], tt_df[target]



    tt_df = tt_df[['TransactionID', target]]

    predictions = np.zeros(len(tt_df))

    oof = np.zeros((len(tr_df),1))



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

        print('Fold:', fold_)

        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]

        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

        tr_data = lgb.Dataset(tr_x, label=tr_y)

        vl_data = lgb.Dataset(vl_x, label=vl_y)



        estimator = lgb.train(

            lgb_params,

            tr_data,

            valid_sets=[vl_data],

            verbose_eval=200,

        )

        

        oof[val_idx] = estimator.predict(vl_x).reshape(-1, 1)

        pp_p = estimator.predict(P)

        # Y_label = estimator.predict(X)

        predictions += pp_p / NFOLDS

        # tr_df['VLABEL'] += Y_label / NFOLDS



        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data

        gc.collect()

        

        feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)),

                                       columns=['Value', 'Feature'])

        print(feature_imp)



    # tr_df[['TransactionID', 'VLABEL']].to_csv('submission.csv', index=False)

    tt_df['prediction'] = predictions



    return tt_df, oof
lgb_params = {

                    'objective':'binary',

                    'boosting_type':'gbdt',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.01,

                    'n_estimators':800,

                    'verbose':-1,

                }

 

test_predictions, oof = make_predictions(trn_stack, tet_stack, train_fe, 'isFraud', lgb_params, NFOLDS=2)
test_predictions['isFraud'] = test_predictions['prediction']

test_predictions.sort_values('TransactionID', inplace=True)

test_predictions[['TransactionID','isFraud']].to_csv('submission.csv', index=False)
test_predictions.head()