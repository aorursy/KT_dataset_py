import numpy as np

import pandas as pd

from sklearn import model_selection, preprocessing

import xgboost as xgb

def process_log():

    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

    train=train[(train.price_doc>1e6) & (train.price_doc!=2e6)  & (train.price_doc!=3e6)  ]

    train['price_doc']*=0.969

    train=train.reset_index(drop=True)

    id_test = test.id



    times=pd.concat([train.timestamp,test.timestamp])

    y_train = train["price_doc"]

    

    num_train=len(train)

    times=pd.concat([train.timestamp,test.timestamp])

    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

    x_test = test.drop(["id", "timestamp"], axis=1)

    df_all=pd.concat([x_train,x_test])

    df_cat=None

    for c in df_all.columns:

        if df_all[c].dtype == 'object':

            if c=='sub_area':

                oh=pd.get_dummies(df_all[c],prefix=c)

                if df_cat is None:

                    df_cat=oh

                else:

                    df_cat=pd.concat([df_cat,oh],axis=1)

                df_all.drop([c],inplace=True,axis=1)

            else:

                lbl = preprocessing.LabelEncoder()

                lbl.fit(list(df_all[c].values))

                df_all[c] = lbl.transform(list(df_all[c].values))



    if df_cat is not None:

        df_all = pd.concat([df_all, df_cat], axis=1)



    x_train=df_all[:len(x_train)]

    x_test=df_all[len(x_train):]



    xgb_params = {

        'eta': 0.05,

        'max_depth': 5,

        'subsample': 0.7,

        'colsample_bytree': 0.7,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'silent': 1,

    }



    x_train=df_all[:len(x_train)]

    x_test=df_all[len(x_train):]





    num_boost_rounds=345

    dtrain = xgb.DMatrix(x_train, np.log(y_train))

    dtest = xgb.DMatrix(x_test)

    model = xgb.train(dict(xgb_params, max_depth=5,silent=1), dtrain,num_boost_round= num_boost_rounds)

    y_predict_log=np.exp(model.predict(dtest))

    y_predict=y_predict_log

    return id_test,y_predict

def process():

    train = pd.read_csv('../input/train.csv',parse_dates=['timestamp'])

    train['price_doc']*=0.969

    test = pd.read_csv('../input/test.csv',parse_dates=['timestamp'])

    id_test = test.id



    times=pd.concat([train.timestamp,test.timestamp])

    y_train = train["price_doc"]

    num_train=len(train)



    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

    x_test = test.drop(["id", "timestamp"], axis=1)

    df_all=pd.concat([x_train,x_test])



    for c in df_all.columns:

        if df_all[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(df_all[c].values))

            df_all[c] = lbl.transform(list(df_all[c].values))

    x_train=df_all[:len(x_train)]

    x_test=df_all[len(x_train):]



    

    xgb_params = {

        'eta': 0.05,

        'max_depth': 5,

        'subsample': 0.7,

        'colsample_bytree': 0.7,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'silent': 1,

    }



    dtrain = xgb.DMatrix(x_train, y_train)

    dtest = xgb.DMatrix(x_test)



    num_boost_rounds=345

    model = xgb.train(dict(xgb_params, silent=1), dtrain,num_boost_round= num_boost_rounds)

    y_predict = model.predict(dtest)

  

        

    return id_test,y_predict

if __name__=='__main__':

    id_test,y_predict=process()

    id_test,y_predict_log=process_log()

    print('Mean:',y_predict.mean(), 'LB 0.3113')

    print ('LOG Mean:',y_predict_log.mean(),'LB 0.314-0.315')

    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

    output.to_csv('xgb.csv', index=False)

    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict_log})

    output.to_csv('xgb_log.csv', index=False)
