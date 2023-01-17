import pandas as pd

import numpy as np

from scipy.optimize import curve_fit



%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.metrics import log_loss

from sklearn.preprocessing import OneHotEncoder



import xgboost as xgb



from tensorflow.keras.optimizers import Nadam

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import tensorflow.keras.layers as KL

from datetime import timedelta

import numpy as np

import pandas as pd
def giba_model():

    def exponential(x, a, k, b):

        return a*np.exp(x*k) + b



    def rmse( yt, yp ):

        return np.sqrt( np.mean( (yt-yp)**2 ) )  



    train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

    train['Date'] = pd.to_datetime( train['Date'] )

    train['Province_State'] = train['Province_State'].fillna('')



    test  = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

    test['Date'] = pd.to_datetime( test['Date'] )

    test['Province_State'] = test['Province_State'].fillna('')

    test['Id'] = -1

    test['ConfirmedCases'] = 0

    test['Fatalities'] = 0



    publictest = test.loc[ test.Date > train.Date.max() ].copy()

    train = pd.concat( (train, publictest ) )



    train['ForecastId'] = pd.merge( train, test, on=['Country_Region','Province_State','Date'], how='left' )['ForecastId_y'].values



    train.sort_values( ['Country_Region','Province_State','Date'], inplace=True )

    train = train.reset_index(drop=True)



    train['cid'] = train['Country_Region'] + '_' + train['Province_State']





    train['log0'] = np.log1p( train['ConfirmedCases'] )

    train['log1'] = np.log1p( train['Fatalities'] )



    train['log0'] = train.groupby('cid')['log0'].cummax()

    train['log1'] = train.groupby('cid')['log1'].cummax()



    train = train.loc[ (train.log0 > 0) | (train.ForecastId.notnull()) ].copy()

    train = train.reset_index(drop=True)



    train['day'] = train.groupby('cid')['Id'].cumcount()





    def create_features( df, traindate, lag=1 ):

        df['lag0_1'] = df.groupby('cid')['target0'].shift(lag)

        df['lag1_1'] = df.groupby('cid')['target1'].shift(lag)

        df['lag0_1'] = df.groupby('cid')['lag0_1'].fillna( method='bfill' )

        df['lag1_1'] = df.groupby('cid')['lag1_1'].fillna( method='bfill' )



        df['m0'] = df.groupby('cid')['lag0_1'].rolling(2).mean().values

        df['m1'] = df.groupby('cid')['lag0_1'].rolling(3).mean().values

        df['m2'] = df.groupby('cid')['lag0_1'].rolling(4).mean().values

        df['m3'] = df.groupby('cid')['lag0_1'].rolling(5).mean().values

        df['m4'] = df.groupby('cid')['lag0_1'].rolling(7).mean().values

        df['m5'] = df.groupby('cid')['lag0_1'].rolling(10).mean().values

        df['m6'] = df.groupby('cid')['lag0_1'].rolling(12).mean().values

        df['m7'] = df.groupby('cid')['lag0_1'].rolling(16).mean().values

        df['m8'] = df.groupby('cid')['lag0_1'].rolling(20).mean().values



        df['n0'] = df.groupby('cid')['lag1_1'].rolling(2).mean().values

        df['n1'] = df.groupby('cid')['lag1_1'].rolling(3).mean().values

        df['n2'] = df.groupby('cid')['lag1_1'].rolling(4).mean().values

        df['n3'] = df.groupby('cid')['lag1_1'].rolling(5).mean().values

        df['n4'] = df.groupby('cid')['lag1_1'].rolling(7).mean().values

        df['n5'] = df.groupby('cid')['lag1_1'].rolling(10).mean().values

        df['n6'] = df.groupby('cid')['lag1_1'].rolling(12).mean().values

        df['n7'] = df.groupby('cid')['lag1_1'].rolling(16).mean().values

        df['n8'] = df.groupby('cid')['lag1_1'].rolling(20).mean().values



        df['m0'] = df.groupby('cid')['m0'].fillna( method='bfill' )

        df['m1'] = df.groupby('cid')['m1'].fillna( method='bfill' )

        df['m2'] = df.groupby('cid')['m2'].fillna( method='bfill' )

        df['m3'] = df.groupby('cid')['m3'].fillna( method='bfill' )

        df['m4'] = df.groupby('cid')['m4'].fillna( method='bfill' )

        df['m5'] = df.groupby('cid')['m5'].fillna( method='bfill' )

        df['m6'] = df.groupby('cid')['m6'].fillna( method='bfill' )

        df['m7'] = df.groupby('cid')['m7'].fillna( method='bfill' )

        df['m8'] = df.groupby('cid')['m8'].fillna( method='bfill' )



        df['n0'] = df.groupby('cid')['n0'].fillna( method='bfill' )

        df['n1'] = df.groupby('cid')['n1'].fillna( method='bfill' )

        df['n2'] = df.groupby('cid')['n2'].fillna( method='bfill' )

        df['n3'] = df.groupby('cid')['n3'].fillna( method='bfill' )

        df['n4'] = df.groupby('cid')['n4'].fillna( method='bfill' )

        df['n5'] = df.groupby('cid')['n5'].fillna( method='bfill' )

        df['n6'] = df.groupby('cid')['n6'].fillna( method='bfill' )

        df['n7'] = df.groupby('cid')['n7'].fillna( method='bfill' )

        df['n8'] = df.groupby('cid')['n8'].fillna( method='bfill' )



        df['flag_China'] = 1*(df['Country_Region'] == 'China')

        df['flag_Italy'] = 1*(df['Country_Region'] == 'Italy')

        df['flag_Spain'] = 1*(df['Country_Region'] == 'Spain')

        df['flag_US']    = 1*(df['Country_Region'] == 'US')

        df['flag_Brazil']= 1*(df['Country_Region'] == 'Brazil')



    #     ohe = OneHotEncoder(sparse=False)

    #     country_ohe = ohe.fit_transform( df[['cid']] )

    #     country_ohe = pd.DataFrame( country_ohe )

    #     country_ohe.columns = df['cid'].unique().tolist()



    #     df = pd.concat( ( df, country_ohe ), axis=1, sort=False )



        tr = df.loc[ df.Date  < traindate ].copy()

        vl = df.loc[ df.Date == traindate ].copy()



        tr = tr.loc[ tr.lag0_1 > 0 ]



        return tr, vl    

    

    def train_period( 

                    train, 

                    valid_days = ['2020-03-13'],

                    lag = 1,

                    seed = 1,

                    ):



        train['target0'] = np.log1p( train['ConfirmedCases'] )

        train['target1'] = np.log1p( train['Fatalities'] )



        param = {

            'subsample': 0.80,

            'colsample_bytree': 0.85,

            'max_depth': 7,

            'gamma': 0.000,

            'learning_rate': 0.01,

            'min_child_weight': 5.00,

            'reg_alpha': 0.000,

            'reg_lambda': 0.400,

            'silent':1,

            'objective':'reg:squarederror',

            #'booster':'dart',

            #'tree_method': 'gpu_hist',

            'nthread': -1,

            'seed': seed,

            }



        tr, vl = create_features( train.copy(), valid_days[0] , lag=lag )

        features = [f for f in tr.columns if f not in [

            'Id',

            'ConfirmedCases',

            'Fatalities',

            'log0',

            'log1',

            'target0',

            'target1',

            'Province_State',

            'Country_Region',

            'Date',

            'ForecastId',

            'cid',

            #'day',

        ] ]



        dtrain = xgb.DMatrix( tr[features], tr['target0'] )

        dvalid = xgb.DMatrix( vl[features], vl['target0'] )

        watchlist = [(dvalid, 'eval')]

        model0 = xgb.train( param, dtrain, 767, watchlist , verbose_eval=0 )#, early_stopping_rounds=25 )



        dtrain = xgb.DMatrix( tr[features], tr['target1'] )

        dvalid = xgb.DMatrix( vl[features], vl['target1'] )

        watchlist = [(dvalid, 'eval')]

        model1 = xgb.train( param, dtrain, 767, watchlist , verbose_eval=0 )#, early_stopping_rounds=25 )



        ypred0 = model0.predict( dvalid )

        ypred1 = model1.predict( dvalid )

        vl['ypred0'] = ypred0

        vl['ypred1'] = ypred1



        #walkforwarding scoring all dates

        feats = ['Province_State','Country_Region','Date']

        for day in valid_days:

            tr, vl = create_features( train.copy(), day, lag=2 )

            dvalid = xgb.DMatrix( vl[features] )

            ypred0 = model0.predict( dvalid )

            ypred1 = model1.predict( dvalid )

            vl['ypred0'] = ypred0

            vl['ypred1'] = ypred1



            train[ 'ypred0' ] = pd.merge( train[feats], vl[feats+['ypred0']], on=feats, how='left' )['ypred0'].values

            train.loc[ train.ypred0<0, 'ypred0'] = 0

            train.loc[ train.ypred0.notnull(), 'target0'] = train.loc[ train.ypred0.notnull() , 'ypred0']



            train[ 'ypred1' ] = pd.merge( train[feats], vl[feats+['ypred1']], on=feats, how='left' )['ypred1'].values

            train.loc[ train.ypred1<0, 'ypred1'] = 0

            train.loc[ train.ypred1.notnull(), 'target1'] = train.loc[ train.ypred1.notnull() , 'ypred1']



            px = np.where( (train.Date==day ) )[0]

            print( day, rmse( train['log0'].iloc[px], train['target0'].iloc[px] ), rmse( train['log1'].iloc[px], train['target1'].iloc[px] )  )



        VALID = train.loc[ (train.Date>=valid_days[0])&(train.Date<=valid_days[-1]) ].copy() 

        del VALID['ypred0'],VALID['ypred1']



        sc0 = rmse( VALID['log0'], VALID['target0'] )

        sc1 = rmse( VALID['log1'], VALID['target1'] )

        print( sc0, sc1, (sc0+sc1)/2 )



        return VALID.copy()    

    

    

    VALID0 = train_period( train, 

                          valid_days = ['2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18','2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31'],

                          lag = 1,

                          seed = 1 )    

    

    VALID1 = train_period( train, 

                          valid_days = ['2020-03-16','2020-03-17','2020-03-18','2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31'],

                          lag = 1,

                          seed = 1 )    

    

    VALID2 = train_period( train, 

                          valid_days = ['2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31'],

                          lag = 1,

                          seed = 1 )    



    VALID3 = train_period( train, 

                          valid_days = ['2020-03-22','2020-03-23','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31'],

                          lag = 1,

                          seed = 1 )   

        

    sa0 = rmse( VALID0['log0'], VALID0['target0'] )

    sa1 = rmse( VALID1['log0'], VALID1['target0'] )

    sa2 = rmse( VALID2['log0'], VALID2['target0'] )

    sa3 = rmse( VALID3['log0'], VALID3['target0'] )



    sb0 = rmse( VALID0['log1'], VALID0['target1'] )

    sb1 = rmse( VALID1['log1'], VALID1['target1'] )

    sb2 = rmse( VALID2['log1'], VALID2['target1'] )

    sb3 = rmse( VALID3['log1'], VALID3['target1'] )





    print('13-31: ' + str(sa0)[:6] + ', ' + str(sb0)[:6] + ' = ' + str(0.5*sa0+0.5*sb0)[:6]  )

    print('16-31: ' + str(sa1)[:6] + ', ' + str(sb1)[:6] + ' = ' + str(0.5*sa1+0.5*sb1)[:6]  )

    print('19-31: ' + str(sa2)[:6] + ', ' + str(sb2)[:6] + ' = ' + str(0.5*sa2+0.5*sb2)[:6]  )

    print('22-31: ' + str(sa3)[:6] + ', ' + str(sb3)[:6] + ' = ' + str(0.5*sa3+0.5*sb3)[:6]  )



    print( 'Avg: ',  (sa0+sb0+sa1+sb1+sa2+sb2+sa3+sb3) / 8 )

    

    TEST  = train_period( train, 

                          valid_days = ['2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10',

                                        '2020-04-11','2020-04-12','2020-04-13','2020-04-14','2020-04-15','2020-04-16','2020-04-17','2020-04-18','2020-04-19','2020-04-20',

                                        '2020-04-21','2020-04-22','2020-04-23','2020-04-24','2020-04-25','2020-04-26','2020-04-27','2020-04-28','2020-04-29','2020-04-30'],

                          lag = 1,

                          seed = 1 )    

    

    VALID2_sub = VALID2.copy()

    VALID2_sub['target0'] = np.log1p( VALID2_sub['ConfirmedCases']  )

    VALID2_sub['target1'] = np.log1p( VALID2_sub['Fatalities']  )

    sub = pd.concat( (VALID2_sub,TEST.loc[ TEST.Date>='2020-04-01' ] ) )



    sub = sub[['ForecastId','target0','target1']]

    sub.columns = ['ForecastId','ConfirmedCases','Fatalities']

    sub['ForecastId'] = sub['ForecastId'].astype( np.int )

    sub['ConfirmedCases'] = np.expm1( sub['ConfirmedCases'] )

    sub['Fatalities'] = np.expm1( sub['Fatalities'] )

    print( sub.describe()  )

    

    # :21 / 22:31

    # :18 / 19:31

    # :15 / 16:31

    # :12 / 13:31

    VALID0.to_csv('fold-13-31.csv', index=False)

    VALID1.to_csv('fold-16-31.csv', index=False)

    VALID2.to_csv('fold-19-31.csv', index=False)

    VALID3.to_csv('fold-22-31.csv', index=False)

    TEST.to_csv('fold-submission.csv', index=False)    

    

    return sub



def get_nn_sub():

    df = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

    sub_df = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")



    coo_df = pd.read_csv("../input/covid19week1/train.csv").rename(columns={"Country/Region": "Country_Region"})

    coo_df = coo_df.groupby("Country_Region")[["Lat", "Long"]].mean().reset_index()

    coo_df = coo_df[coo_df["Country_Region"].notnull()]



    loc_group = ["Province_State", "Country_Region"]





    def preprocess(df):

        df["Date"] = df["Date"].astype("datetime64[ms]")

        df["days"] = (df["Date"] - pd.to_datetime("2020-01-01")).dt.days

        df["weekend"] = df["Date"].dt.dayofweek//5



        df = df.merge(coo_df, how="left", on="Country_Region")

        df["Lat"] = (df["Lat"] // 30).astype(np.float32).fillna(0)

        df["Long"] = (df["Long"] // 60).astype(np.float32).fillna(0)



        for col in loc_group:

            df[col].fillna("none", inplace=True)

        return df



    df = preprocess(df)

    sub_df = preprocess(sub_df)



    print(df.shape)



    TARGETS = ["ConfirmedCases", "Fatalities"]



    for col in TARGETS:

        df[col] = np.log1p(df[col])



    NUM_SHIFT = 5



    features = ["Lat", "Long"]



    for s in range(1, NUM_SHIFT+1):

        for col in TARGETS:

            df["prev_{}_{}".format(col, s)] = df.groupby(loc_group)[col].shift(s)

            features.append("prev_{}_{}".format(col, s))



    df = df[df["Date"] >= df["Date"].min() + timedelta(days=NUM_SHIFT)].copy()



    TEST_FIRST = sub_df["Date"].min() # pd.to_datetime("2020-03-13") #

    TEST_DAYS = (df["Date"].max() - TEST_FIRST).days + 1



    dev_df, test_df = df[df["Date"] < TEST_FIRST].copy(), df[df["Date"] >= TEST_FIRST].copy()



    def nn_block(input_layer, size, dropout_rate, activation):

        out_layer = KL.Dense(size, activation=None)(input_layer)

        #out_layer = KL.BatchNormalization()(out_layer)

        out_layer = KL.Activation(activation)(out_layer)

        out_layer = KL.Dropout(dropout_rate)(out_layer)

        return out_layer





    def get_model():

        inp = KL.Input(shape=(len(features),))



        hidden_layer = nn_block(inp, 64, 0.0, "relu")

        gate_layer = nn_block(hidden_layer, 32, 0.0, "sigmoid")

        hidden_layer = nn_block(hidden_layer, 32, 0.0, "relu")

        hidden_layer = KL.multiply([hidden_layer, gate_layer])



        out = KL.Dense(len(TARGETS), activation="linear")(hidden_layer)



        model = tf.keras.models.Model(inputs=[inp], outputs=out)

        return model



    get_model().summary()



    def get_input(df):

        return [df[features]]



    NUM_MODELS = 10





    def train_models(df, save=False):

        models = []

        for i in range(NUM_MODELS):

            model = get_model()

            model.compile(loss="mean_squared_error", optimizer=Nadam(lr=1e-4))

            hist = model.fit(get_input(df), df[TARGETS],

                             batch_size=2048, epochs=500, verbose=0, shuffle=True)

            if save:

                model.save_weights("model{}.h5".format(i))

            models.append(model)

        return models



    models = train_models(dev_df)





    prev_targets = ['prev_ConfirmedCases_1', 'prev_Fatalities_1']



    def predict_one(df, models):

        pred = np.zeros((df.shape[0], 2))

        for model in models:

            pred += model.predict(get_input(df))/len(models)

        pred = np.maximum(pred, df[prev_targets].values)

        pred[:, 0] = np.log1p(np.expm1(pred[:, 0]) + 0.1)

        pred[:, 1] = np.log1p(np.expm1(pred[:, 1]) + 0.01)

        return np.clip(pred, None, 15)



    print([mean_squared_error(dev_df[TARGETS[i]], predict_one(dev_df, models)[:, i]) for i in range(len(TARGETS))])





    def rmse(y_true, y_pred):

        return np.sqrt(mean_squared_error(y_true, y_pred))



    def evaluate(df):

        error = 0

        for col in TARGETS:

            error += rmse(df[col].values, df["pred_{}".format(col)].values)

        return np.round(error/len(TARGETS), 5)





    def predict(test_df, first_day, num_days, models, val=False):

        temp_df = test_df.loc[test_df["Date"] == first_day].copy()

        y_pred = predict_one(temp_df, models)



        for i, col in enumerate(TARGETS):

            test_df["pred_{}".format(col)] = 0

            test_df.loc[test_df["Date"] == first_day, "pred_{}".format(col)] = y_pred[:, i]



        print(first_day, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())

        if val:

            print(evaluate(test_df[test_df["Date"] == first_day]))





        y_prevs = [None]*NUM_SHIFT



        for i in range(1, NUM_SHIFT):

            y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values



        for d in range(1, num_days):

            date = first_day + timedelta(days=d)

            print(date, np.isnan(y_pred).sum(), y_pred.min(), y_pred.max())



            temp_df = test_df.loc[test_df["Date"] == date].copy()

            temp_df[prev_targets] = y_pred

            for i in range(2, NUM_SHIFT+1):

                temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]] = y_prevs[i-1]



            y_pred, y_prevs = predict_one(temp_df, models), [None, y_pred] + y_prevs[1:-1]





            for i, col in enumerate(TARGETS):

                test_df.loc[test_df["Date"] == date, "pred_{}".format(col)] = y_pred[:, i]



            if val:

                print(evaluate(test_df[test_df["Date"] == date]))



        return test_df



    test_df = predict(test_df, TEST_FIRST, TEST_DAYS, models, val=True)

    print(evaluate(test_df))



    for col in TARGETS:

        test_df[col] = np.expm1(test_df[col])

        test_df["pred_{}".format(col)] = np.expm1(test_df["pred_{}".format(col)])



    models = train_models(df, save=True)



    sub_df_public = sub_df[sub_df["Date"] <= df["Date"].max()].copy()

    sub_df_private = sub_df[sub_df["Date"] > df["Date"].max()].copy()



    pred_cols = ["pred_{}".format(col) for col in TARGETS]

    #sub_df_public = sub_df_public.merge(test_df[["Date"] + loc_group + pred_cols].rename(columns={col: col[5:] for col in pred_cols}), 

    #                                    how="left", on=["Date"] + loc_group)

    sub_df_public = sub_df_public.merge(test_df[["Date"] + loc_group + TARGETS], how="left", on=["Date"] + loc_group)



    SUB_FIRST = sub_df_private["Date"].min()

    SUB_DAYS = (sub_df_private["Date"].max() - sub_df_private["Date"].min()).days + 1



    sub_df_private = df.append(sub_df_private, sort=False)



    for s in range(1, NUM_SHIFT+1):

        for col in TARGETS:

            sub_df_private["prev_{}_{}".format(col, s)] = sub_df_private.groupby(loc_group)[col].shift(s)



    sub_df_private = sub_df_private[sub_df_private["Date"] >= SUB_FIRST].copy()



    sub_df_private = predict(sub_df_private, SUB_FIRST, SUB_DAYS, models)



    for col in TARGETS:

        sub_df_private[col] = np.expm1(sub_df_private["pred_{}".format(col)])



    sub_df = sub_df_public.append(sub_df_private, sort=False)

    sub_df["ForecastId"] = sub_df["ForecastId"].astype(np.int16)



    return sub_df[["ForecastId"] + TARGETS]
sub1 = giba_model()
sub2 = get_nn_sub()
sub1.sort_values("ForecastId", inplace=True)

sub2.sort_values("ForecastId", inplace=True)
from sklearn.metrics import mean_squared_error



TARGETS = ["ConfirmedCases", "Fatalities"]



[np.sqrt(mean_squared_error(np.log1p(sub1[t].values), np.log1p(sub2[t].values))) for t in TARGETS]
sub_df = sub1.copy()

for t in TARGETS:

    sub_df[t] = np.expm1(np.log1p(sub1[t].values)*0.5 + np.log1p(sub2[t].values)*0.5)

    

sub_df.to_csv("submission.csv", index=False)