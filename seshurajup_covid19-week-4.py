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





import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge



import datetime

import gc

from tqdm import tqdm
root = '/kaggle/input/covid19-global-forecasting-week-4'



def get_cpmp_sub(save_oof=False, save_public_test=False):

    train = pd.read_csv(f'{root}/train.csv')

    train['Province_State'].fillna('', inplace=True)

    train['Date'] = pd.to_datetime(train['Date'])

    train['day'] = train.Date.dt.dayofyear

    #train = train[train.day <= 85]

    train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]

    train



    test = pd.read_csv(f'{root}/test.csv')

    test['Province_State'].fillna('', inplace=True)

    test['Date'] = pd.to_datetime(test['Date'])

    test['day'] = test.Date.dt.dayofyear

    test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]

    test



    day_min = train['day'].min()

    train['day'] -= day_min

    test['day'] -= day_min



    min_test_val_day = test.day.min()

    max_test_val_day = train.day.max()

    max_test_day = test.day.max()

    num_days = max_test_day + 1



    min_test_val_day, max_test_val_day, num_days



    train['ForecastId'] = -1

    test['Id'] = -1

    test['ConfirmedCases'] = 0

    test['Fatalities'] = 0



    debug = False



    data = pd.concat([train,

                      test[test.day > max_test_val_day][train.columns]

                     ]).reset_index(drop=True)

    if debug:

        data = data[data['geo'] >= 'France_'].reset_index(drop=True)

    #del train, test

    gc.collect()



    dates = data[data['geo'] == 'France_'].Date.values



    if 0:

        gr = data.groupby('geo')

        data['ConfirmedCases'] = gr.ConfirmedCases.transform('cummax')

        data['Fatalities'] = gr.Fatalities.transform('cummax')



    geo_data = data.pivot(index='geo', columns='day', values='ForecastId')

    num_geo = geo_data.shape[0]

    geo_data



    geo_id = {}

    for i,g in enumerate(geo_data.index):

        geo_id[g] = i





    ConfirmedCases = data.pivot(index='geo', columns='day', values='ConfirmedCases')

    Fatalities = data.pivot(index='geo', columns='day', values='Fatalities')



    if debug:

        cases = ConfirmedCases.values

        deaths = Fatalities.values

    else:

        cases = np.log1p(ConfirmedCases.values)

        deaths = np.log1p(Fatalities.values)





    def get_dataset(start_pred, num_train, lag_period):

        days = np.arange( start_pred - num_train + 1, start_pred + 1)

        lag_cases = np.vstack([cases[:, d - lag_period : d] for d in days])

        lag_deaths = np.vstack([deaths[:, d - lag_period : d] for d in days])

        target_cases = np.vstack([cases[:, d : d + 1] for d in days])

        target_deaths = np.vstack([deaths[:, d : d + 1] for d in days])

        geo_ids = np.vstack([geo_ids_base for d in days])

        country_ids = np.vstack([country_ids_base for d in days])

        return lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days



    def update_valid_dataset(data, pred_death, pred_case):

        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data

        day = days[-1] + 1

        new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])

        new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 

        new_target_cases = cases[:, day:day+1]

        new_target_deaths = deaths[:, day:day+1] 

        new_geo_ids = geo_ids  

        new_country_ids = country_ids  

        new_days = 1 + days

        return new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, new_geo_ids, new_country_ids, new_days



    def fit_eval(lr_death, lr_case, data, start_lag_death, end_lag_death, num_lag_case, fit, score):

        lag_cases, lag_deaths, target_cases, target_deaths, geo_ids, country_ids, days = data



        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], country_ids])

        X_death = np.hstack([lag_deaths[:, -num_lag_case:], country_ids])

        X_death = np.hstack([lag_cases[:, -start_lag_death:-end_lag_death], lag_deaths[:, -num_lag_case:], country_ids])

        y_death = target_deaths

        y_death_prev = lag_deaths[:, -1:]

        if fit:

            if 0:

                keep = (y_death > 0).ravel()

                X_death = X_death[keep]

                y_death = y_death[keep]

                y_death_prev = y_death_prev[keep]

            lr_death.fit(X_death, y_death)

        y_pred_death = lr_death.predict(X_death)

        y_pred_death = np.maximum(y_pred_death, y_death_prev)



        X_case = np.hstack([lag_cases[:, -num_lag_case:], geo_ids])

        X_case = lag_cases[:, -num_lag_case:]

        y_case = target_cases

        y_case_prev = lag_cases[:, -1:]

        if fit:

            lr_case.fit(X_case, y_case)

        y_pred_case = lr_case.predict(X_case)

        y_pred_case = np.maximum(y_pred_case, y_case_prev)



        if score:

            death_score = val_score(y_death, y_pred_death)

            case_score = val_score(y_case, y_pred_case)

        else:

            death_score = 0

            case_score = 0



        return death_score, case_score, y_pred_death, y_pred_case



    def train_model(train, valid, start_lag_death, end_lag_death, num_lag_case, num_val, score=True):

        alpha = 3

        lr_death = Ridge(alpha=alpha, fit_intercept=False)

        lr_case = Ridge(alpha=alpha, fit_intercept=True)



        (train_death_score, train_case_score, train_pred_death, train_pred_case,

        ) = fit_eval(lr_death, lr_case, train, start_lag_death, end_lag_death, num_lag_case, fit=True, score=score)



        death_scores = []

        case_scores = []



        death_pred = []

        case_pred = []



        for i in range(num_val):



            (valid_death_score, valid_case_score, valid_pred_death, valid_pred_case,

            ) = fit_eval(lr_death, lr_case, valid, start_lag_death, end_lag_death, num_lag_case, fit=False, score=score)



            death_scores.append(valid_death_score)

            case_scores.append(valid_case_score)

            death_pred.append(valid_pred_death)

            case_pred.append(valid_pred_case)



            if 0:

                print('val death: %0.3f' %  valid_death_score,

                      'val case: %0.3f' %  valid_case_score,

                      'val : %0.3f' %  np.mean([valid_death_score, valid_case_score]),

                      flush=True)

            valid = update_valid_dataset(valid, valid_pred_death, valid_pred_case)



        if score:

            death_scores = np.sqrt(np.mean([s**2 for s in death_scores]))

            case_scores = np.sqrt(np.mean([s**2 for s in case_scores]))

            if 0:

                print('train death: %0.3f' %  train_death_score,

                      'train case: %0.3f' %  train_case_score,

                      'val death: %0.3f' %  death_scores,

                      'val case: %0.3f' %  case_scores,

                      'val : %0.3f' % ( (death_scores + case_scores) / 2),

                      flush=True)

            else:

                print('%0.4f' %  case_scores,

                      ', %0.4f' %  death_scores,

                      '= %0.4f' % ( (death_scores + case_scores) / 2),

                      flush=True)

        death_pred = np.hstack(death_pred)

        case_pred = np.hstack(case_pred)

        return death_scores, case_scores, death_pred, case_pred



    countries = [g.split('_')[0] for g in geo_data.index]

    countries = pd.factorize(countries)[0]



    country_ids_base = countries.reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)

    country_ids_base.shape



    geo_ids_base = np.arange(num_geo).reshape((-1, 1))

    ohe = OneHotEncoder(sparse=False)

    geo_ids_base = 0.1 * ohe.fit_transform(geo_ids_base)

    geo_ids_base.shape



    def val_score(true, pred):

        pred = np.log1p(np.round(np.expm1(pred) - 0.2))

        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))



    def val_score(true, pred):

        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))







    start_lag_death, end_lag_death = 14, 6,

    num_train = 5

    num_lag_case = 14

    lag_period = max(start_lag_death, num_lag_case)



    def get_oof(start_val_delta=0):   

        start_val = min_test_val_day + start_val_delta

        last_train = start_val - 1

        num_val = max_test_val_day - start_val + 1

        print(dates[start_val], start_val, num_val)

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 

                                                            start_lag_death, end_lag_death, num_lag_case, num_val)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases



        sub = train[['Date', 'Id', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        #sub = sub.fillna(0)

        sub = sub[sub.day >= start_val]

        sub = sub[['Id', 'ConfirmedCases', 'Fatalities']].copy()

        return sub





    if save_oof:

        for start_val_delta, date in zip(range(3, -8, -3),

                                  ['2020-03-25', '2020-03-22', '2020-03-19', '2020-03-16']):

            print(date, end=' ')

            oof = get_oof(start_val_delta)

            oof.to_csv('../submissions/cpmp-%s.csv' % date, index=None)



    def get_sub(start_val_delta=0):   

        start_val = min_test_val_day + start_val_delta

        last_train = start_val - 1

        num_val = max_test_val_day - start_val + 1

        print(dates[last_train], start_val, num_val)

        num_lag_case = 14

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        _, _, val_death_preds, val_case_preds = train_model(train_data, valid_data, 

                                                            start_lag_death, end_lag_death, num_lag_case, num_val)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases



        sub = test[['Date', 'ForecastId', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        sub = sub.fillna(0)

        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

        return sub

        return sub





    known_test = train[['geo', 'day', 'ConfirmedCases', 'Fatalities']

              ].merge(test[['geo', 'day', 'ForecastId']], how='left', on=['geo', 'day'])

    known_test = known_test[['ForecastId', 'ConfirmedCases', 'Fatalities']][known_test.ForecastId.notnull()].copy()

    known_test



    unknow_test = test[test.day > max_test_val_day]

    unknow_test



    def get_final_sub():   

        start_val = max_test_val_day + 1

        last_train = start_val - 1

        num_val = max_test_day - start_val + 1

        print(dates[last_train], start_val, num_val)

        num_lag_case = num_val + 3

        train_data = get_dataset(last_train, num_train, lag_period)

        valid_data = get_dataset(start_val, 1, lag_period)

        (_, _, val_death_preds, val_case_preds

        ) = train_model(train_data, valid_data, start_lag_death, end_lag_death, num_lag_case, num_val, score=False)



        pred_deaths = Fatalities.iloc[:, start_val:start_val+num_val].copy()

        pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = ConfirmedCases.iloc[:, start_val:start_val+num_val].copy()

        pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases

        print(unknow_test.shape, pred_deaths.shape, pred_cases.shape)



        sub = unknow_test[['Date', 'ForecastId', 'geo', 'day']]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        #sub = sub.fillna(0)

        sub = sub[['ForecastId', 'ConfirmedCases', 'Fatalities']]

        sub = pd.concat([known_test, sub])

        return sub



    if save_public_test:

        sub = get_sub()

    else:

        sub = get_final_sub()

    return sub
def get_nn_sub():

    df = pd.read_csv(f'{root}/train.csv')

    sub_df = pd.read_csv(f'{root}/test.csv')



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



        hidden_layer = nn_block(inp, 128, 0.0, "relu")

        gate_layer = nn_block(hidden_layer, 64, 0.0, "sigmoid")

        hidden_layer = nn_block(hidden_layer, 64, 0.0, "relu")

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

                             batch_size=2048, epochs=2500, verbose=0, shuffle=True)

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
sub1 = get_cpmp_sub()

sub1['ForecastId'] = sub1['ForecastId'].astype('int')
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