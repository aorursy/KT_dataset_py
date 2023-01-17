from tensorflow.keras.optimizers import Nadam

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import tensorflow.keras.layers as KL

from datetime import timedelta

import numpy as np

import pandas as pd



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
sub_df = get_nn_sub()



sub_df.to_csv("submission.csv", index=False)