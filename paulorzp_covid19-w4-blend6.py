from tensorflow.keras.optimizers import Nadam

from sklearn.metrics import mean_squared_error

import tensorflow as tf

import tensorflow.keras.layers as KL

from datetime import timedelta

import numpy as np

import pandas as pd

import tensorflow.keras.backend as K



import os

pd.options.display.max_rows = 500

pd.options.display.max_columns = 500



import matplotlib.pyplot as plt

%matplotlib inline



from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.linear_model import LinearRegression, Ridge



import datetime

import gc

from tqdm import tqdm



import xgboost as xgb



import lightgbm as lgb



from sklearn.preprocessing import LabelEncoder



import warnings

warnings.filterwarnings("ignore")



def rmse( yt, yp ):

    return np.sqrt( np.mean( (yt-yp)**2 ) )



class CovidModel:

    def __init__(self):

        pass

    

    def predict_first_day(self, date):

        return None

    

    def predict_next_day(self, yesterday_pred_df):

        return None





class CovidModelAhmet(CovidModel):

    def preprocess(self, df, meta_df):

        df["Date"] = pd.to_datetime(df['Date'])



        df = df.merge(meta_df, on=self.loc_group, how="left")

        df["lat"] = (df["lat"] // 30).astype(np.float32).fillna(0)

        df["lon"] = (df["lon"] // 60).astype(np.float32).fillna(0)



        df["population"] = np.log1p(df["population"]).fillna(-1)

        df["area"] = np.log1p(df["area"]).fillna(-1)



        for col in self.loc_group:

            df[col].fillna("", inplace=True)

            

        df['day'] = df.Date.dt.dayofyear

        df['geo'] = ['_'.join(x) for x in zip(df['Country_Region'], df['Province_State'])]

        return df



    def get_model(self):

        

        def nn_block(input_layer, size, dropout_rate, activation):

            out_layer = KL.Dense(size, activation=None)(input_layer)

            out_layer = KL.Activation(activation)(out_layer)

            out_layer = KL.Dropout(dropout_rate)(out_layer)

            return out_layer

    

        ts_inp = KL.Input(shape=(len(self.ts_features),))

        global_inp = KL.Input(shape=(len(self.global_features),))



        inp = KL.concatenate([global_inp, ts_inp])

        hidden_layer = nn_block(inp, 64, 0.0, "relu")

        gate_layer = nn_block(hidden_layer, 32, 0.0, "sigmoid")

        hidden_layer = nn_block(hidden_layer, 32, 0.0, "relu")

        hidden_layer = KL.multiply([hidden_layer, gate_layer])



        out = KL.Dense(len(self.TARGETS), activation="linear")(hidden_layer)



        model = tf.keras.models.Model(inputs=[global_inp, ts_inp], outputs=out)

        return model

    

    def get_input(self, df):

        return [df[self.global_features], df[self.ts_features]]

        

    def train_models(self, df, num_models=20, save=False):

        

        def custom_loss(y_true, y_pred):

            return K.sum(K.sqrt(K.sum(K.square(y_true - y_pred), axis=0, keepdims=True)))/len(self.TARGETS)

    

        models = []

        for i in range(num_models):

            model = self.get_model()

            model.compile(loss=custom_loss, optimizer=Nadam(lr=1e-4))

            hist = model.fit(self.get_input(df), df[self.TARGETS],

                             batch_size=2048, epochs=200, verbose=0, shuffle=True)

            if save:

                model.save_weights("model{}.h5".format(i))

            models.append(model)

        return models

    

    

    def predict_one(self, df):

        

        pred = np.zeros((df.shape[0], 2))

        for model in self.models:

            pred += model.predict(self.get_input(df))/len(self.models)

        pred = np.maximum(pred, df[self.prev_targets].values)

        pred[:, 0] = np.log1p(np.expm1(pred[:, 0]) + 0.1)

        pred[:, 1] = np.log1p(np.expm1(pred[:, 1]) + 0.01)

        return np.clip(pred, None, 15)

    



    def __init__(self):

        df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

        sub_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")



        meta_df = pd.read_csv("../input/covid19-forecasting-metadata/region_metadata.csv")



        self.loc_group = ["Province_State", "Country_Region"]



        df = self.preprocess(df, meta_df)

        sub_df = self.preprocess(sub_df, meta_df)

        

        df = df.merge(sub_df[["ForecastId", "Date", "geo"]], how="left", on=["Date", "geo"])

        df = df.append(sub_df[sub_df["Date"] > df["Date"].max()], sort=False)

        

        df["day"] = df["day"] - df["day"].min()



        self.TARGETS = ["ConfirmedCases", "Fatalities"]

        self.prev_targets = ['prev_ConfirmedCases_1', 'prev_Fatalities_1']



        for col in self.TARGETS:

            df[col] = np.log1p(df[col])



        self.NUM_SHIFT = 7



        self.global_features = ["lat", "lon", "population", "area"]

        self.ts_features = []



        for s in range(1, self.NUM_SHIFT+1):

            for col in self.TARGETS:

                df["prev_{}_{}".format(col, s)] = df.groupby(self.loc_group)[col].shift(s)

                self.ts_features.append("prev_{}_{}".format(col, s))



        self.df = df[df["Date"] >= df["Date"].min() + timedelta(days=self.NUM_SHIFT)].copy()



        

    def predict_first_day(self, day):

        self.models = self.train_models(self.df[self.df["day"] < day])

        

        temp_df = self.df.loc[self.df["day"] == day].copy()

        y_pred = self.predict_one(temp_df)

            

        self.y_prevs = [None]*self.NUM_SHIFT



        for i in range(1, self.NUM_SHIFT):

            self.y_prevs[i] = temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]].values

            

        temp_df[self.TARGETS] = y_pred

        self.day = day

        return temp_df[["geo", "day"] + self.TARGETS]

    

    

    def predict_next_day(self, yesterday_pred_df):

        self.day = self.day + 1



        temp_df = self.df.loc[self.df["day"] == self.day].copy()

        

        yesterday_pred_df = temp_df[["geo"]].merge(yesterday_pred_df[["geo"] + self.TARGETS], on="geo", how="left")

        temp_df[self.prev_targets] = yesterday_pred_df[self.TARGETS].values



        for i in range(2, self.NUM_SHIFT+1):

            temp_df[['prev_ConfirmedCases_{}'.format(i), 'prev_Fatalities_{}'.format(i)]] = self.y_prevs[i-1]



        y_pred, self.y_prevs = self.predict_one(temp_df), [None, temp_df[self.prev_targets].values] + self.y_prevs[1:-1]



        temp_df[self.TARGETS] = y_pred

        return temp_df[["geo", "day"] + self.TARGETS]

class CovidModelCPMP(CovidModel):

    

    def __init__(self):

        train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

        train['Province_State'].fillna('', inplace=True)

        train['Date'] = pd.to_datetime(train['Date'])

        train['day'] = train.Date.dt.dayofyear

        train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]

        test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

        test['Province_State'].fillna('', inplace=True)

        test['Date'] = pd.to_datetime(test['Date'])

        test['day'] = test.Date.dt.dayofyear

        test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]

        day_min = train['day'].min()

        train['day'] -= day_min

        test['day'] -= day_min  

        self.min_test_val_day = test.day.min()

        self.max_test_val_day = train.day.max()

        self.max_test_day = test.day.max()



        train['ForecastId'] = -1

        test['Id'] = -1

        test['ConfirmedCases'] = 0

        test['Fatalities'] = 0    

        data = pd.concat([train,

                  test[test.day > self.max_test_val_day][train.columns]

                 ]).reset_index(drop=True)

        self.data = data

        self.train = train

        self.test = test

        self.dates = data[data['geo'] == 'France_'].Date.values

        region_meta = pd.read_csv('../input/covid19-forecasting-metadata/region_metadata.csv')

        region_meta['Province_State'].fillna('', inplace=True)

        region_meta['geo'] = ['_'.join(x) for x in zip(region_meta['Country_Region'], region_meta['Province_State'], )]

        population = data[['geo']].merge(region_meta, how='left', on='geo').fillna(0)

        population = population.groupby('geo')[['population']].first()

        population['population'] = np.log1p(population['population'])

        self.population = population[['population']].values

        continents = region_meta['continent']

        continents = pd.factorize(continents)[0]

        continents_ids_base = continents.reshape((-1, 1))

        ohe = OneHotEncoder(sparse=False)

        self.continents_ids_base = ohe.fit_transform(continents_ids_base)

        

        self.geo_data = data.pivot(index='geo', columns='day', values='ForecastId')

        self.num_geo = self.geo_data.shape[0]

        self.ConfirmedCases = data.pivot(index='geo', columns='day', values='ConfirmedCases')

        self.Fatalities = data.pivot(index='geo', columns='day', values='Fatalities')

        self.cases = np.log1p(self.ConfirmedCases.values)

        self.deaths = np.log1p(self.Fatalities.values)

        self.case_threshold = 30

        

        self.c_case = 10

        self.t_case = 100

        self.c_death = 10

        self.t_death = 5



        time_cases = self.c_case * (self.cases >= np.log1p(self.t_case)) 

        time_cases = np.cumsum(time_cases, axis=1)

        self.time_cases = 1 * np.log1p(time_cases) 



        time_deaths = self.c_death * (self.deaths >= np.log1p(self.t_death))

        time_deaths = np.cumsum(time_deaths, axis=1)

        self.time_deaths = 1 *np.log1p(time_deaths) 



        countries = [g.split('_')[0] for g in self.geo_data.index]

        countries = pd.factorize(countries)[0]

        country_ids_base = countries.reshape((-1, 1))

        ohe = OneHotEncoder(sparse=False)

        self.country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)



        self.start_lag_death = 13

        self.end_lag_death = 5

        self.num_train = 5

        self.num_lag_case = 14

        self.lag_period = max(self.start_lag_death, self.num_lag_case)

        

        # For tetsing purpose       

        self.df = train[['geo', 'day', 'ConfirmedCases', 'Fatalities']].copy()

        self.df.ConfirmedCases = np.log1p(self.df.ConfirmedCases)

        self.df.Fatalities = np.log1p(self.df.Fatalities)

        

    def get_country_ids(self):

        countries = [g.split('_')[0] for g in self.geo_data.index]

        countries = pd.factorize(countries)[0]

        countries[self.cases[:, :self.last_train+1].max(axis=1) < np.log1p(self.case_threshold)] = -1

        countries = pd.factorize(countries)[0]





        country_ids_base = countries.reshape((-1, 1))

        ohe = OneHotEncoder(sparse=False)

        country_ids_base = 0.2 * ohe.fit_transform(country_ids_base)

        return country_ids_base

    

    def val_score(self, true, pred):

        return np.sqrt(mean_squared_error(true.ravel(), pred.ravel()))

    

    def get_dataset(self, start_pred, num_train):

        days = np.arange( start_pred - num_train + 1, start_pred + 1)

        lag_cases = np.vstack([self.cases[:, d - self.lag_period : d] for d in days])

        lag_deaths = np.vstack([self.deaths[:, d - self.lag_period : d] for d in days])

        target_cases = np.vstack([self.cases[:, d : d + 1] for d in days])

        target_deaths = np.vstack([self.deaths[:, d : d + 1] for d in days])

        continents_ids = np.vstack([self.continents_ids_base for d in days])

        country_ids = np.vstack([self.country_ids_base for d in days])

        population = np.vstack([self.population for d in days])

        time_case = np.vstack([self.time_cases[:, d - 1: d ] for d in days])

        time_death = np.vstack([self.time_deaths[:, d - 1 : d ] for d in days])

        return (lag_cases, lag_deaths, target_cases, target_deaths, 

            continents_ids, country_ids, population, time_case, time_death, days)

    

    def update_time(self, time_death, time_case, pred_death, pred_case):

        new_time_death = np.expm1(time_death) + self.c_death * (pred_death >= np.log1p(self.t_death))

        new_time_death = 1 *np.log1p(new_time_death) 

        new_time_case = np.expm1(time_case) + self.c_case * (pred_case >= np.log1p(self.t_case))

        new_time_case = 1 *np.log1p(new_time_case) 

        return new_time_death, new_time_case



    def update_valid_dataset(self, dataset, pred_death, pred_case, pred_day):

        (lag_cases, lag_deaths, target_cases, target_deaths, 

         continents_ids, country_ids, population, time_case, time_death, days) = dataset

        if pred_day != days[-1]:

            print('error', pred_day, days[-1])

            return None

        day = days[-1] + 1

        new_lag_cases = np.hstack([lag_cases[:, 1:], pred_case])

        new_lag_deaths = np.hstack([lag_deaths[:, 1:], pred_death]) 

        new_target_cases = self.cases[:, day:day+1]

        new_target_deaths = self.deaths[:, day:day+1] 

        new_continents_ids = continents_ids  

        new_country_ids = country_ids  

        new_population = population  

        new_time_death, new_time_case = self.update_time(time_death, time_case, pred_death, pred_case)

        new_days = 1 + days

        return (new_lag_cases, new_lag_deaths, new_target_cases, new_target_deaths, 

            new_continents_ids, new_country_ids, new_population, 

                new_time_case, new_time_death, new_days)

        

    def fit_eval(self, dataset, fit):

        (lag_cases, lag_deaths, target_cases, target_deaths, 

         continents_ids, country_ids, population, 

         time_case, time_death, days) = dataset



        X_death = np.hstack([lag_cases[:, -self.start_lag_death:-self.end_lag_death], 

                             lag_deaths[:, -self.num_lag_case:], 

                             country_ids,

                             continents_ids,

                              population,

                             time_case,

                             time_death,

                            ])

        y_death = target_deaths

        y_death_prev = lag_deaths[:, -1:]

        if fit:

             self.lr_death.fit(X_death, y_death)

        y_pred_death = self.lr_death.predict(X_death)

        y_pred_death = np.maximum(y_pred_death, y_death_prev)



        X_case = np.hstack([lag_cases[:, -self.num_lag_case:], 

                            country_ids, 

                            continents_ids,

                            population,

                             time_case,

                             #time_death,

                           ])

        y_case = target_cases

        y_case_prev = lag_cases[:, -1:]

        if fit:

            self.lr_case.fit(X_case, y_case)

        y_pred_case = self.lr_case.predict(X_case)

        y_pred_case = np.maximum(y_pred_case, y_case_prev)



        return y_pred_death, y_pred_case

     

    def get_pred_df(self, val_death_preds, val_case_preds, ):

        pred_deaths = self.Fatalities.iloc[:, self.start_val:self.start_val+self.num_val].copy()

        #pred_deaths.iloc[:, :] = np.expm1(val_death_preds)

        pred_deaths.iloc[:, :] = val_death_preds

        pred_deaths = pred_deaths.stack().reset_index()

        pred_deaths.columns = ['geo', 'day', 'Fatalities']

        pred_deaths



        pred_cases = self.ConfirmedCases.iloc[:, self.start_val:self.start_val+self.num_val].copy()

        #pred_cases.iloc[:, :] = np.expm1(val_case_preds)

        pred_cases.iloc[:, :] = val_case_preds

        pred_cases = pred_cases.stack().reset_index()

        pred_cases.columns = ['geo', 'day', 'ConfirmedCases']

        pred_cases



        sub = self.data[['geo', 'day']]

        sub = sub[sub.day == self.start_val]

        sub = sub.merge(pred_cases, how='left', on=['geo', 'day'])

        sub = sub.merge(pred_deaths, how='left', on=['geo', 'day'])

        sub = sub[(sub.day >= self.start_val) & (sub.day <= self.end_val)]

        return sub

    

    def predict_first_day(self, day):

        self.start_val = day

        self.end_val = day + 1

        self.num_val = self.end_val - self.start_val + 1

        score = True

        self.last_train = self.start_val - 1

        print(self.dates[self.last_train], self.start_val, self.num_val)

        self.country_ids_base = self.get_country_ids()

        train_data = self.get_dataset(self.last_train, self.num_train)

        alpha = 3

        self.lr_death = Ridge(alpha=alpha, fit_intercept=True)

        self.lr_case = Ridge(alpha=alpha, fit_intercept=True)

        _ = self.fit_eval(train_data, fit=True)

        

        self.valid_data = self.get_dataset(self.start_val, 1)

        val_death_preds, val_case_preds = self.fit_eval(self.valid_data, fit=False)

        df = self.get_pred_df(val_death_preds, val_case_preds)

        return df

    

    def predict_next_day(self, yesterday_pred_df):

        yesterday_pred_df = yesterday_pred_df.sort_values(by='geo').reset_index(drop=True)

        if yesterday_pred_df.day.nunique() != 1:

            print('error', yesterday_pred_df.day.unique())

            return None

        pred_death = yesterday_pred_df[['Fatalities']].values

        pred_case = yesterday_pred_df[['ConfirmedCases']].values

        pred_day = yesterday_pred_df.day.unique()[0]

        

        new_valid_data = self. update_valid_dataset(self.valid_data, 

                                                    pred_death, pred_case, pred_day)

        if len(new_valid_data) > 0:

            self.valid_data = new_valid_data

        self.start_val = pred_day + 1

        self.end_val = pred_day + 2

        val_death_preds, val_case_preds = self.fit_eval(self.valid_data, fit=False)

        df = self.get_pred_df(val_death_preds, val_case_preds)

         

        return df
class CovidModel:

    def __init__(self):

        pass

    

    def predict_first_day(self, date):

        return None

    

    def predict_next_day(self, yesterday_pred_df):

        return None

    



class CovidModelGIBA(CovidModel):

    def __init__(self, lag=1, seed=1 ):



        self.lag  = lag

        self.seed = seed

        print( 'Lag:', lag, 'Seed:', seed )

        

        train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

        train['Date'] = pd.to_datetime( train['Date'] )

        self.maxdate  = str(train['Date'].max())[:10]

        self.testdate = str( train['Date'].max() + pd.Timedelta(days=1) )[:10]

        print( 'Last Date in Train:',self.maxdate, 'Test first Date:',self.testdate )

        train['Province_State'].fillna('', inplace=True)

        train['day'] = train.Date.dt.dayofyear

        self.day_min = train['day'].min()

        train['day'] -= self.day_min

        train['geo'] = ['_'.join(x) for x in zip(train['Country_Region'], train['Province_State'])]



        test  = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

        test['Date'] = pd.to_datetime( test['Date'] )

        test['Province_State'].fillna('', inplace=True)

        test['day'] = test.Date.dt.dayofyear

        test['day'] -= self.day_min

        test['geo'] = ['_'.join(x) for x in zip(test['Country_Region'], test['Province_State'])]

        test['Id'] = -1

        test['ConfirmedCases'] = 0

        test['Fatalities'] = 0



        self.trainmaxday  = train['day'].max()

        self.testday1 = train['day'].max() + 1

        self.testdayN = test['day'].max()

        

        publictest = test.loc[ test.Date > train.Date.max() ].copy()

        train = pd.concat( (train, publictest ), sort=False )

        train.sort_values( ['Country_Region','Province_State','Date'], inplace=True )

        train = train.reset_index(drop=True)



        train['ForecastId'] = pd.merge( train, test, on=['Country_Region','Province_State','Date'], how='left' )['ForecastId_y'].values



        train['cid'] = train['Country_Region'] + '_' + train['Province_State']



        train['log0'] = np.log1p( train['ConfirmedCases'] )

        train['log1'] = np.log1p( train['Fatalities'] )



        train = train.loc[ (train.log0 > 0) | (train.ForecastId.notnull()) | (train.Date >= '2020-03-17') ].copy()

        train = train.reset_index(drop=True)



        train['days_since_1case'] = train.groupby('cid')['Id'].cumcount()



        dt = pd.read_csv('../input/covid19-lockdown-dates-by-country/countryLockdowndates.csv')

        dt.columns = ['Country_Region','Province_State','Date','Type','Reference']

        dt = dt.loc[ dt.Date == dt.Date ]

        dt['Province_State'] = dt['Province_State'].fillna('')

        dt['Date'] = pd.to_datetime( dt['Date'] )

        dt['Date'] = dt['Date'] + pd.Timedelta(days=8)

        dt['Type'] = pd.factorize( dt['Type'] )[0]

        dt['cid'] = dt['Country_Region'] + '_' + dt['Province_State']

        del dt['Reference'], dt['Country_Region'], dt['Province_State']

        train = pd.merge( train, dt, on=['cid','Date'], how='left' )

        train['Type'] = train.groupby('cid')['Type'].fillna( method='ffill' )



        train['target0'] = np.log1p( train['ConfirmedCases'] )

        train['target1'] = np.log1p( train['Fatalities'] )

        # dt = pd.read_csv('../input/covid19-country-data-wk3-release/Data Join - RELEASE.csv')

        # dt['Province_State'] = dt['Province_State'].fillna('')

        # dt['Country_Region'] = dt['Country_Region'].fillna('')

        # train = pd.merge( train, dt, on=['Country_Region','Province_State'], how='left' )

        # #Fix



        #print( train.head(4) )

        #print( train.shape ) 

        

        self.train = train.copy()

   

    

    def create_features( self, df, valid_day ):



        #df = df.loc[ df.day<=valid_day ].copy()

        df = df.loc[ df.day>=(valid_day-50) ].copy()

        

        df['lag0_1'] = df.groupby('cid')['target0'].shift(self.lag)

        df['lag0_1'] = df.groupby('cid')['lag0_1'].fillna( method='bfill' )



        df['lag0_8'] = df.groupby('cid')['target0'].shift(8)

        df['lag0_8'] = df.groupby('cid')['lag0_8'].fillna( method='bfill' )

        

        df['lag1_1'] = df.groupby('cid')['target1'].shift(self.lag)

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

        df['m9'] = df.groupby('cid')['lag0_1'].rolling(25).mean().values



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

        df['m9'] = df.groupby('cid')['m9'].fillna( method='bfill' )



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

        #df['flag_Italy'] = 1*(df['Country_Region'] == 'Italy')

        #df['flag_Spain'] = 1*(df['Country_Region'] == 'Spain')

        df['flag_US']    = 1*(df['Country_Region'] == 'US')

        #df['flag_Brazil']= 1*(df['Country_Region'] == 'Brazil')

        

        df['flag_Kosovo_']   = 1*(df['cid'] == 'Kosovo_')

        df['flag_Korea']     = 1*(df['cid'] == 'Korea, South_')

        df['flag_Nepal_']    = 1*(df['cid'] == 'Nepal_')

        df['flag_Holy See_'] = 1*(df['cid'] == 'Holy See_')

        df['flag_Suriname_'] = 1*(df['cid'] == 'Suriname_')

        df['flag_Ghana_']    = 1*(df['cid'] == 'Ghana_')

        df['flag_Togo_']     = 1*(df['cid'] == 'Togo_')

        df['flag_Malaysia_'] = 1*(df['cid'] == 'Malaysia_')

        df['flag_US_Rhode']  = 1*(df['cid'] == 'US_Rhode Island')

        df['flag_Bolivia_']  = 1*(df['cid'] == 'Bolivia_')

        df['flag_China_Tib'] = 1*(df['cid'] == 'China_Tibet')

        df['flag_Bahrain_']  = 1*(df['cid'] == 'Bahrain_')

        df['flag_Honduras_'] = 1*(df['cid'] == 'Honduras_')

        df['flag_Bangladesh']= 1*(df['cid'] == 'Bangladesh_')

        df['flag_Paraguay_'] = 1*(df['cid'] == 'Paraguay_')



        tr = df.loc[ df.day  < valid_day ].copy()

        vl = df.loc[ df.day == valid_day ].copy()



        tr = tr.loc[ tr.lag0_1 > 0 ].copy()



        maptarget0 = tr.groupby('cid')['target0'].agg( log0_max='max' ).reset_index()

        maptarget1 = tr.groupby('cid')['target1'].agg( log1_max='max' ).reset_index()

        vl['log0_max'] = pd.merge( vl, maptarget0, on='cid' , how='left' )['log0_max'].values

        vl['log1_max'] = pd.merge( vl, maptarget1, on='cid' , how='left' )['log1_max'].values

        vl['log0_max'] = vl['log0_max'].fillna(0)

        vl['log1_max'] = vl['log1_max'].fillna(0)



        return tr, vl

    



    def train_models(self, valid_day = 10 ):



        train = self.train.copy()



        #Fix some anomalities:

        train.loc[ (train.cid=='China_Guizhou') & (train.Date=='2020-03-17') , 'target0' ] = np.log1p( 146 )

        train.loc[ (train.cid=='Guyana_')&(train.Date>='2020-03-22')&(train.Date<='2020-03-30') , 'target0' ] = np.log1p( 12 )

        train.loc[ (train.cid=='US_Virgin Islands')&(train.Date>='2020-03-29')&(train.Date<='2020-03-29') , 'target0' ] = np.log1p( 24 )

        train.loc[ (train.cid=='US_Virgin Islands')&(train.Date>='2020-03-30')&(train.Date<='2020-03-30') , 'target0' ] = np.log1p( 27 )



        train.loc[ (train.cid=='Iceland_')&(train.Date>='2020-03-15')&(train.Date<='2020-03-15') , 'target1' ] = np.log1p( 0 )

        train.loc[ (train.cid=='Kazakhstan_')&(train.Date>='2020-03-20')&(train.Date<='2020-03-20') , 'target1' ] = np.log1p( 0 )

        train.loc[ (train.cid=='Serbia_')&(train.Date>='2020-03-26')&(train.Date<='2020-03-26') , 'target1' ] = np.log1p( 5 )

        train.loc[ (train.cid=='Serbia_')&(train.Date>='2020-03-27')&(train.Date<='2020-03-27') , 'target1' ] = np.log1p( 6 )

        train.loc[ (train.cid=='Slovakia_')&(train.Date>='2020-03-22')&(train.Date<='2020-03-31') , 'target1' ] = np.log1p( 1 )

        train.loc[ (train.cid=='US_Hawaii')&(train.Date>='2020-03-25')&(train.Date<='2020-03-31') , 'target1' ] = np.log1p( 1 )



        param = {

            'subsample': 1.000,

            'colsample_bytree': 0.85,

            'max_depth': 6,

            'gamma': 0.000,

            'learning_rate': 0.010,

            'min_child_weight': 5.00,

            'reg_alpha': 0.000,

            'reg_lambda': 0.400,

            'silent':1,

            'objective':'reg:squarederror',

            #'booster':'dart',

            #'tree_method': 'gpu_hist',

            'nthread': 12,#-1,

            'seed': self.seed

            }    

        

        tr, vl = self.create_features( train.copy(), valid_day )

        #Features for Cases

        features = [f for f in tr.columns if f not in [

            #'flag_China','flag_US',

            #'flag_Kosovo_','flag_Korea','flag_Nepal_','flag_Holy See_','flag_Suriname_','flag_Ghana_','flag_Togo_','flag_Malaysia_','flag_US_Rhode','flag_Bolivia_','flag_China_Tib','flag_Bahrain_','flag_Honduras_','flag_Bangladesh','flag_Paraguay_',

            'lag0_8',

            'Id','ConfirmedCases','Fatalities','log0','log1','target0','target1','ypred0','ypred1','Province_State','Country_Region','Date','ForecastId','cid','geo','day',

            'GDP_region','TRUE POPULATION','pct_in_largest_city',' TFR ',' Avg_age ','latitude','longitude','abs_latitude','temperature', 'humidity',

            'Personality_pdi','Personality_idv','Personality_mas','Personality_uai','Personality_ltowvs','Personality_assertive','personality_perform','personality_agreeableness',

            'murder','High_rises','max_high_rises','AIR_CITIES','AIR_AVG','continent_gdp_pc','continent_happiness','continent_generosity','continent_corruption','continent_Life_expectancy'        

        ] ]

        self.features0 = features

        #Features for Fatalities

        features = [f for f in tr.columns if f not in [

            'm0','m1','m2','m3',

            #'flag_China','flag_US',

            #'flag_Kosovo_','flag_Korea','flag_Nepal_','flag_Holy See_','flag_Suriname_','flag_Ghana_','flag_Togo_','flag_Malaysia_','flag_US_Rhode','flag_Bolivia_','flag_China_Tib','flag_Bahrain_','flag_Honduras_','flag_Bangladesh','flag_Paraguay_',

            'Id','ConfirmedCases','Fatalities','log0','log1','target0','target1','ypred0','ypred1','Province_State','Country_Region','Date','ForecastId','cid','geo','day',

            'GDP_region','TRUE POPULATION','pct_in_largest_city',' TFR ',' Avg_age ','latitude','longitude','abs_latitude','temperature', 'humidity',

            'Personality_pdi','Personality_idv','Personality_mas','Personality_uai','Personality_ltowvs','Personality_assertive','personality_perform','personality_agreeableness',

            'murder','High_rises','max_high_rises','AIR_CITIES','AIR_AVG','continent_gdp_pc','continent_happiness','continent_generosity','continent_corruption','continent_Life_expectancy'        

        ] ]

        self.features1 = features

        



        nrounds0 = 680

        nrounds1 = 630

         #lag 1###############################################################

        dtrain = xgb.DMatrix( tr[self.features0], tr['target0'] )

        param['seed'] = self.seed

        self.model0 = xgb.train( param, dtrain, nrounds0, verbose_eval=0 )

        param['seed'] = self.seed+1

        self.model1 = xgb.train( param, dtrain, nrounds0, verbose_eval=0 )

        

        dtrain = xgb.DMatrix( tr[self.features1], tr['target1'] )

        param['seed'] = self.seed

        self.model2 = xgb.train( param, dtrain, nrounds1, verbose_eval=0 ) 

        param['seed'] = self.seed+1

        self.model3 = xgb.train( param, dtrain, nrounds1, verbose_eval=0 )

        

        self.vl = vl

        

        return 1

    

        

    def predict_first_day(self, day ):

        

        self.day = day

        self.train_models( day )

        

        dvalid = xgb.DMatrix( self.vl[self.features0] )

        ypred0 = ( self.model0.predict( dvalid ) + self.model1.predict( dvalid )  ) / 2

        dvalid = xgb.DMatrix( self.vl[self.features1] )

        ypred1 = ( self.model2.predict( dvalid ) + self.model3.predict( dvalid )  ) / 2

        

        self.vl['ypred0'] = ypred0

        self.vl['ypred1'] = ypred1

        self.vl.loc[ self.vl.ypred0<self.vl.log0_max, 'ypred0'] =  self.vl.loc[ self.vl.ypred0<self.vl.log0_max, 'log0_max']

        self.vl.loc[ self.vl.ypred1<self.vl.log1_max, 'ypred1'] =  self.vl.loc[ self.vl.ypred1<self.vl.log1_max, 'log1_max']

        

        VALID = self.vl[["geo", "day", 'ypred0', 'ypred1']].copy()

        VALID.columns = ["geo", "day", 'ConfirmedCases', 'Fatalities']        

        return VALID.reset_index(drop=True)

    

    

    def predict_next_day(self, yesterday ):



        self.day += 1

        

        feats = ['geo','day']        

        self.train[ 'ypred0' ] = pd.merge( self.train[feats], yesterday[feats+['ConfirmedCases']], on=feats, how='left' )['ConfirmedCases'].values

        self.train.loc[ self.train.ypred0.notnull(), 'target0'] = self.train.loc[ self.train.ypred0.notnull() , 'ypred0']



        self.train[ 'ypred1' ] = pd.merge( self.train[feats], yesterday[feats+['Fatalities']], on=feats, how='left' )['Fatalities'].values

        self.train.loc[ self.train.ypred1.notnull(), 'target1'] = self.train.loc[ self.train.ypred1.notnull() , 'ypred1']

        del self.train['ypred0'], self.train['ypred1']

        

        tr, vl = self.create_features( self.train.copy(), self.day )        

        dvalid = xgb.DMatrix( vl[self.features0] )

        ypred0 = (self.model0.predict( dvalid ) + self.model1.predict( dvalid ) )/2

        dvalid = xgb.DMatrix( vl[self.features1] )

        ypred1 = (self.model2.predict( dvalid ) + self.model3.predict( dvalid ) )/2

    

        vl['ypred0'] = ypred0

        vl['ypred1'] = ypred1

        vl.loc[ vl.ypred0<vl.log0_max, 'ypred0'] =  vl.loc[ vl.ypred0<vl.log0_max, 'log0_max']

        vl.loc[ vl.ypred1<vl.log1_max, 'ypred1'] =  vl.loc[ vl.ypred1<vl.log1_max, 'log1_max']

        

        self.vl = vl

        VALID = vl[["geo", "day", 'ypred0', 'ypred1']].copy()

        VALID.columns = ["geo", "day", 'ConfirmedCases', 'Fatalities']        

        return VALID.reset_index(drop=True)
## defining constants

PATH_TRAIN = "/kaggle/input/covid19-global-forecasting-week-4/train.csv"

PATH_TEST = "/kaggle/input/covid19-global-forecasting-week-4/test.csv"



PATH_SUBMISSION = "submission.csv"

PATH_OUTPUT = "output.csv"



PATH_REGION_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_metadata.csv"

PATH_REGION_DATE_METADATA = "/kaggle/input/covid19-forecasting-metadata/region_date_metadata.csv"



VAL_DAYS = 7

MAD_FACTOR = 0.5

DAYS_SINCE_CASES = [1, 10, 50, 100, 500, 1000, 5000, 10000]



SEED = 2357



LGB_PARAMS = {"objective": "regression",

              "num_leaves": 5,

              "learning_rate": 0.013,

              "bagging_fraction": 0.91,

              "feature_fraction": 0.81,

              "reg_alpha": 0.13,

              "reg_lambda": 0.13,

              "metric": "rmse",

              "seed": SEED

             }
## reading data

train = pd.read_csv(PATH_TRAIN)

test = pd.read_csv(PATH_TEST)



region_metadata = pd.read_csv(PATH_REGION_METADATA)

region_date_metadata = pd.read_csv(PATH_REGION_DATE_METADATA)
## preparing data

train = train.merge(test[["ForecastId", "Province_State", "Country_Region", "Date"]], on = ["Province_State", "Country_Region", "Date"], how = "left")

test = test[~test.Date.isin(train.Date.unique())]



df_panel = pd.concat([train, test], sort = False)



# combining state and country into 'geography'

df_panel["geography"] = df_panel.Country_Region.astype(str) + ": " + df_panel.Province_State.astype(str)

df_panel.loc[df_panel.Province_State.isna(), "geography"] = df_panel[df_panel.Province_State.isna()].Country_Region



# fixing data issues with cummax

df_panel.ConfirmedCases = df_panel.groupby("geography")["ConfirmedCases"].cummax()

df_panel.Fatalities = df_panel.groupby("geography")["Fatalities"].cummax()



# merging external metadata

df_panel = df_panel.merge(region_metadata, on = ["Country_Region", "Province_State"])

df_panel = df_panel.merge(region_date_metadata, on = ["Country_Region", "Province_State", "Date"], how = "left")



# label encoding continent

df_panel.continent = LabelEncoder().fit_transform(df_panel.continent)

df_panel.Date = pd.to_datetime(df_panel.Date, format = "%Y-%m-%d")



df_panel.sort_values(["geography", "Date"], inplace = True)
## feature engineering

min_date_train = np.min(df_panel[~df_panel.Id.isna()].Date)

max_date_train = np.max(df_panel[~df_panel.Id.isna()].Date)



min_date_test = np.min(df_panel[~df_panel.ForecastId.isna()].Date)

max_date_test = np.max(df_panel[~df_panel.ForecastId.isna()].Date)



n_dates_test = len(df_panel[~df_panel.ForecastId.isna()].Date.unique())



print("Train date range:", str(min_date_train), " - ", str(max_date_train))

print("Test date range:", str(min_date_test), " - ", str(max_date_test))



# creating lag features

for lag in range(1, 41):

    df_panel[f"lag_{lag}_cc"] = df_panel.groupby("geography")["ConfirmedCases"].shift(lag)

    df_panel[f"lag_{lag}_ft"] = df_panel.groupby("geography")["Fatalities"].shift(lag)

    df_panel[f"lag_{lag}_rc"] = df_panel.groupby("geography")["Recoveries"].shift(lag)



for case in DAYS_SINCE_CASES:

    df_panel = df_panel.merge(df_panel[df_panel.ConfirmedCases >= case].groupby("geography")["Date"].min().reset_index().rename(columns = {"Date": f"case_{case}_date"}), on = "geography", how = "left")
## function for preparing features

def prepare_features(df, gap):

    

    df["perc_1_ac"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap}_ft"] - df[f"lag_{gap}_rc"]) / df[f"lag_{gap}_cc"]

    df["perc_1_cc"] = df[f"lag_{gap}_cc"] / df.population

    

    df["diff_1_cc"] = df[f"lag_{gap}_cc"] - df[f"lag_{gap + 1}_cc"]

    df["diff_2_cc"] = df[f"lag_{gap + 1}_cc"] - df[f"lag_{gap + 2}_cc"]

    df["diff_3_cc"] = df[f"lag_{gap + 2}_cc"] - df[f"lag_{gap + 3}_cc"]

    

    df["diff_1_ft"] = df[f"lag_{gap}_ft"] - df[f"lag_{gap + 1}_ft"]

    df["diff_2_ft"] = df[f"lag_{gap + 1}_ft"] - df[f"lag_{gap + 2}_ft"]

    df["diff_3_ft"] = df[f"lag_{gap + 2}_ft"] - df[f"lag_{gap + 3}_ft"]

    

    df["diff_123_cc"] = (df[f"lag_{gap}_cc"] - df[f"lag_{gap + 3}_cc"]) / 3

    df["diff_123_ft"] = (df[f"lag_{gap}_ft"] - df[f"lag_{gap + 3}_ft"]) / 3



    df["diff_change_1_cc"] = df.diff_1_cc / df.diff_2_cc

    df["diff_change_2_cc"] = df.diff_2_cc / df.diff_3_cc

    

    df["diff_change_1_ft"] = df.diff_1_ft / df.diff_2_ft

    df["diff_change_2_ft"] = df.diff_2_ft / df.diff_3_ft



    df["diff_change_12_cc"] = (df.diff_change_1_cc + df.diff_change_2_cc) / 2

    df["diff_change_12_ft"] = (df.diff_change_1_ft + df.diff_change_2_ft) / 2

    

    df["change_1_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 1}_cc"]

    df["change_2_cc"] = df[f"lag_{gap + 1}_cc"] / df[f"lag_{gap + 2}_cc"]

    df["change_3_cc"] = df[f"lag_{gap + 2}_cc"] / df[f"lag_{gap + 3}_cc"]



    df["change_1_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 1}_ft"]

    df["change_2_ft"] = df[f"lag_{gap + 1}_ft"] / df[f"lag_{gap + 2}_ft"]

    df["change_3_ft"] = df[f"lag_{gap + 2}_ft"] / df[f"lag_{gap + 3}_ft"]



    df["change_1_3_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 3}_cc"]

    df["change_1_3_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 3}_ft"]

    

    df["change_1_7_cc"] = df[f"lag_{gap}_cc"] / df[f"lag_{gap + 7}_cc"]

    df["change_1_7_ft"] = df[f"lag_{gap}_ft"] / df[f"lag_{gap + 7}_ft"]

    

    for case in DAYS_SINCE_CASES:

        df[f"days_since_{case}_case"] = (df[f"case_{case}_date"] - df.Date).astype("timedelta64[D]")

        df.loc[df[f"days_since_{case}_case"] < gap, f"days_since_{case}_case"] = np.nan



    df["country_flag"] = df.Province_State.isna().astype(int)



    # target variable is log of change from last known value

    df["target_cc"] = np.log1p(df.ConfirmedCases - df[f"lag_{gap}_cc"])

    df["target_ft"] = np.log1p(df.Fatalities - df[f"lag_{gap}_ft"])

    

    features = [

        f"lag_{gap}_cc",

        f"lag_{gap}_ft",

        f"lag_{gap}_rc",

        "perc_1_ac",

        "perc_1_cc",

        "diff_1_cc",

        "diff_2_cc",

        "diff_3_cc",

        "diff_1_ft",

        "diff_2_ft",

        "diff_3_ft",

        "diff_123_cc",

        "diff_123_ft",

        "diff_change_1_cc",

        "diff_change_2_cc",

        "diff_change_1_ft",

        "diff_change_2_ft",

        "diff_change_12_cc",

        "diff_change_12_ft",

        "change_1_cc",

        "change_2_cc",

        "change_3_cc",

        "change_1_ft",

        "change_2_ft",

        "change_3_ft",

        "change_1_3_cc",

        "change_1_3_ft",

        "change_1_7_cc",

        "change_1_7_ft",

        "days_since_1_case",

        "days_since_10_case",

        "days_since_50_case",

        "days_since_100_case",

        "days_since_500_case",

        "days_since_1000_case",

        "days_since_5000_case",

        "days_since_10000_case",

        "country_flag",

        "lat",

        "lon",

        "continent",

        "population",

        "area",

        "density",

        "target_cc",

        "target_ft"

    ]

    

    return df[features]
## function for building and predicting using LGBM model

def build_predict_lgbm(df_train, df_test, gap):

    

    df_train.dropna(subset = ["target_cc", "target_ft", f"lag_{gap}_cc", f"lag_{gap}_ft"], inplace = True)

    

    target_cc = df_train.target_cc

    target_ft = df_train.target_ft

    

    test_lag_cc = df_test[f"lag_{gap}_cc"].values

    test_lag_ft = df_test[f"lag_{gap}_ft"].values

    

    df_train.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    df_test.drop(["target_cc", "target_ft"], axis = 1, inplace = True)

    

    categorical_features = ["continent"]

    

    dtrain_cc = lgb.Dataset(df_train, label = target_cc, categorical_feature = categorical_features)

    dtrain_ft = lgb.Dataset(df_train, label = target_ft, categorical_feature = categorical_features)



    model_cc = lgb.train(LGB_PARAMS, train_set = dtrain_cc, num_boost_round = 200)

    model_ft = lgb.train(LGB_PARAMS, train_set = dtrain_ft, num_boost_round = 200)

    

    # inverse transform from log of change from last known value

    y_pred_cc = np.expm1(model_cc.predict(df_test, num_boost_round = 200)) + test_lag_cc

    y_pred_ft = np.expm1(model_ft.predict(df_test, num_boost_round = 200)) + test_lag_ft

    

    return y_pred_cc, y_pred_ft, model_cc, model_ft
## function for predicting moving average decay model

def predict_mad(df_test, gap, val = False):

    

    df_test["avg_diff_cc"] = (df_test[f"lag_{gap}_cc"] - df_test[f"lag_{gap + 3}_cc"]) / 3

    df_test["avg_diff_ft"] = (df_test[f"lag_{gap}_ft"] - df_test[f"lag_{gap + 3}_ft"]) / 3



    if val:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / VAL_DAYS

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / VAL_DAYS

    else:

        y_pred_cc = df_test[f"lag_{gap}_cc"] + gap * df_test.avg_diff_cc - (1 - MAD_FACTOR) * df_test.avg_diff_cc * np.sum([x for x in range(gap)]) / n_dates_test

        y_pred_ft = df_test[f"lag_{gap}_ft"] + gap * df_test.avg_diff_ft - (1 - MAD_FACTOR) * df_test.avg_diff_ft * np.sum([x for x in range(gap)]) / n_dates_test



    return y_pred_cc, y_pred_ft
## building lag x-days models

df_train = df_panel[~df_panel.Id.isna()]

df_test_full = df_panel[~df_panel.ForecastId.isna()]



df_preds_val = []

df_preds_test = []



for date in df_test_full.Date.unique():

    

    print("Processing date:", date)

    

    # ignore date already present in train data

    if date in df_train.Date.values:

        df_pred_test = df_test_full.loc[df_test_full.Date == date, ["ForecastId", "ConfirmedCases", "Fatalities"]].rename(columns = {"ConfirmedCases": "ConfirmedCases_test", "Fatalities": "Fatalities_test"})

        

        # multiplying predictions by 41 to not look cool on public LB

        df_pred_test.ConfirmedCases_test = df_pred_test.ConfirmedCases_test * 41

        df_pred_test.Fatalities_test = df_pred_test.Fatalities_test * 41

    else:

        df_test = df_test_full[df_test_full.Date == date]

        

        gap = (pd.Timestamp(date) - max_date_train).days

        

        if gap <= VAL_DAYS:

            val_date = max_date_train - pd.Timedelta(VAL_DAYS, "D") + pd.Timedelta(gap, "D")



            df_build = df_train[df_train.Date < val_date]

            df_val = df_train[df_train.Date == val_date]

            

            X_build = prepare_features(df_build, gap)

            X_val = prepare_features(df_val, gap)

            

            y_val_cc_lgb, y_val_ft_lgb, _, _ = build_predict_lgbm(X_build, X_val, gap)

            y_val_cc_mad, y_val_ft_mad = predict_mad(df_val, gap, val = True)

            

            df_pred_val = pd.DataFrame({"Id": df_val.Id.values,

                                        "ConfirmedCases_val_lgb": y_val_cc_lgb,

                                        "Fatalities_val_lgb": y_val_ft_lgb,

                                        "ConfirmedCases_val_mad": y_val_cc_mad,

                                        "Fatalities_val_mad": y_val_ft_mad,

                                       })



            df_preds_val.append(df_pred_val)



        X_train = prepare_features(df_train, gap)

        X_test = prepare_features(df_test, gap)



        y_test_cc_lgb, y_test_ft_lgb, model_cc, model_ft = build_predict_lgbm(X_train, X_test, gap)

        y_test_cc_mad, y_test_ft_mad = predict_mad(df_test, gap)

        

        if gap == 1:

            model_1_cc = model_cc

            model_1_ft = model_ft

            features_1 = X_train.columns.values

        elif gap == 14:

            model_14_cc = model_cc

            model_14_ft = model_ft

            features_14 = X_train.columns.values

        elif gap == 28:

            model_28_cc = model_cc

            model_28_ft = model_ft

            features_28 = X_train.columns.values



        df_pred_test = pd.DataFrame({"ForecastId": df_test.ForecastId.values,

                                     "ConfirmedCases_test_lgb": y_test_cc_lgb,

                                     "Fatalities_test_lgb": y_test_ft_lgb,

                                     "ConfirmedCases_test_mad": y_test_cc_mad,

                                     "Fatalities_test_mad": y_test_ft_mad,

                                    })

    

    df_preds_test.append(df_pred_test)
## validation score

df_panel = df_panel.merge(pd.concat(df_preds_val, sort = False), on = "Id", how = "left")

df_panel = df_panel.merge(pd.concat(df_preds_test, sort = False), on = "ForecastId", how = "left")



rmsle_cc_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_lgb.isna()].ConfirmedCases_val_lgb)))

rmsle_ft_lgb = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_lgb.isna()].Fatalities_val_lgb)))



rmsle_cc_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases), np.log1p(df_panel[~df_panel.ConfirmedCases_val_mad.isna()].ConfirmedCases_val_mad)))

rmsle_ft_mad = np.sqrt(mean_squared_error(np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities), np.log1p(df_panel[~df_panel.Fatalities_val_mad.isna()].Fatalities_val_mad)))



print("LGB CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_lgb, 2))

print("LGB FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_lgb, 2))

print("LGB Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_lgb + rmsle_ft_lgb) / 2, 2))

print("\n")

print("MAD CC RMSLE Val of", VAL_DAYS, "days for CC:", round(rmsle_cc_mad, 2))

print("MAD FT RMSLE Val of", VAL_DAYS, "days for FT:", round(rmsle_ft_mad, 2))

print("MAD Overall RMSLE Val of", VAL_DAYS, "days:", round((rmsle_cc_mad + rmsle_ft_mad) / 2, 2))
## preparing submission file

df_test = df_panel.loc[~df_panel.ForecastId.isna(), ["ForecastId", "Country_Region", "Province_State", "Date",

                                                     "ConfirmedCases_test", "ConfirmedCases_test_lgb", "ConfirmedCases_test_mad",

                                                     "Fatalities_test", "Fatalities_test_lgb", "Fatalities_test_mad"]].reset_index()



df_test["ConfirmedCases"] = 0.13 * df_test.ConfirmedCases_test_lgb + 0.87 * df_test.ConfirmedCases_test_mad

df_test["Fatalities"] = 0.13 * df_test.Fatalities_test_lgb + 0.87 * df_test.Fatalities_test_mad



# Since LGB models don't predict these geographies well

df_test.loc[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"]), "ConfirmedCases"] = df_test[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"])].ConfirmedCases_test_mad.values

df_test.loc[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"]), "Fatalities"] = df_test[df_test.Country_Region.isin(["Diamond Princess", "MS Zaandam"])].Fatalities_test_mad.values



df_test.loc[df_test.Date.isin(df_train.Date.values), "ConfirmedCases"] = df_test[df_test.Date.isin(df_train.Date.values)].ConfirmedCases_test.values

df_test.loc[df_test.Date.isin(df_train.Date.values), "Fatalities"] = df_test[df_test.Date.isin(df_train.Date.values)].Fatalities_test.values



df_submission = df_test[["ForecastId", "ConfirmedCases", "Fatalities"]]

df_submission.ForecastId = df_submission.ForecastId.astype(int)
TARGETS = ["ConfirmedCases", "Fatalities"]





def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))



df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

df[TARGETS] = np.log1p(df[TARGETS].values)

sub_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")



def preprocess(df):

    for col in ["Country_Region", "Province_State"]:

        df[col].fillna("", inplace=True)



    df["Date"] = pd.to_datetime(df['Date'])

    df['day'] = df.Date.dt.dayofyear

    df['geo'] = ['_'.join(x) for x in zip(df['Country_Region'], df['Province_State'])]

    return df



df = preprocess(df)

sub_df = preprocess(sub_df)



sub_df["day"] -= df["day"].min()

df["day"] -= df["day"].min()
TEST_FIRST = sub_df[sub_df["Date"] > df["Date"].max()]["Date"].min()

print(TEST_FIRST)

TEST_DAYS = (sub_df["Date"].max() - TEST_FIRST).days + 1

TEST_FIRST = (TEST_FIRST - df["Date"].min()).days



print(TEST_FIRST, TEST_DAYS)





def get_blend(pred_dfs, weights, verbose=True):

    if verbose:

        for n1, n2 in [("cpmp", "giba1"),("cpmp", "giba2"), ("cpmp", "ahmet"), ("giba1", "ahmet"), ("giba2", "ahmet")]:

            print(n1, n2, np.round(rmse(pred_dfs[n1][TARGETS[0]], pred_dfs[n2][TARGETS[0]]), 4), np.round(rmse(pred_dfs[n1][TARGETS[1]], pred_dfs[n2][TARGETS[1]]), 4))

    

    blend_df = pred_dfs["cpmp"].copy()

    blend_df[TARGETS] = 0

    for name, pred_df in pred_dfs.items():

        blend_df[TARGETS] += weights[name]*pred_df[TARGETS].values

        

    return blend_df





cov_models = {"ahmet": CovidModelAhmet(), "cpmp": CovidModelCPMP(), 'giba1': CovidModelGIBA(lag=1), 'giba2': CovidModelGIBA(lag=2)}

weights = {"ahmet": 0.35, "cpmp": 0.30, "giba1": 0.175, "giba2": 0.175}

pred_dfs = {name: cm.predict_first_day(TEST_FIRST).sort_values("geo") for name, cm in cov_models.items()}





blend_df = get_blend(pred_dfs, weights)

eval_df = blend_df.copy()



for d in range(1, TEST_DAYS):

    pred_dfs = {name: cm.predict_next_day(blend_df).sort_values("geo") for name, cm in cov_models.items()}

    blend_df = get_blend(pred_dfs, weights)

    eval_df = eval_df.append(blend_df)

    print(d, eval_df.shape, flush=True)
eval_df.head()
print(sub_df.shape)

sub_df = sub_df.merge(df.append(eval_df, sort=False), on=["geo", "day"], how="left")

print(sub_df.shape)

print(sub_df[TARGETS].isnull().mean())
sub_df.head()
flat = [

            'China_Anhui',

            'China_Beijing',

            'China_Chongqing',

            'China_Fujian',

            'China_Gansu',

            'China_Guangdong',

            'China_Guangxi',

            'China_Guizhou',

            'China_Hainan',

            'China_Hebei',

            'China_Heilongjiang',

            'China_Henan',

            'China_Hubei',

            'China_Hunan',

            'China_Jiangsu',

            'China_Jiangxi',

            'China_Jilin',

            'China_Liaoning',

            'China_Ningxia',

            'China_Qinghai',

            'China_Shaanxi',

            'China_Shandong',

            'China_Shanxi',

            'China_Sichuan',

            'China_Tibet',

            'China_Xinjiang',

            'China_Yunnan',

            'China_Zhejiang',

            'Diamond Princess_',

            'Holy See_', 

        ]     
dt = sub_df.loc[ sub_df.Date_x == "2020-04-07"  ].copy()

dt = dt.loc[ dt.geo.isin(flat)  ].copy()

dt = dt[['geo','Date_x','day','ConfirmedCases','Fatalities']].copy()

dt = dt.reset_index(drop=True)
sub_df['ow0'] = pd.merge( sub_df, dt, on='geo', how='left' )['ConfirmedCases_y'].values

sub_df['ow1'] = pd.merge( sub_df, dt, on='geo', how='left' )['Fatalities_y'].values
sub_df.loc[ sub_df.ow0.notnull() & (sub_df.Date_x >= '2020-04-08') , 'ConfirmedCases'  ] = sub_df.loc[ sub_df.ow0.notnull() & (sub_df.Date_x >= '2020-04-08') , 'ow0'  ]

sub_df.loc[ sub_df.ow1.notnull() & (sub_df.Date_x >= '2020-04-08') , 'Fatalities'  ] = sub_df.loc[ sub_df.ow1.notnull() & (sub_df.Date_x >= '2020-04-08') , 'ow1'  ]
sub_df.sort_values("ForecastId", inplace=True)

df_submission.sort_values("ForecastId", inplace=True)
for t in TARGETS:

    df_submission[t] = np.expm1(np.log1p(df_submission[t].values)*0.55 + sub_df[t].values*0.45)
df_submission.to_csv("submission.csv", index=False, float_format='%.1f')