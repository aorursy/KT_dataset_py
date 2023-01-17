import pandas as pd

import numpy as np



train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

train['Date'] = pd.to_datetime( train['Date'] )

train['Date'].max()
test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

test['Date'] = pd.to_datetime( test['Date'] )

test['Date'].max()
def BLEND_WEEK4_1():

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

                                 #continents_ids,

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

                                #continents_ids,

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

                'subsample': 0.950,

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









    print( eval_df.head() )



    print(sub_df.shape)

    sub_df = sub_df.merge(df.append(eval_df, sort=False), on=["geo", "day"], how="left")

    print(sub_df.shape)

    print(sub_df[TARGETS].isnull().mean())



    sub_df[sub_df["geo"] == "France_"][["day"] +TARGETS].plot(x="day")

    plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)



    sub_df[sub_df["geo"] == "Brazil_"][["day"] +TARGETS].plot(x="day")

    plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)



    sub_df[sub_df["geo"] == "Turkey_"][["day"] +TARGETS].plot(x="day")

    plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)



    sub_df[TARGETS] = np.expm1(sub_df[TARGETS].values)

    sub_df.sort_values("ForecastId", inplace=True)

    

    sub_df.to_csv("submission1.csv", index=False, columns=["ForecastId"] + TARGETS)

    sub_df.head()    

    
BLEND_WEEK4_1()
def BLEND_WEEK4_2():

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



        train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

        train['Date'] = pd.to_datetime( train['Date'] )

        train['Province_State'] = train['Province_State'].fillna('')



        test  = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

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

            decay = 1.

            decayK = 1.002

            for day in valid_days:

                tr, vl = create_features( train.copy(), day, lag=2 )

                dvalid = xgb.DMatrix( vl[features] )

                ypred0 = model0.predict( dvalid )

                ypred1 = model1.predict( dvalid )

                vl['ypred0'] = ypred0

                vl['ypred1'] = ypred1



                train[ 'ypred0' ] = pd.merge( train[feats], vl[feats+['ypred0']], on=feats, how='left' )['ypred0'].values * decay

                train.loc[ train.ypred0<0, 'ypred0'] = 0

                train.loc[ train.ypred0.notnull(), 'target0'] = train.loc[ train.ypred0.notnull() , 'ypred0']



                train[ 'ypred1' ] = pd.merge( train[feats], vl[feats+['ypred1']], on=feats, how='left' )['ypred1'].values * decay

                train.loc[ train.ypred1<0, 'ypred1'] = 0

                train.loc[ train.ypred1.notnull(), 'target1'] = train.loc[ train.ypred1.notnull() , 'ypred1']



                px = np.where( (train.Date==day ) )[0]

                print( day, rmse( train['log0'].iloc[px], train['target0'].iloc[px] ), rmse( train['log1'].iloc[px], train['target1'].iloc[px] )  )

                decay = decay*decayK

                decayK = decayK*0.99995



            VALID = train.loc[ (train.Date>=valid_days[0])&(train.Date<=valid_days[-1]) ].copy() 

            del VALID['ypred0'],VALID['ypred1']



            sc0 = rmse( VALID['log0'], VALID['target0'] )

            sc1 = rmse( VALID['log1'], VALID['target1'] )

            print( sc0, sc1, (sc0+sc1)/2 )



            return VALID.copy()    





        VALID0 = train_period( train, 

                              valid_days = ['2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13','2020-04-14'],

                              lag = 1,

                              seed = 1 )    



        VALID1 = train_period( train, 

                              valid_days = ['2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13','2020-04-14'],

                              lag = 1,

                              seed = 2 )    



        VALID2 = train_period( train, 

                              valid_days = ['2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13','2020-04-14'],

                              lag = 1,

                              seed = 3 )    



        VALID3 = train_period( train, 

                              valid_days = ['2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13','2020-04-14'],

                              lag = 1,

                              seed = 4 )   



        sa0 = rmse( VALID0['log0'], VALID0['target0'] )

        sa1 = rmse( VALID1['log0'], VALID1['target0'] )

        sa2 = rmse( VALID2['log0'], VALID2['target0'] )

        sa3 = rmse( VALID3['log0'], VALID3['target0'] )



        sb0 = rmse( VALID0['log1'], VALID0['target1'] )

        sb1 = rmse( VALID1['log1'], VALID1['target1'] )

        sb2 = rmse( VALID2['log1'], VALID2['target1'] )

        sb3 = rmse( VALID3['log1'], VALID3['target1'] )





        print('24-11: ' + str(sa0)[:6] + ', ' + str(sb0)[:6] + ' = ' + str(0.5*sa0+0.5*sb0)[:6]  )

        print('27-11: ' + str(sa1)[:6] + ', ' + str(sb1)[:6] + ' = ' + str(0.5*sa1+0.5*sb1)[:6]  )

        print('30-11: ' + str(sa2)[:6] + ', ' + str(sb2)[:6] + ' = ' + str(0.5*sa2+0.5*sb2)[:6]  )

        print('02-11: ' + str(sa3)[:6] + ', ' + str(sb3)[:6] + ' = ' + str(0.5*sa3+0.5*sb3)[:6]  )



        print( 'Avg: ',  (sa0+sb0+sa1+sb1+sa2+sb2+sa3+sb3) / 8 )



        TEST  = train_period( train, 

                              valid_days = ['2020-04-15','2020-04-16','2020-04-17','2020-04-18','2020-04-19','2020-04-20',

                                            '2020-04-21','2020-04-22','2020-04-23','2020-04-24','2020-04-25','2020-04-26','2020-04-27','2020-04-28','2020-04-29','2020-04-30',

                                            '2020-05-01','2020-05-02','2020-05-03','2020-05-04','2020-05-05','2020-05-06','2020-05-07','2020-05-08','2020-05-09','2020-05-10',

                                            '2020-05-11','2020-05-12','2020-05-13','2020-05-14'],

                              lag = 1,

                              seed = 1 )    



        VALID2_sub = VALID3.copy()

        VALID2_sub['target0'] = np.log1p( VALID2_sub['ConfirmedCases']  )

        VALID2_sub['target1'] = np.log1p( VALID2_sub['Fatalities']  )

        sub = pd.concat( (VALID2_sub,TEST.loc[ TEST.Date>='2020-04-15' ] ) )



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

        VALID0.to_csv('fold-24-14.csv', index=False)

        VALID1.to_csv('fold-27-14.csv', index=False)

        VALID2.to_csv('fold-30-14.csv', index=False)

        VALID3.to_csv('fold-02-14.csv', index=False)

        TEST.to_csv('fold-submission.csv', index=False)    



        return sub







    def get_nn_sub():

        df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

        sub_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")



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



        sub_df_public  = sub_df[sub_df["Date"] <= df["Date"].max()].copy()

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



    sub_df.to_csv("submission2.csv", index=False)







    test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

    test['Date'] = pd.to_datetime( test['Date'] )

    test = test.merge( sub_df, on='ForecastId', how='left' )

    print( test.head(10) )



    test.loc[ test.Country_Region=='Spain' ].plot( x='Date', y='ConfirmedCases' )

    test.loc[ test.Country_Region=='Spain' ].plot( x='Date', y='Fatalities' )



    test.loc[ test.Country_Region=='Brazil' ].plot( x='Date', y='ConfirmedCases' )

    test.loc[ test.Country_Region=='Brazil' ].plot( x='Date', y='Fatalities' )



    test.loc[ test.Country_Region=='Italy' ].plot( x='Date', y='ConfirmedCases' )

    test.loc[ test.Country_Region=='Italy' ].plot( x='Date', y='Fatalities' )



    test.loc[ test.Country_Region=='Iran' ].plot( x='Date', y='ConfirmedCases' )

    test.loc[ test.Country_Region=='Iran' ].plot( x='Date', y='Fatalities' )



    tmp = test.groupby('Date')[['ConfirmedCases','Fatalities']].agg('sum').reset_index()

    tmp['ConfirmedCases'] = np.log1p( tmp['ConfirmedCases'] )

    tmp['Fatalities'] = np.log1p( tmp['Fatalities'] )

    tmp.plot( x='Date', y='ConfirmedCases' )

    tmp.plot( x='Date', y='Fatalities' )



BLEND_WEEK4_2()
def model_simple_decay():

    

    import pandas as pd

    import numpy as np

    from scipy.optimize import curve_fit



    %matplotlib inline

    %config InlineBackend.figure_format = 'retina'

    import matplotlib.pyplot as plt



    from sklearn.linear_model import LinearRegression

    from sklearn.metrics import log_loss



    import xgboost as xgb



    from tqdm.notebook import tqdm





    def rmse( yt, yp ):

        return np.sqrt( np.mean( (yt-yp)**2 ) )



    def funclog(x, a, b,c):

        return a*np.log(b+x)+c



    def ewa( var, alpha=0.5 ):

        x    = var.copy()

        for i in range( 1,len(var) ):

            x[i] = x[i]*alpha + x[i-1]*(1-alpha)

        return x





    train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

    train['Date'] = pd.to_datetime( train['Date'] )

    maxdate  = str(train['Date'].max())[:10]

    testdate = str( train['Date'].max() + pd.Timedelta(days=1) )[:10]

    print( maxdate, testdate )

    train['Province_State'] = train['Province_State'].fillna('')



    test  = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

    test['Date'] = pd.to_datetime( test['Date'] )

    test['Province_State'] = test['Province_State'].fillna('')

    test['Id'] = -1

    test['ConfirmedCases'] = 0

    test['Fatalities'] = 0



    publictest = test.loc[ test.Date > train.Date.max() ].copy()

    train = pd.concat( (train, publictest ), sort=False )

    train.sort_values( ['Country_Region','Province_State','Date'], inplace=True )

    train = train.reset_index(drop=True)



    train['ForecastId'] = pd.merge( train, test, on=['Country_Region','Province_State','Date'], how='left' )['ForecastId_y'].values



    train['cid'] = train['Country_Region'] + '_' + train['Province_State']





    #Fix some anomalities:

    train.loc[ (train.cid=='China_Guizhou') & (train.Date=='2020-03-17') , 'ConfirmedCases' ] = 146

    train.loc[ (train.cid=='Guyana_')&(train.Date>='2020-03-22')&(train.Date<='2020-03-30') , 'ConfirmedCases' ] = 12

    train.loc[ (train.cid=='US_Virgin Islands')&(train.Date>='2020-03-29')&(train.Date<='2020-03-29') , 'ConfirmedCases' ] = 24

    train.loc[ (train.cid=='US_Virgin Islands')&(train.Date>='2020-03-30')&(train.Date<='2020-03-30') , 'ConfirmedCases' ] = 27



    train.loc[ (train.cid=='Iceland_')&(train.Date>='2020-03-15')&(train.Date<='2020-03-15') , 'Fatalities' ] = 0

    train.loc[ (train.cid=='Kazakhstan_')&(train.Date>='2020-03-20')&(train.Date<='2020-03-20') , 'Fatalities' ] = 0

    train.loc[ (train.cid=='Serbia_')&(train.Date>='2020-03-26')&(train.Date<='2020-03-26') , 'Fatalities' ] = 5

    train.loc[ (train.cid=='Serbia_')&(train.Date>='2020-03-27')&(train.Date<='2020-03-27') , 'Fatalities' ] = 6

    train.loc[ (train.cid=='Slovakia_')&(train.Date>='2020-03-22')&(train.Date<='2020-03-31') , 'Fatalities' ] = 1

    train.loc[ (train.cid=='US_Hawaii')&(train.Date>='2020-03-25')&(train.Date<='2020-03-31') , 'Fatalities' ] = 1



    train['log0'] = np.log1p( train['ConfirmedCases'] )

    train['log1'] = np.log1p( train['Fatalities'] )



    train = train.loc[ (train.log0 > 0) | (train.ForecastId.notnull()) | (train.Date >= '2020-03-17') ].copy()

    train = train.reset_index(drop=True)



    train['day'] = train.groupby('cid')['Id'].cumcount()



    #Fix



    print( train.head(4) )

    print(train.shape)







    cid = train['cid'].unique().tolist()

    train['dif0'] = train['log0'] - train.groupby('cid')['log0'].shift(1)

    train['dif0'].fillna(0, inplace=True)





    #K = [ 0.00814423,  0.54210745, -1.36271011,  0.29145165,  6.90285755,-1.70264593]

    K = [ -3.61052304e-03,  1.97547109e-01,  1.98486015e+00,  9.03304453e-02,4.54479999e+00,  2.49382214e+00]



    thr = '2020-04-01'

    VAL = []

    for c in cid:

        tmp = train.loc[ (train.cid==c) & (train.Date<thr) ].copy()

        tmp['dif1'] = K[0]*ewa( tmp['dif0'].values, K[1] ) + K[2]*ewa( tmp['dif0'].values[::-1], K[3] )[::-1]

        tmp['dif1'] = tmp['dif1'] * (tmp['log0'].values[-1]/(tmp['dif1'].sum()+0.000001))



        tmp['dif2'] = tmp['dif1'] - tmp['dif1'].shift(1)

        tmp['dif2'].fillna(0, inplace=True)

        decay = tmp['dif2'].values[-tmp.shape[0]//10:].mean()

        lastv = tmp['log0'].values[-1]



        tmp2 = train.loc[ (train.cid==c) & (train.Date >= thr) ].copy()

        tmp2['dif1'] = [ tmp['dif1'].values[-1]*(1-(K[4]*i/tmp2.shape[0])) - decay*(1-(K[5]*i/tmp2.shape[0])) for i in range(tmp2.shape[0]) ]

        tmp2['dif2'] = 0



        ts = pd.concat( (tmp, tmp2) )

        ts['dif2'] = ts['dif1'].cumsum() 

        ts.loc[ (ts.dif2<lastv)&(ts.Date>=thr), 'dif2'] = lastv

        ts.loc[ (ts.dif2>13), 'dif2'] = 13

        ts['dif2'] = ts['dif2'].cummax()

        VAL.append( ts )



    VAL = pd.concat( VAL, sort=False )



    tmp = VAL.loc[ (VAL.Date>=thr)&(VAL.Date<='2020-04-14')  ].copy()

    sc = rmse( tmp.log0, tmp.dif2 )

    print(sc)







    K = [ -3.61052304e-03,  1.97547109e-01,  1.98486015e+00,  9.03304453e-02,4.54479999e+00,  2.49382214e+00]



    thr = '2020-04-15'

    VAL = []

    for c in cid:

        tmp = train.loc[ (train.cid==c) & (train.Date<thr) ].copy()

        tmp['dif1'] = K[0]*ewa( tmp['dif0'].values, K[1] ) + K[2]*ewa( tmp['dif0'].values[::-1], K[3] )[::-1]

        tmp['dif1'] = tmp['dif1'] * (tmp['log0'].values[-1]/(tmp['dif1'].sum()+0.000001))



        tmp['dif2'] = tmp['dif1'] - tmp['dif1'].shift(1)

        tmp['dif2'].fillna(0, inplace=True)

        decay = tmp['dif2'].values[-tmp.shape[0]//10:].mean()

        lastv = tmp['log0'].values[-1]



        tmp2 = train.loc[ (train.cid==c) & (train.Date >= thr) ].copy()

        tmp2['dif1'] = [ tmp['dif1'].values[-1]*(1-(K[4]*i/tmp2.shape[0])) - decay*(1-(K[5]*i/tmp2.shape[0])) for i in range(tmp2.shape[0]) ]

        tmp2['dif2'] = 0



        ts = pd.concat( (tmp, tmp2) )

        ts['dif2'] = ts['dif1'].cumsum() 

        ts.loc[ (ts.dif2<lastv)&(ts.Date>=thr), 'dif2'] = lastv

        ts.loc[ (ts.dif2>13), 'dif2'] = 13

        ts['dif2'] = ts['dif2'].cummax()

        VAL.append( ts )



    VAL1 = pd.concat( VAL, sort=False )





    K = [ 0.00814423,  0.54210745, -1.36271011,  0.29145165,  6.90285755,-1.70264593]



    thr = '2020-04-15'

    VAL = []

    for c in cid:

        tmp = train.loc[ (train.cid==c) & (train.Date<thr) ].copy()

        tmp['dif1'] = K[0]*ewa( tmp['dif0'].values, K[1] ) + K[2]*ewa( tmp['dif0'].values[::-1], K[3] )[::-1]

        tmp['dif1'] = tmp['dif1'] * (tmp['log0'].values[-1]/(tmp['dif1'].sum()+0.000001))



        tmp['dif2'] = tmp['dif1'] - tmp['dif1'].shift(1)

        tmp['dif2'].fillna(0, inplace=True)

        decay = tmp['dif2'].values[-tmp.shape[0]//10:].mean()

        lastv = tmp['log0'].values[-1]



        tmp2 = train.loc[ (train.cid==c) & (train.Date >= thr) ].copy()

        tmp2['dif1'] = [ tmp['dif1'].values[-1]*(1-(K[4]*i/tmp2.shape[0])) - decay*(1-(K[5]*i/tmp2.shape[0])) for i in range(tmp2.shape[0]) ]

        tmp2['dif2'] = 0



        ts = pd.concat( (tmp, tmp2) )

        ts['dif2'] = ts['dif1'].cumsum() 

        ts.loc[ (ts.dif2<lastv)&(ts.Date>=thr), 'dif2'] = lastv

        ts.loc[ (ts.dif2>13), 'dif2'] = 13

        ts['dif2'] = ts['dif2'].cummax()

        VAL.append( ts )



    VAL2 = pd.concat( VAL, sort=False )







    tmp = VAL1.loc[ VAL1.cid == 'US_New York' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'US_California' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Brazil_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Spain_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Italy_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Iran_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Chile_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Congo (Brazzaville)_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'India_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Peru_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Japan_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'US_Washington' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'US_Maryland' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Trinidad and Tobago_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Kosovo_' ].copy()

    tmp[['Date','log0','dif2']].plot(x='Date')





    tmp = VAL1.groupby(['Date'])['dif2'].agg('sum').reset_index()

    tmp[['Date','dif2']].plot(x='Date')



    tmp = VAL2.groupby(['Date'])['dif2'].agg('sum').reset_index()

    tmp[['Date','dif2']].plot(x='Date')



    VAL1.sort_values( ['cid','Date','Id','ForecastId'], inplace=True )

    VAL2.sort_values( ['cid','Date','Id','ForecastId'], inplace=True )



    VAL1 = VAL1.loc[ VAL1.Date >='2020-04-02' ].copy()

    VAL2 = VAL2.loc[ VAL2.Date >='2020-04-02' ].copy()





    VAL1['ConfirmedCases'] = np.expm1( 0.5*VAL1['dif2'] + 0.5*VAL2['dif2'] )











    K = [1.34183856e-03, 6.74294944e-01, 1.25931206e+00, 6.75211842e-02, 4.20000395e-01, 2.20018025e+00]

    thr = '2020-04-15'

    VAL = []

    for c in cid:

        tmp = train.loc[ (train.cid==c) & (train.Date<thr) ].copy()

        tmp['dif1'] = K[0]*ewa( tmp['dif0'].values, K[1] ) + K[2]*ewa( tmp['dif0'].values[::-1], K[3] )[::-1]

        tmp['dif1'] = tmp['dif1'] * (tmp['log1'].values[-1]/(tmp['dif1'].sum()+0.000001))



        tmp['dif2'] = tmp['dif1'] - tmp['dif1'].shift(1)

        tmp['dif2'].fillna(0, inplace=True)

        decay = tmp['dif2'].values[-tmp.shape[0]//10:].mean()

        lastv = tmp['log1'].values[-1]



        tmp2 = train.loc[ (train.cid==c) & (train.Date >= thr) ].copy()

        tmp2['dif1'] = [ tmp['dif1'].values[-1]*(1-(K[4]*i/tmp2.shape[0])) - decay*(1-(K[5]*i/tmp2.shape[0])) for i in range(tmp2.shape[0]) ]

        tmp2['dif2'] = 0



        ts = pd.concat( (tmp, tmp2) )

        ts['dif2'] = ts['dif1'].cumsum() 

        ts.loc[ (ts.dif2<lastv)&(ts.Date>=thr), 'dif2'] = lastv

        ts['dif2'] = ts['dif2'].cummax()

        VAL.append( ts )



    VAL3 = pd.concat( VAL, sort=False )





    K = [0.00524396, 0.29401264, 1.58642304, 0.0290667 , 2.45286858, 0.34968403]



    thr = '2020-04-15'

    VAL = []

    for c in cid:

        tmp = train.loc[ (train.cid==c) & (train.Date<thr) ].copy()

        tmp['dif1'] = K[0]*ewa( tmp['dif0'].values, K[1] ) + K[2]*ewa( tmp['dif0'].values[::-1], K[3] )[::-1]

        tmp['dif1'] = tmp['dif1'] * (tmp['log1'].values[-1]/(tmp['dif1'].sum()+0.000001))



        tmp['dif2'] = tmp['dif1'] - tmp['dif1'].shift(1)

        tmp['dif2'].fillna(0, inplace=True)

        decay = tmp['dif2'].values[-tmp.shape[0]//10:].mean()

        lastv = tmp['log1'].values[-1]



        tmp2 = train.loc[ (train.cid==c) & (train.Date >= thr) ].copy()

        tmp2['dif1'] = [ tmp['dif1'].values[-1]*(1-(K[4]*i/tmp2.shape[0])) - decay*(1-(K[5]*i/tmp2.shape[0])) for i in range(tmp2.shape[0]) ]

        tmp2['dif2'] = 0



        ts = pd.concat( (tmp, tmp2) )

        ts['dif2'] = ts['dif1'].cumsum() 

        ts.loc[ (ts.dif2<lastv)&(ts.Date>=thr), 'dif2'] = lastv

        ts['dif2'] = ts['dif2'].cummax()

        VAL.append( ts )



    VAL4 = pd.concat( VAL, sort=False )







    tmp = VAL3.groupby(['Date'])['dif2'].agg('sum').reset_index()

    tmp[['Date','dif2']].plot(x='Date')



    tmp = VAL4.groupby(['Date'])['dif2'].agg('sum').reset_index()

    tmp[['Date','dif2']].plot(x='Date')



    VAL3.sort_values( ['cid','Date','Id','ForecastId'], inplace=True )

    VAL4.sort_values( ['cid','Date','Id','ForecastId'], inplace=True )



    VAL3 = VAL3.loc[ VAL3.Date >='2020-04-02' ].copy()

    VAL4 = VAL4.loc[ VAL4.Date >='2020-04-02' ].copy()



    VAL1['Fatalities'] = np.expm1( 0.5*VAL3['dif2'] + 0.5*VAL4['dif2'] )



    VAL1['dif1'] = np.log1p( VAL1['ConfirmedCases'] )

    VAL1['dif2'] = np.log1p( VAL1['Fatalities'] )



    tmp = VAL1.groupby(['Date'])['dif1'].agg('sum').reset_index()

    tmp[['Date','dif1']].plot(x='Date')



    tmp = VAL1.groupby(['Date'])['dif2'].agg('sum').reset_index()

    tmp[['Date','dif2']].plot(x='Date')





    tmp = VAL1.groupby(['Date'])['ConfirmedCases'].agg('sum').reset_index()

    tmp[['Date','ConfirmedCases']].plot(x='Date')



    tmp = VAL1.groupby(['Date'])['Fatalities'].agg('sum').reset_index()

    tmp[['Date','Fatalities']].plot(x='Date')





    tmp = VAL1.loc[ VAL1.cid == 'US_New York' ].copy()

    tmp[['Date','ConfirmedCases','Fatalities']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'US_California' ].copy()

    tmp[['Date','ConfirmedCases','Fatalities']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Italy_' ].copy()

    tmp[['Date','ConfirmedCases','Fatalities']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Spain_' ].copy()

    tmp[['Date','ConfirmedCases','Fatalities']].plot(x='Date')



    tmp = VAL1.loc[ VAL1.cid == 'Brazil_' ].copy()

    tmp[['Date','ConfirmedCases','Fatalities']].plot(x='Date')



    sub = VAL1[['ForecastId','ConfirmedCases','Fatalities']].copy()

    sub['ForecastId'] = sub['ForecastId'].astype(int)

    

    sub.sort_values("ForecastId", inplace=True)



    sub.to_csv( 'submission3.csv', index=False )



    return 
model_simple_decay()
sub1 = pd.read_csv("submission1.csv")

sub2 = pd.read_csv("submission2.csv")

sub3 = pd.read_csv("submission3.csv")



print( np.corrcoef( sub1['ConfirmedCases'], sub2['ConfirmedCases'] )[0][1] )

print( np.corrcoef( sub1['Fatalities']    , sub2['Fatalities'] )[0][1] )

print( np.corrcoef( sub1['ConfirmedCases'], sub3['ConfirmedCases'] )[0][1] )

print( np.corrcoef( sub1['Fatalities']    , sub3['Fatalities'] )[0][1] )



sub1['ConfirmedCases'] = 0.35*sub1['ConfirmedCases'] + 0.25*sub2['ConfirmedCases'] + 0.40*sub3['ConfirmedCases']

sub1['Fatalities']     = 0.35*sub1['Fatalities']     + 0.25*sub2['Fatalities']     + 0.40*sub3['Fatalities']



sub1.to_csv("submission.csv", index=False)
test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

test['Date'] = pd.to_datetime( test['Date'] )

test = test.merge( sub1, on='ForecastId', how='left' )

print( test.head(10) )



print('Spain')

test.loc[ test.Country_Region=='Spain' ].plot( x='Date', y='ConfirmedCases' )

test.loc[ test.Country_Region=='Spain' ].plot( x='Date', y='Fatalities' )



print('Brazil')

test.loc[ test.Country_Region=='Brazil' ].plot( x='Date', y='ConfirmedCases' )

test.loc[ test.Country_Region=='Brazil' ].plot( x='Date', y='Fatalities' )



print('Italy')

test.loc[ test.Country_Region=='Italy' ].plot( x='Date', y='ConfirmedCases' )

test.loc[ test.Country_Region=='Italy' ].plot( x='Date', y='Fatalities' )



print('Iran')

test.loc[ test.Country_Region=='Iran' ].plot( x='Date', y='ConfirmedCases' )

test.loc[ test.Country_Region=='Iran' ].plot( x='Date', y='Fatalities' )



print('Global')

tmp = test.groupby('Date')[['ConfirmedCases','Fatalities']].agg('sum').reset_index()

tmp['ConfirmedCases'] = np.log1p( tmp['ConfirmedCases'] )

tmp['Fatalities'] = np.log1p( tmp['Fatalities'] )

tmp.plot( x='Date', y='ConfirmedCases' )

tmp.plot( x='Date', y='Fatalities' )
