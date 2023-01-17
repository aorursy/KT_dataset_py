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

            'max_depth': 5,

            'gamma': 0.000,

            'learning_rate': 0.010,

            'min_child_weight': 6.00,

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

#         for n1, n2 in [("cpmp", "giba1"),("cpmp", "giba2"), ("cpmp", "ahmet"), ("giba1", "ahmet"), ("giba2", "ahmet")]:

        for n1, n2 in [("giba1", "ahmet"), ("giba2", "ahmet")]:

            print(n1, n2, np.round(rmse(pred_dfs[n1][TARGETS[0]], pred_dfs[n2][TARGETS[0]]), 4), np.round(rmse(pred_dfs[n1][TARGETS[1]], pred_dfs[n2][TARGETS[1]]), 4))

    

#     blend_df = pred_dfs["cpmp"].copy()

    blend_df = pred_dfs["giba1"].copy()

    blend_df[TARGETS] = 0

    for name, pred_df in pred_dfs.items():

        blend_df[TARGETS] += weights[name]*pred_df[TARGETS].values

        

    return blend_df





# cov_models = {"ahmet": CovidModelAhmet(), "cpmp": CovidModelCPMP(), 'giba1': CovidModelGIBA(lag=1), 'giba2': CovidModelGIBA(lag=2)}

cov_models = {"ahmet": CovidModelAhmet(), 'giba1': CovidModelGIBA(lag=1), 'giba2': CovidModelGIBA(lag=2)}

# weights = {"ahmet": 0.35, "cpmp": 0.30, "giba1": 0.175, "giba2": 0.175}

weights = {"ahmet": 0.45, "giba1": 0.275, "giba2": 0.275}



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
sub_df[sub_df["geo"] == "France_"][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
sub_df[sub_df["geo"] == "Brazil_"][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
sub_df[sub_df["geo"] == flat[0]][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
sub_df[sub_df["geo"] == flat[1]][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
sub_df[sub_df["geo"] == flat[2]][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
sub_df[sub_df["geo"] == flat[-1]][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
sub_df[sub_df["geo"] == flat[-2]][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
dt = sub_df.loc[ sub_df.Date_x == "2020-04-13"  ].copy()

dt = dt.loc[ dt.geo.isin(flat)  ].copy()

dt = dt[['geo','Date_x','day','ConfirmedCases','Fatalities']].copy()

dt = dt.reset_index(drop=True)

dt
sub_df['ow0'] = pd.merge( sub_df, dt, on='geo', how='left' )['ConfirmedCases_y'].values

sub_df['ow1'] = pd.merge( sub_df, dt, on='geo', how='left' )['Fatalities_y'].values

sub_df.tail(60)
sub_df.loc[ sub_df.geo.isin(flat) ]
sub_df.loc[ sub_df.ow0.notnull() & (sub_df.Date_x >= '2020-04-14') , 'ConfirmedCases'  ] = sub_df.loc[ sub_df.ow0.notnull() & (sub_df.Date_x >= '2020-04-14') , 'ow0'  ]

sub_df.loc[ sub_df.ow1.notnull() & (sub_df.Date_x >= '2020-04-14') , 'Fatalities'  ]     = sub_df.loc[ sub_df.ow1.notnull() & (sub_df.Date_x >= '2020-04-14') , 'ow1'  ]
sub_df.loc[ sub_df.geo.isin(flat) ]
sub_df[sub_df["geo"] == flat[0]][["day"] +TARGETS].plot(x="day")

plt.axvline(TEST_FIRST, color='r', linestyle='--', lw=2)
sub_df[TARGETS] = np.expm1(sub_df[TARGETS].values)

sub_df.to_csv("submission.csv", index=False, columns=["ForecastId"] + TARGETS)

sub_df.head()
sub_df.shape