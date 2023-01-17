import numpy as np

import pandas as pd

import category_encoders as ce

import os, gc, pickle, time, datetime

from tqdm import tqdm_notebook as tqdm



from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit



import lightgbm as lgb

import matplotlib.pyplot as plt

import seaborn as sns



SEED = 42

np.random.seed(SEED)

os.listdir('../input/ashrae-energy-prediction')
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings                    

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

    mem_usg = df.memory_usage().sum() / 1024**2 

    return df, NAlist
class PreprocessingDataset:

    def __init__(self):

        self.df = None

        

    def prep(self, df, df_weather, df_building, mode='train'):

        # merge data

        df = pd.merge(df, df_building, how="left", on=["building_id"])

        df = pd.merge(df, df_weather, how='left', on=["site_id", "timestamp"])

        self.df, _ = reduce_mem_usage(df)

        del df, df_weather, df_building

        gc.collect()

        

        # Datetime

        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        self.df['month'] = self.df['timestamp'].dt.month.astype(np.uint8)

        self.df['day'] = self.df['timestamp'].dt.day.astype(np.uint8)

        self.df['hour'] = self.df['timestamp'].dt.hour.astype(np.uint8)

        self.df['weekday'] = self.df['timestamp'].dt.weekday.astype(np.uint8)

        # Sort Timestamp

        self.df = self.df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

        del self.df['timestamp']

        gc.collect()

        

        # Year Built

        self.df['year_built'] = self.df['year_built'] - 1900

        

        # square_feet

        self.df['square_feet'] = np.log(self.df['square_feet'])

        

        # LabelEncoder

        list_cols = ['primary_use']

        if mode == 'train':

            self.ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

            self.df = self.ce_oe.fit_transform(self.df)

        elif mode == 'test':

            self.df = self.ce_oe.transform(self.df)

        

        # Drop Columns

        drop_cols = ["precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"]

        self.df.drop(drop_cols, axis=1, inplace=True)

        

        # Data Type

        # float32

        cols = self.df.select_dtypes(np.float64).columns

        for c in cols:

            self.df[c] = self.df[c].astype(np.float32)

        # category

        cols = ["site_id", "building_id", "primary_use", "hour", "day", "weekday", "month", "meter"]

        for c in cols:

            self.df[c] = self.df[c].astype('category')

            

        # sort row_id

        if mode == 'test':

            self.df = self.df.sort_values(by='row_id').reset_index(drop=True)

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

df_weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

df_building = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")



data = PreprocessingDataset()

data.prep(train, df_weather_train, df_building, mode='train')

del train, df_weather_train, df_building
data.df.head()
data.df.shape
class Trainer:

    

    def __init__(self):

        pass

    

    def train(self, df, params, cv, num_boost_round, early_stopping_rounds, verbose):

        self.y = np.log1p(df['meter_reading'])

        self.x = df.drop(['meter_reading'], axis=1)

        self.cv = cv

        self.oof = 0.0

        self.models = []

        self.features = self.x.columns

        

        for i, (trn_idx, val_idx) in enumerate(cv.split(self.x)):

            print('Fold {} Model Creating...'.format(i+1))

            _start = time.time()



            train_data = lgb.Dataset(self.x.iloc[trn_idx], label=self.y.iloc[trn_idx])

            val_data = lgb.Dataset(self.x.iloc[val_idx], label=self.y.iloc[val_idx], reference=train_data)



            model = lgb.train(params, 

                              train_data, 

                              num_boost_round=num_boost_round,

                              valid_sets=(train_data, val_data),

                              early_stopping_rounds=early_stopping_rounds,

                              verbose_eval=verbose)



            y_pred = model.predict(self.x.iloc[val_idx], num_iteration=model.best_iteration)

            error = np.sqrt(mean_squared_error(y_pred, self.y.iloc[val_idx]))

            self.oof += error / cv.n_splits



            print('Fold {}: {:.5f}'.format(i+1, error))



            elapsedtime = time.time() - _start

            print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=elapsedtime))))

            print('')

        

            self.models.append(model)

        print('OOF Error: {:.5f}'.format(self.oof))

        

        return model

        

    def predict(self, df):

        

        if 'row_id' in df.columns:

            df.drop('row_id', axis=1, inplace=True)

            

        i=0

        res=[]

        step_size = 500000

        for j in tqdm(range(int(np.ceil(df.shape[0]/500000)))):

            res.append(np.expm1(sum([model.predict(df.iloc[i:i+step_size], num_iteration=model.best_iteration) for model in self.models]) / self.cv.n_splits))

            i+=step_size

            

        res = np.concatenate(res)

        

        return res

    

    def get_feature_importance(self):

        importance = np.zeros(len(self.features))

        

        for i in range(len(self.models)):

            importance += self.models[i].feature_importance() / len(self.models)

        

        importance_df = pd.DataFrame({

            'feature': self.features,

            'importance': importance

        })

        importance_df = importance_df.sort_values(by='importance', ascending=False)



        fig = plt.figure(figsize=(12, 20))

        sns.barplot(x='importance', y='feature', data=importance_df)

        plt.show()
# Config

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'subsample': 0.7,

    'learning_rate': 0.01,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.9

}



num_folds = 4

# cv = KFold(num_folds, shuffle=True, random_state=42)

cv = TimeSeriesSplit(num_folds)

num_boost_round = 6000

early_stopping_rounds = 100

verbose = 1000
%%time

model = Trainer()

_ = model.train(data.df, params, cv, num_boost_round, early_stopping_rounds, verbose)
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

df_weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

df_building = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")



data.prep(test, df_weather_test, df_building, mode='test')

del test, df_weather_test, df_building

gc.collect()



pred = model.predict(data.df)
sub = pd.read_csv("../input/ashrae-energy-prediction/sample_submission.csv")



sub["meter_reading"] = pred

sub.to_csv("submission_oof_{:.3f}.csv".format(model.oof), index=False)
model.get_feature_importance()