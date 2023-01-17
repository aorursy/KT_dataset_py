import pandas as pd

import numpy as np



from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from scipy import stats



from sklearn.metrics import mean_squared_error

from keras import backend as k

from sklearn.model_selection import GridSearchCV



def deleteTextFromColumn(df, column_name, text):

    df[column_name] = df[column_name].str.replace(text, '')

    return df



def splitTextDate(df):

    if 'date' not in df.columns:

        text = 'No date column is in dataframe'

        raise ValueError(text)

    

    df['year'] = pd.to_numeric(df['date'].str[0:4])

    df['month'] = pd.to_numeric(df['date'].str[4:6])

    df['day'] = pd.to_numeric(df['date'].str[6:])

    

    return df



def root_mean_squared_error(y_true, y_pred):

    return k.sqrt(k.mean(k.square(y_pred - y_true), axis=-1))



def evaluate_prediction(y_pred):

    answer_df = pd.read_csv('./dataset/answer.csv')

    y_test = answer_df['price']

    rms = np.sqrt(mean_squared_error(y_test, y_pred))

    return rms



def print_best_params(model, params):

    grid_model = GridSearchCV(

        model, 

        param_grid = params,

        cv=5,

        scoring='neg_mean_squared_error')



    grid_model.fit(df_train_features, df_train_target)

    rmse = np.sqrt(-1*grid_model.best_score_)

    print(

        '{0} 5 CV 시 최적 평균 RMSE 값 {1}, 최적 alpha:{2}'.format(model.__class__.__name__, np.round(rmse, 4), grid_model.best_params_))

    return grid_model.best_estimator_

'''

train_feature = [

            # 'date',

            'year',

            'month',

            'bedrooms',

            'bathrooms',

            'sqft_living',

            'sqft_lot',

            'floors',

            'waterfront',

            'view',

            'condition',

            'grade',

            'sqft_above',

            'sqft_basement',

            'yr_built',

            'yr_renovated',

            'zipcode',

            'lat',

            'long',

            # 'sqft_living15',

            # 'sqft_lot15',

            # 'is_renovated',

            # 'total_increased',

            # 'sqft_ratio',

            # 'sqft_ratio15',

            # 'total_sqrt',

            # 'total_sqrt15',

            # 'total_rooms',

            # 'sqft_total_size',

            'mean',

            # 'living_increased',

            # 'has_basement',

            # 'attic'

        ]

'''



class Dataset():

    '''

        TODO:

        - valid, train split하기

        - outlier 제거하기

        - 시계열 분석

    '''

    def __init__(self,

                 train_feature,

                 del_outlier=True,

                 apply_log=True,

                 apply_scale=False,

                 label_encode=True):

        self.df_train = pd.read_csv('../input/train.csv')

        self.df_test = pd.read_csv('../input/test.csv')

        

        self.train_feature = train_feature

        self.apply_scale = apply_scale

        self.label_encode = label_encode



        self.process_text()

        self.process_long()

        self.check_attic()

        self.check_basement()

        self.check_living_increased()

        self.check_lot_increased()

        self.total_rooms()

        self.total_size()

        self.total_sqrt()

        self.total_sqrt15()

        self.check_sqft_ratio()

        self.check_sqft15_ratio()

        self.check_is_renovated()

        self.zipcode_per_price()

        self.check_total_increased()

        self.check_floor_area_ratio()

        self.check_sqft_floor()

        self.check_how_old()



        if del_outlier:

            self.delete_outlier()

        if apply_log:

            self.apply_log()



    def process_text(self):

        self.df_train = deleteTextFromColumn(self.df_train, 'date', 'T000000')

        self.df_test = deleteTextFromColumn(self.df_test, 'date', 'T000000')

        self.df_train = splitTextDate(self.df_train)

        self.df_test = splitTextDate(self.df_test)



        self.df_train['date'] = self.df_train['date'].apply(pd.to_numeric)

        self.df_test['date'] = self.df_test['date'].apply(pd.to_numeric)



    def process_long(self):

        self.df_train['long'] = np.abs(self.df_train['long'])

        self.df_test['long'] = np.abs(self.df_test['long'])



    def check_attic(self):

        self.df_train['attic'] = self.df_train['floors'] % 1 == 0.5

        self.df_test['attic'] = self.df_test['floors'] % 1 == 0.5



        self.df_train['floors'] = self.df_train['floors'] // 1

        self.df_train['floors'] = self.df_train['floors'] // 1



    def check_basement(self):

        self.df_train['has_basement'] = self.df_train['sqft_basement'] > 0

        self.df_test['has_basement'] = self.df_test['sqft_basement'] > 0

    

    def check_living_increased(self):

        self.df_train['living_increased'] = self.df_train['sqft_living15'] > self.df_train['sqft_living']

        self.df_test['living_increased'] = self.df_test['sqft_living15'] > self.df_test['sqft_living']

    

    def check_lot_increased(self):

        self.df_train['living_increased'] = self.df_train['sqft_lot15'] > self.df_train['sqft_lot']

        self.df_test['living_increased'] = self.df_test['sqft_lot15'] > self.df_test['sqft_lot']

    

    def check_total_increased(self):

        self.df_train['total_increased'] = self.df_train['total_sqrt15'] > self.df_train['total_sqrt']

        self.df_test['total_increased'] = self.df_test['total_sqrt15'] > self.df_test['total_sqrt']

    

    def check_sqft_ratio(self):

        self.df_train['sqft_ratio'] = self.df_train['sqft_living'] / self.df_train['sqft_lot']

        self.df_test['sqft_ratio'] = self.df_test['sqft_living'] / self.df_test['sqft_lot']

    

    def check_sqft15_ratio(self):

        self.df_train['sqft_ratio15'] = self.df_train['sqft_living15'] / self.df_train['sqft_lot15']

        self.df_test['sqft_ratio15'] = self.df_test['sqft_living15'] / self.df_test['sqft_lot15']

    

    def check_floor_area_ratio(self):

        self.df_train['floor_area_ratio'] = self.df_train['sqft_living'] / self.df_train['sqft_lot']

        self.df_test['floor_area_ratio'] = self.df_test['sqft_living'] / self.df_test['sqft_lot']



    def check_sqft_floor(self):

        self.df_train['sqft_floor'] = self.df_train['sqft_above'] / self.df_train['floors']

        self.df_test['sqft_floor'] = self.df_test['sqft_above'] / self.df_test['floors']



    def check_is_renovated(self):

        self.df_train['is_renovated'] = self.df_train['yr_renovated'] - self.df_train['yr_built']

        self.df_train['is_renovated'] = self.df_train['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

        self.df_test['is_renovated'] = self.df_test['yr_renovated'] - self.df_test['yr_built']

        self.df_test['is_renovated'] = self.df_test['is_renovated'].apply(lambda x: 0 if x == 0 else 1)



    def check_how_old(self):

        self.df_train['how_old'] = self.df_train['year'] - self.df_train[['yr_built', 'yr_renovated']].max(axis=1)

        self.df_test['how_old'] = self.df_test['year'] - self.df_test[['yr_built', 'yr_renovated']].max(axis=1)



    def total_rooms(self):

        self.df_train['total_rooms'] = self.df_train['bedrooms'] + self.df_train['bathrooms']

        self.df_test['total_rooms'] = self.df_test['bedrooms'] + self.df_test['bathrooms']



    def total_size(self):

        self.df_train['sqft_total_size'] = self.df_train['sqft_above'] + self.df_train['sqft_basement']

        self.df_test['sqft_total_size'] = self.df_test['sqft_above'] + self.df_test['sqft_basement']

 

    def total_sqrt(self):

        self.df_train['total_sqrt'] = self.df_train['sqft_living'] + self.df_train['sqft_lot']

        self.df_test['total_sqrt'] = self.df_test['sqft_living'] + self.df_test['sqft_lot']

    

    def total_sqrt15(self):

        self.df_train['total_sqrt15'] = self.df_train['sqft_living15'] + self.df_train['sqft_lot15']

        self.df_test['total_sqrt15'] = self.df_test['sqft_living15'] + self.df_test['sqft_lot15']



    def delete_outlier(self):

        self.df_train = self.df_train.loc[self.df_train['id']!=456]

        self.df_train = self.df_train.loc[self.df_train['id']!=2302]

        self.df_train = self.df_train.loc[self.df_train['id']!=4123]

        self.df_train = self.df_train.loc[self.df_train['id']!=7259]

        self.df_train = self.df_train.loc[self.df_train['id']!=2777]

        self.df_train = self.df_train.loc[self.df_train['id']!=8990]



    def zipcode_per_price(self):

        self.df_train['per_price'] = self.df_train['price']/self.df_train['sqft_total_size']

        # self.df_train['year_month'] = self.df_train['year'].str + self.df_train['month'].str

        # self.df_test['year_month'] = self.df_test['year'].str + self.df_train['month'].str

        zipcode_price_per_month = self.df_train.groupby(['zipcode', 'year', 'month'])['per_price'].agg({'mean','count'}).reset_index()

        self.df_train = pd.merge(self.df_train, zipcode_price_per_month, how='left', on=['zipcode', 'year', 'month'])

        self.df_test = pd.merge(self.df_test, zipcode_price_per_month, how='left', on=['zipcode', 'year', 'month'])

        self.df_train['mean'] = self.df_train['mean'] * self.df_train['sqft_total_size']

        self.df_test['mean'] = self.df_test['mean'] * self.df_test['sqft_total_size']



    def apply_log(self):

        log_list = [

            'price',

            'long',

            'lat',

            'bathrooms',

            'sqft_living',

            'sqft_lot',

            'sqft_above',

            'sqft_basement',

            'sqft_living15',

            'sqft_lot15',

            'total_sqrt',

            'total_sqrt15',

            'mean',

            'sqft_floor',

            'floor_area_ratio'

        ]

        for name in log_list:

            self.df_train[name] = np.log(self.df_train[name] + 1)

            if name is not 'price':

                self.df_test[name] = np.log(self.df_test[name] + 1) # df_test doesn't contain `price`



    def get_train_testset(self):

        df_train_dummy = self.df_train

        df_test_dummy = self.df_test

        train_feature = self.train_feature



        x_train = df_train_dummy.drop([feature for feature in df_train_dummy.columns if feature not in train_feature ], axis=1)

        x_test = df_test_dummy.drop([feature for feature in df_test_dummy.columns if feature not in train_feature ], axis=1)

        y_train = df_train_dummy['price']



        x_train = pd.get_dummies(x_train)

        x_test =  pd.get_dummies(x_test)

        #y_train = y_train.values.reshape(-1,1)

        

        if self.label_encode:

            le = LabelEncoder()

            le.fit(x_train['zipcode'])

            le.fit(x_test['zipcode'])



            x_train['zipcode'] = le.transform(x_train['zipcode'])

            x_test['zipcode'] = le.transform(x_test['zipcode'])



        if self.apply_scale:

            x_scaler = StandardScaler().fit(x_train)

            y_scaler = StandardScaler().fit(y_train)



            x_train = x_scaler.transform(x_train)

            x_test = x_scaler.transform(x_test)

            y_train = y_scaler.transform(y_train)



            self.scaler = y_scaler

        return x_train, x_test, y_train

    

    def get_scaler(self):

        return self.scaler

import os

import subprocess

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from tensorflow.keras import models

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from tensorflow.keras import backend as K



from lightgbm import LGBMRegressor

from sklearn.linear_model import ElasticNet

from catboost import CatBoostRegressor

from xgboost import XGBRegressor

import lightgbm as lgb

import xgboost as xgb





class KerasModel():

    '''

        TODO:

        - k-fold 다른 방식으로 구현하기.

        - 

    '''

    def __init__(self, x_train, x_test, y_train, y_scaler, output='output.csv'):

        self.output = output

        

        self.x_train = x_train

        self.x_test = x_test

        self.y_train = y_train



        self.y_scaler = y_scaler



        self.model_path = './model/'

        if not os.path.exists(self.model_path):

            os.mkdir(self.model_path)



        self.model_path1 = os.path.join(self.model_path, 'best_model1.hdf5')

        self.model_path2 = os.path.join(self.model_path, 'best_model2.hdf5')



        self.epoch = 200

        self.patient = 20

        self.k = 5

        self.num_val_samples = len(self.x_train) // self.k



        self.callbacks1 = [

            EarlyStopping(monitor='val_loss', patience=self.patient, mode='min', verbose=1),

            ModelCheckpoint(filepath=self.model_path1, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),

            ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = self.patient / 3, min_lr=0.000001, verbose=1, mode='min'),

            TensorBoard(log_dir='./output/graph/model1_2', histogram_freq=0, write_graph=True, write_images=True)

        ]



        self.callbacks2 = [

            EarlyStopping(monitor='val_loss', patience=self.patient, mode='min', verbose=1),

            ModelCheckpoint(filepath=self.model_path2, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),

            ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = self.patient / 3, min_lr=0.000001, verbose=1, mode='min'),

            TensorBoard(log_dir='./output/graph/model2_2', histogram_freq=0, write_graph=True, write_images=True)

        ]



    def make_model1(self):

        model = models.Sequential()

        model.add(layers.Dense(8, activation='relu',	input_dim=self.x_train.shape[1]))

        model.add(layers.Dense(8, activation='relu'))

        model.add(layers.Dense(8, activation='relu'))

        model.add(layers.Dense(1))



        optimizer = optimizers.Adam(lr=0.001)

        model.compile(optimizer= optimizer, loss = root_mean_squared_error, metrics=[root_mean_squared_error, 'mse', 'mae'])

        return model



    def make_model2(self):

        model = models.Sequential()

        model.add(layers.Dense(8, activation='relu',	input_dim=self.x_train.shape[1]))

        model.add(layers.Dense(8, activation='relu'))

        model.add(layers.Dense(8, activation='relu'))

        model.add(layers.Dense(1))



        optimizer = optimizers.RMSprop(lr=0.001)

        model.compile(optimizer= optimizer, loss = root_mean_squared_error, metrics=[root_mean_squared_error, 'mse', 'mae'])

        return model

    

    def start(self):

        for i in range(self.k):

            print('Fold num #', i+1)

            val_data = self.x_train[i * self.num_val_samples : (i+1) * self.num_val_samples]

            val_targets = self.y_train[i * self.num_val_samples : (i+1) * self.num_val_samples]



            partial_train_data = np.concatenate(

                [self.x_train[:i*self.num_val_samples],

                self.x_train[(i+1) * self.num_val_samples:]],

                axis=0

            )

            partial_train_targets = np.concatenate(

                [self.y_train[:i*self.num_val_samples],

                self.y_train[(i+1) * self.num_val_samples:]],

                axis=0

            )

            

            model1 = self.make_model1()

            model2 = self.make_model2()

            

            history1 = model1.fit(

                partial_train_data, 

                partial_train_targets,

                validation_data=(val_data, val_targets), 

                epochs=self.epoch, 

                batch_size=16, 

                callbacks=self.callbacks1)



            history2 = model2.fit(

                partial_train_data, 

                partial_train_targets,

                validation_data=(val_data, val_targets), 

                epochs=self.epoch, 

                batch_size=16, 

                callbacks=self.callbacks2)



        best_model1 = self.make_model1()

        best_model1.load_weights('./model/best_model1.hdf5')

        best_model1.fit(

            self.x_train, 

            self.y_train, 

            epochs=self.epoch, 

            batch_size=8, 

            shuffle=True, 

            validation_split=0.2,

            callbacks=self.callbacks1

            )



        best_model2 = self.make_model2()

        best_model2.load_weights('./model/best_model2.hdf5')

        best_model2.fit(

            self.x_train, 

            self.y_train, 

            epochs=self.epoch, 

            batch_size=8, 

            shuffle=True, 

            validation_split=0.2,

            callbacks=self.callbacks2

            )



        y_preds_best1 = best_model1.predict(self.x_test)

        inv_y_preds_best1 =  np.expm1(self.y_scaler.inverse_transform(y_preds_best1))



        y_preds_best2 = best_model2.predict(self.x_test)

        inv_y_preds_best2 =  np.expm1(self.y_scaler.inverse_transform(y_preds_best2))



        avg_pred = ( inv_y_preds_best1 + inv_y_preds_best2 - 2) / 2

        avg_pred = avg_pred.astype(int)



        self.result = avg_pred

        self.save()

    

    def save(self):

        sub = pd.read_csv('./dataset/sample_submission.csv')

        sub['price'] = self.result

        sub.to_csv(self.output, index=False)



    def submit(self, message='submit'):

        command = 'kaggle competitions submit -c 2019-2nd-ml-month-with-kakr -f {} -m {}'.format(self.output, message)

        subprocess.call (command, shell=True)



class CatBoostModel():

    '''

        TODO:

        - k-fold 다른 방식으로 구현하기.

        - 

    '''

    def __init__(self, x_train, x_test, y_train, y_scaler, output='output.csv', apply_log=False):

        self.output = output

        

        self.x_train = pd.DataFrame(x_train)

        self.x_test = pd.DataFrame(x_test)

        self.y_train = pd.DataFrame(y_train)



        self.y_scaler = y_scaler

        self.apply_log = apply_log



        self.model_path = './model/'

        if not os.path.exists(self.model_path):

            os.mkdir(self.model_path)





    def start(self):

        self.cat_model = CatBoostRegressor(iterations=250, learning_rate=0.05, depth=5)

        self.cat_model.fit(self.x_train, self.y_train)

        

        y_preds= self.cat_model.predict(self.x_test)



        if self.apply_log:

            inv_y_preds =  np.expm1(self.y_scaler.inverse_transform(y_preds)) - 1

        else:

            inv_y_preds = self.y_scaler.inverse_transform(y_preds)

            

        self.result = inv_y_preds.astype(int)

        

        

    def save(self):

        sub = pd.read_csv('./dataset/sample_submission.csv')

        sub['price'] = self.result

        sub.to_csv(self.output, index=False)



    def evaluate(self):

        print('final score is {}'.format(evaluate_prediction(self.result)))

        

    def submit(self, message='submit'):

        command = 'kaggle competitions submit -c 2019-2nd-ml-month-with-kakr -f {} -m {}'.format(self.output, message)

        subprocess.call (command, shell=True)



class LGBMRegressorModel():

    '''

        TODO:

        - k-fold 다른 방식으로 구현하기.

        - 

    '''

    def __init__(self, x_train, x_test, y_train, y_scaler=None, output='output.csv', apply_log=False):

        self.output = output

        

        self.x_train = x_train

        self.x_test = x_test

        self.y_train = y_train



        self.y_scaler = y_scaler



        self.apply_log = apply_log



        self.model_path = './model/'

        if not os.path.exists(self.model_path):

            os.mkdir(self.model_path)





    def start(self):

        param = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 0

        }



        folds = KFold(n_splits=5, shuffle=True, random_state=0)

        oof = np.zeros(len(self.x_train))

        predictions = np.zeros(len(self.x_test))

        feature_importance_df = pd.DataFrame()



        for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.x_train)):

            trn_data = lgb.Dataset(self.x_train.iloc[trn_idx], label=self.y_train.iloc[trn_idx])#, categorical_feature=categorical_feats)

            val_data = lgb.Dataset(self.x_train.iloc[val_idx], label=self.y_train.iloc[val_idx])#, categorical_feature=categorical_feats)



            num_round = 10000

            clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 100)

            oof[val_idx] = clf.predict(self.x_train.iloc[val_idx], num_iteration=clf.best_iteration)



            #predictions

            predictions += clf.predict(self.x_test, num_iteration=clf.best_iteration) / folds.n_splits

            

        cv = np.sqrt(mean_squared_error(oof, self.y_train))

        print(predictions)

        

        y_preds = predictions

        if self.apply_log:

            if self.y_scaler is None:

                inv_y_preds =  np.expm1(y_preds)

            else:

                inv_y_preds =  np.expm1(self.y_scaler.inverse_transform(y_preds))

        else:

            if self.y_scaler is None:

                inv_y_preds = y_preds

            else:

                inv_y_preds = self.y_scaler.inverse_transform(y_preds)

        self.result = inv_y_preds.astype(int)

        

    def save(self):

        sub = pd.read_csv('./dataset/sample_submission.csv')

        sub['price'] = self.result

        sub.to_csv(self.output, index=False)

        

    def evaluate(self):

        print('final score is {}'.format(evaluate_prediction(self.result)))



    def submit(self, message='submit'):

        command = 'kaggle competitions submit -c 2019-2nd-ml-month-with-kakr -f {} -m {}'.format(self.output, message)

        subprocess.call (command, shell=True)



class XGBGressor():

    def __init__(self, x_train, x_test, y_train, y_scaler=None, output='output.csv', apply_log=False, seed=0):

        self.output = output

        self.seed = seed

        self.x_train = x_train

        self.x_test = x_test

        self.y_train = y_train



        self.y_scaler = y_scaler



        self.apply_log = apply_log

    

    def rmse_exp(self, predictions, dmat):

        labels = dmat.get_label()

        diffs = np.expm1(predictions) - np.expm1(labels)

        squared_diffs = np.square(diffs)

        avg = np.mean(squared_diffs)

        return ('rmse_exp', np.sqrt(avg))



    def start(self):

        x_train = xgb.DMatrix(self.x_train, self.y_train)

        x_test = xgb.DMatrix(self.x_test)



        xgb_params = {

            'eta': 0.02,

            'max_depth': 6,

            'subsample': 0.8,

            'colsample_bytree': 0.4,

            'objective': 'reg:linear',    # 회귀

            'eval_metric': 'rmse',        # kaggle에서 요구하는 검증모델

            'silent': True,               # 학습 동안 메세지 출력할지 말지

            'seed': self.seed

        }



        cv_output = xgb.cv(xgb_params,

                   x_train,                        

                   num_boost_round=5000,         # 학습 횟수

                   early_stopping_rounds=100,    # overfitting 방지

                   nfold=8,                    # 높을 수록 실제 검증값에 가까워지고 낮을 수록 빠름

                   verbose_eval=100,             # 몇 번째마다 메세지를 출력할 것인지

                   feval=self.rmse_exp,               # price 속성을 log scaling 했기 때문에, 다시 exponential

                   maximize=False,

                   show_stdv=False,              # 학습 동안 std(표준편차) 출력할지 말지

                   )



        # scoring

        best_rounds = cv_output.index.size

        score = round(cv_output.iloc[-1]['test-rmse_exp-mean'], 2)

        print(f'\nBest Rounds: {best_rounds}')

        print(f'Best Score: {score}')

        print(self.x_train.columns)



        # plotting

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

        cv_output[['train-rmse-mean', 'test-rmse-mean']].plot(ax=ax1)

        ax1.set_title('RMSE_log', fontsize=20)

        cv_output[['train-rmse_exp-mean', 'test-rmse_exp-mean']].plot(ax=ax2)

        ax2.set_title('RMSE', fontsize=20)



        plt.show()



        self.model = xgb.train(xgb_params, x_train, num_boost_round=best_rounds)

        

        y_preds = self.model.predict(x_test)

        if self.apply_log:

            if self.y_scaler is None:

                inv_y_preds =  np.expm1(y_preds)

            else:

                inv_y_preds =  np.expm1(self.y_scaler.inverse_transform(y_preds))

        else:

            if self.y_scaler is None:

                inv_y_preds = y_preds

            else:

                inv_y_preds = self.y_scaler.inverse_transform(y_preds)

        self.result = inv_y_preds.astype(int)

        

    def save(self):

        sub = pd.read_csv('./input/2019-2nd-ml-month-with-kakr/sample_submission.csv')

        sub['price'] = self.result

        sub.to_csv(self.output, index=False)

        

    def evaluate(self):

        print('final score is {}'.format(evaluate_prediction(self.result)))



    def submit(self, message='submit'):

        command = 'kaggle competitions submit -c 2019-2nd-ml-month-with-kakr -f {} -m {}'.format(self.output, message)

        subprocess.call (command, shell=True)

train_feature = [

            #'date',

            'year',

            'month',

#             'day',

            'bedrooms',

            'bathrooms',

            'sqft_living',

            'sqft_lot',

            'floors',

            'waterfront',

            'view',

            'condition',

            'grade',

            'sqft_above',

            'sqft_basement',

            'yr_built',

            'yr_renovated',

            'zipcode',

            'lat',

            'long',

            'sqft_living15',

            'sqft_lot15',

#             'sqft_floor',

            'floor_area_ratio',

            # 'is_renovated',

#             'total_increased',

            # 'sqft_ratio',

            # 'sqft_ratio15',

            # 'total_sqrt',

            # 'total_sqrt15',

#             'total_rooms',

            # 'sqft_total_size',

            'mean',

            # 'living_increased',

            # 'has_basement',

            # 'attic',

            'how_old',

        ]



dataset = Dataset(train_feature, apply_log=True, apply_scale=False, label_encode=False)
x_train, x_test, y_train = dataset.get_train_testset()
xgb_model = XGBGressor(x_train, x_test, y_train, y_scaler=None, output='submission.csv', apply_log=True, seed=0)
xgb_model.start()
lgb_model = LGBMRegressorModel(x_train, x_test, y_train, output='output_8.csv', apply_log=True)
lgb_model.start()
sub = pd.read_csv('../input/sample_submission.csv')

sub['price'] = 0.9*xgb_model.result + 0.1*lgb_model.result

sub.to_csv('result.csv', index=False)