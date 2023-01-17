# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!git clone https://github.com/LenzDu/Kaggle-Competition-Sberbank Utils
import sys

 # Add directory holding utility functions to path to allow importing utility funcitons

#sys.path.insert(0, '/kaggle/working/protein-atlas-fastai')

sys.path.append('/kaggle/working/Utils')
# -*- coding: utf-8 -*-

"""

Created on Wed May 17 16:36:14 2017

@author: vrtjso

"""

import numpy as np

import pandas as pd

from datetime import datetime, date

from operator import le, eq

from Utils import sample_vals, FeatureCombination

import gc

from sklearn import model_selection, preprocessing

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression



####Data Cleaning####

print('Data Cleaning...')



#Data importing

trainDf = pd.read_csv('../input/train.csv').set_index('id')

testDf = pd.read_csv('../input/test.csv').set_index('id')

fix = pd.read_excel('../input/BAD_ADDRESS_FIX.xlsx').set_index('id')

testDf['isTrain'] = 0

trainDf['isTrain'] = 1

allDf = pd.concat([trainDf,testDf])

allDf.update(fix, filter_func = lambda x:np.array([True]*x.shape[0])) #update fix data

macro = pd.read_csv('../input/macro.csv')



#Join division and macro





### Change price by rate ###

allDf['timestamp'] = pd.to_datetime(allDf['timestamp'])



allDf['apartment_name'] = allDf.sub_area + allDf['metro_km_avto'].astype(str)

eco_map = {'excellent':4, 'good':3, 'satisfactory':2, 'poor':1, 'no data':0}

allDf['ecology'] = allDf['ecology'].map(eco_map)

#encode subarea in order

# price_by_area = allDf['price_doc'].groupby(allDf.sub_area).mean().sort_values()

# area_dict = {}

# for i in range(0,price_by_area.shape[0]):

#    area_dict[price_by_area.index[i]] = i

# allDf['sub_area'] = allDf['sub_area'].map(area_dict)

for c in allDf.columns:

    if allDf[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(allDf[c].values))

        allDf[c] = lbl.transform(list(allDf[c].values))





###Dealing with Outlier###

allDf.loc[allDf.full_sq>2000,'full_sq'] = np.nan

allDf.loc[allDf.full_sq<3,'full_sq'] = np.nan

allDf.loc[allDf.life_sq>500,'life_sq'] = np.nan

allDf.loc[allDf.life_sq<3,'life_sq'] = np.nan

# allDf['lifesq_to_fullsq'] = 0 # 0 for normal, 1 for close,2 for outlier

allDf.loc[allDf.life_sq>0.8*allDf.full_sq,'life_sq'] = np.nan

# allDf.ix[allDf.life_sq>allDf.full_sq,['life_sq','lifesq_to_fullsq']] = np.nan, 2

allDf.loc[allDf.kitch_sq>=allDf.life_sq,'kitch_sq'] = np.nan

allDf.loc[allDf.kitch_sq>500,'kitch_sq'] = np.nan

allDf.loc[allDf.kitch_sq<2,'kitch_sq'] = np.nan

allDf.loc[allDf.state>30,'state'] = np.nan

allDf.loc[allDf.build_year<1800,'build_year'] = np.nan

allDf.loc[allDf.build_year==20052009,'build_year'] = 2005

allDf.loc[allDf.build_year==4965,'build_year'] = np.nan

allDf.loc[allDf.build_year>2021,'build_year'] = np.nan

allDf.loc[allDf.num_room>15,'num_room'] = np.nan

allDf.loc[allDf.num_room==0,'num_room'] = np.nan

allDf.loc[allDf.floor==0,'floor'] = np.nan

allDf.loc[allDf.max_floor==0,'max_floor'] = np.nan

allDf.loc[allDf.floor>allDf.max_floor,'max_floor'] = np.nan

#allDf.ix[allDf.full_sq>300,'full_sq'] = np.nan

#allDf.ix[allDf.life_sq>250,'life_sq'] = np.nan



# brings error down a lot by removing extreme price per sqm

bad_index = allDf[allDf.price_doc/allDf.full_sq > 600000].index

bad_index = bad_index.append(allDf[allDf.price_doc/allDf.full_sq < 10000].index)

allDf.drop(bad_index,0,inplace=True)



####Feature Engineering####

print('Feature Engineering...')

gc.collect()



# allDf['month'] = np.array(month)

allDf['year'] = allDf.timestamp.dt.year  #may be no use because test data is out of range

allDf['weekday'] = allDf.timestamp.dt.weekday



#allDf['week_of_year'] = np.array(week_of_year)

##allDf['year_month'] = np.array(year_month)



#w_map = {2011:0.8, 2012:0.8, 2013:0.9, 2014:1, 2015:1, 2016:0}

#allDf['w'] = [w_map[i] for i in year]



# Assign weight

allDf['w'] = 1

allDf.loc[allDf.price_doc==1000000,'w'] *= 0.5

allDf.loc[allDf.year==2015,'w'] *= 1.5



#Floor

allDf['floor_by_max_floor'] = allDf.floor / allDf.max_floor

#allDf['floor_to_top'] = allDf.max_floor - allDf.floor



#Room

allDf['avg_room_size'] = (allDf.life_sq - allDf.kitch_sq) / allDf.num_room

allDf['life_sq_prop'] = allDf.life_sq / allDf.full_sq

allDf['kitch_sq_prop'] = allDf.kitch_sq / allDf.full_sq



#Calculate age of building

allDf['build_age'] = allDf.year - allDf.build_year

allDf = allDf.drop('build_year', 1)



#Population

allDf['popu_den'] = allDf.raion_popul / allDf.area_m

allDf['gender_rate'] = allDf.male_f / allDf.female_f

allDf['working_rate'] = allDf.work_all / allDf.full_all



#Education

allDf.loc[allDf.preschool_quota==0,'preschool_quota'] = np.nan

allDf['preschool_ratio'] =  allDf.children_preschool / allDf.preschool_quota

allDf['school_ratio'] = allDf.children_school / allDf.school_quota



## Group statistics

allDf['square_full_sq'] = (allDf.full_sq - allDf.full_sq.mean()) ** 2

allDf['square_build_age'] = (allDf.build_age - allDf.build_age.mean()) ** 2

allDf['nan_count'] = allDf[['full_sq','build_age','life_sq','floor','max_floor','num_room']].isnull().sum(axis=1)

allDf['full*maxfloor'] = allDf.max_floor * allDf.full_sq

allDf['full*floor'] = allDf.floor * allDf.full_sq



allDf['full/age'] = allDf.full_sq / (allDf.build_age + 0.5)

allDf['age*state'] = allDf.build_age * allDf.state



# new trial

allDf['main_road_diff'] = allDf['big_road2_km'] - allDf['big_road1_km']

allDf['rate_metro_km'] = allDf['metro_km_walk'] / allDf['ID_metro'].map(allDf.metro_km_walk.groupby(allDf.ID_metro).mean().to_dict())

allDf['rate_road1_km'] = allDf['big_road1_km'] / allDf['ID_big_road1'].map(allDf.big_road1_km.groupby(allDf.ID_big_road1).mean().to_dict())

# best on LB with weekday



allDf['rate_road2_km'] = allDf['big_road2_km'] / allDf['ID_big_road2'].map(allDf.big_road2_km.groupby(allDf.ID_big_road2).mean().to_dict())

allDf['rate_railroad_km'] = allDf['railroad_station_walk_km'] / allDf['ID_railroad_station_walk'].map(allDf.railroad_station_walk_km.groupby(allDf.ID_railroad_station_walk).mean().to_dict())

# increase CV from 2.35 to 2.33 but lower LB a little bit (with month)



allDf.drop(['year','timestamp'], 1, inplace = True)



#Separate train and test again

trainDf = allDf[allDf.isTrain==1].drop(['isTrain'],1)

testDf = allDf[allDf.isTrain==0].drop(['isTrain','price_doc', 'w'],1)



outputFile = 'train_featured.csv'

trainDf.to_csv(outputFile,index=False)

outputFile = 'test_featured.csv'

testDf.to_csv(outputFile,index=False)



# Xgboost handles nan itself

'''

### Dealing with NA ###

#num_room, filled by linear regression of full_sq

if filename == 'train_encoded.csv': #na in num_room only appear in training set

    LR = LinearRegression()

    X = allDf.full_sq[~(np.isnan(allDf.num_room) | np.isnan(allDf.full_sq))].values.reshape(-1, 1)

    y = np.array(allDf.num_room[~(np.isnan(allDf.num_room) | np.isnan(allDf.full_sq))])

    LR.fit(X,y)

    newX = allDf.full_sq[np.isnan(allDf.num_room)].values.reshape(-1, 1)

    newX[np.isnan(newX)] = newX[~np.isnan(newX)].mean() #Special cases (na in full_sq) in test data

    yfit = LR.predict(newX)

    allDf.ix[np.isnan(allDf.num_room),'num_room'] = yfit

#max_floor, twice as the floor

allDf.ix[np.isnan(allDf.max_floor),'max_floor'] = allDf.ix[np.isnan(allDf.max_floor),'floor'] * 2

'''
# author: vrtjso



import numpy as np

import pandas as pd

from sklearn.model_selection import ShuffleSplit, cross_val_score

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import Imputer

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

import xgboost as xgb

import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
n_folds = 5
# 封装一下lightgbm让其可以在stacking里面被调用

class LGBregressor(object):

    def __init__(self,params):

        self.params = params



    def fit(self, X, y, w):

        y /= 10000000

        # self.scaler = StandardScaler().fit(y)

        # y = self.scaler.transform(y)

        split = int(X.shape[0] * 0.8)

        indices = np.random.permutation(X.shape[0])

        train_id, test_id = indices[:split], indices[split:]

        x_train, y_train, w_train, x_valid, y_valid,  w_valid = X[train_id], y[train_id], w[train_id], X[test_id], y[test_id], w[test_id],

        d_train = lgb.Dataset(x_train, y_train, weight=w_train)

        d_valid = lgb.Dataset(x_valid, y_valid, weight=w_valid)

        partial_bst = lgb.train(self.params, d_train, 10000, valid_sets=d_valid, early_stopping_rounds=50, verbose_eval=500)

        num_round = partial_bst.best_iteration

        d_all = lgb.Dataset(X, label = y, weight=w)

        self.bst = lgb.train(self.params, d_all, num_round)



    def predict(self, X):

        return self.bst.predict(X) * 10000000

        # return self.scaler.inverse_transform(self.bst.predict(X))
# 封装一下xgboost让其可以在stacking里面被调用

class XGBregressor(object):

    def __init__(self, params):

        self.params = params



    def fit(self, X, y, w=None):

        #if (w==None):

        #    w = np.ones(X.shape[0])

        split = int(X.shape[0] * 0.8)

        indices = np.random.permutation(X.shape[0])

        train_id, test_id = indices[:split], indices[split:]

        x_train, y_train, w_train, x_valid, y_valid,  w_valid = X[train_id], y[train_id], w[train_id], X[test_id], y[test_id], w[test_id],

        d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)

        d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        partial_bst = xgb.train(self.params, d_train, 10000, early_stopping_rounds=50, evals = watchlist, verbose_eval=500)

        num_round = partial_bst.best_iteration

        d_all = xgb.DMatrix(X, label = y, weight=w)

        self.bst = xgb.train(self.params, d_all, num_round)



    def predict(self, X):

        test = xgb.DMatrix(X)

        return self.bst.predict(test)
# This object modified from Wille on https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/

class Ensemble(object):

    def __init__(self, stacker, base_models):

        self.stacker = stacker

        self.base_models = base_models



    def fit_predict(self, trainDf, testDf):

        X = trainDf.drop(['price_doc', 'w'], 1).values

        y = trainDf['price_doc'].values

        w = trainDf['w'].values

        T = testDf.values



        X_fillna = trainDf.drop(['price_doc', 'w'], 1).fillna(-999).values

        T_fillna = testDf.fillna(-999).values



        kfold = KFold(n_splits=n_folds, shuffle=True)

        S_train = np.zeros((X.shape[0], len(self.base_models)))

        S_test = np.zeros((T.shape[0], len(self.base_models)))

        

        for i, clf in enumerate(self.base_models):

            print('Training base model ' + str(i+1) + '...')

            

            S_test_i = np.zeros((T.shape[0], n_folds))

            j=0

            for train_idx, test_idx in kfold.split(X):

                print('Training round...' + str(j+1) + '...')

            #for j, (train_idx, test_idx) in enumerate(folds):

                if clf not in [xgb1,lgb1]: # sklearn models cannot handle missing values.

                    X = X_fillna

                    T = T_fillna

                X_train = X[train_idx]

                y_train = y[train_idx]

                w_train = w[train_idx]

                X_holdout = X[test_idx]

                # w_holdout = w[test_idx]

                # y_holdout = y[test_idx]

                clf.fit(X_train, y_train, w_train)

                y_pred = clf.predict(X_holdout)

                S_train[test_idx, i] = y_pred

                S_test_i[:, j] = clf.predict(T)

                j=j+1

            S_test[:, i] = S_test_i.mean(1)

        

        #self.S_train, self.S_test, self.y = S_train, S_test, y  # for diagnosis purpose

        self.corr = pd.concat([pd.DataFrame(S_train),trainDf['price_doc']],1).corr() # correlation of predictions by different models.

        # cv_stack = ShuffleSplit(n_splits=6, test_size=0.2)

        # score_stacking = cross_val_score(self.stacker, S_train, y, cv=cv_stack, n_jobs=1, scoring='neg_mean_squared_error')

        # print(np.sqrt(-score_stacking.mean())) # CV result of stacking

        print(S_train.shape)

        print(y.shape)

        

        self.stacker.fit(S_train, y, w)

        y_pred = self.stacker.predict(S_test)

        return y_pred
trainDf = pd.read_csv('/kaggle/working/train_featured.csv')#.sample(frac=0.01)

testDf = pd.read_csv('/kaggle/working/test_featured.csv')#.sample(frac=0.01)
params1 = {'eta':0.05, 'max_depth':5, 'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':1,

          'gamma':0, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse'}

xgb1 = XGBregressor(params1)
params2 = {'booster':'gblinear', 'alpha':0,# for gblinear, delete this line if change back to gbtree

           'eta':0.1, 'max_depth':2, 'subsample':1, 'colsample_bytree':1, 'min_child_weight':1,

          'gamma':0, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse'}

xgb2 = XGBregressor(params2)
#X = trainDf.drop(['price_doc', 'w'], 1).values

#y = trainDf['price_doc'].values

#w = trainDf['w'].values

#X.shape

#xgb2.fit(X, y, w)
RF = RandomForestRegressor(n_estimators=500, max_features=0.2)

ETR = ExtraTreesRegressor(n_estimators=500, max_features=0.3, max_depth=None)

Ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=15),n_estimators=200)

GBR = GradientBoostingRegressor(n_estimators=200,max_depth=5,max_features=0.5)

LR =LinearRegression()



params_lgb = {'objective':'regression','metric':'rmse',

          'learning_rate':0.05,'max_depth':-1,'sub_feature':0.7,'sub_row':1,

          'num_leaves':15,'min_data':30,'max_bin':20,

          'bagging_fraction':0.9,'bagging_freq':40,'verbosity':0}

lgb1 = LGBregressor(params_lgb)



E = Ensemble(xgb2, [xgb1,lgb1,RF,ETR,Ada,GBR])   
prediction = E.fit_predict(trainDf, testDf)
output = pd.read_csv('../input/test.csv')

output = output[['id']]

output['price_doc'] = prediction

output.to_csv(r'Submission_Stack.csv',index=False)



# corr = pd.concat([pd.DataFrame(S_train),trainDf['price_doc']],1).corr() # extract correlation

# 1: 2434 2: 2421
!rm -rf /kaggle/working/Utils
#LB 0.38848 > 0.31448 > 0.31412