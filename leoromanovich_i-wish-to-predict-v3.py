import gc

import os

import sys

import time

import pickle

import random

import numpy as np

import pandas as pd

from numba import jit



import seaborn as sns

from itertools import product

import matplotlib.pyplot as plt

from collections import OrderedDict

from tqdm import tqdm_notebook as tqdm





# ML libs

from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer

from sklearn.model_selection import KFold, ShuffleSplit

from sklearn import metrics



# Models which were used

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge

from catboost import CatBoostRegressor

import lightgbm as lgbm



import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import Dataset, DataLoader

from torchvision import transforms





# friendship of matplotlib and jupyter

%matplotlib inline



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)

print(os.listdir("../input"))

print(sys.version_info)



# Fixing seeds

np.random.seed(0)

random.seed(0)

RANDOM_SEED = 0
@jit

def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())



def trainer(X,

                X_test,

                y,

                params,

                folds,

                columns=None,

                eval_metric='mae'

                ):

#     columns = X.columns if columns is None else columns



    result_dict = {}



    # out-of-fold predictions on train data

    oof = np.zeros(len(X))



    # averaged predictions on test data

    prediction = np.zeros(len(X_test))



    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()



    # to set up scoring parameters

    metrics_dict = {'mae': {'lgb_metric_name': 'mae',

                            'catboost_metric_name': 'MAE',

                            'sklearn_scoring_function': metrics.mean_absolute_error},

                    'group_mae': {'lgb_metric_name': 'mae',

                                  'catboost_metric_name': 'MAE'

                                 },

                    'mse': {'lgb_metric_name': 'mse',

                            'catboost_metric_name': 'MSE',

                            'sklearn_scoring_function': metrics.mean_squared_error}

                    }



    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):



        print(f'Fold {fold_n + 1} started at {time.ctime()}')



        model = lgbm.LGBMRegressor(**params, n_jobs = -1)

            

        if type(X) == np.ndarray:

            X_train, X_valid = X[train_index], X[valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

                    verbose=0, early_stopping_rounds=300)

            

        y_pred_valid = model.predict(X_valid)

        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)   





        oof[valid_index] = y_pred_valid.reshape(-1, )

        if eval_metric != 'group_mae':

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

        else:

            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))



        prediction += y_pred



    prediction /= folds.n_splits



    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))



    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores



    return result_dict



def print_info(dataset, name):

    s = "\n" + "-"*70 + '\n'

    

    print(f"{s}",f"Info about {name} with shape {dataset.shape}",

          f"{s}", dataset.head(),

          f"{s}", dataset.nunique(),

          f"{s}", dataset.describe().astype('int32'),

          f"{s}",'Count NaN values \n', dataset.isna().sum(),

         )



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (

                start_mem - end_mem) / start_mem))

    return df
nrows = None

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

try:

    train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train_v2.csv', nrows=nrows)

except:

    train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', nrows=nrows)

# set index to ID to avoid droping it later

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
print_info(train, 'train')

print_info(test, 'test')
print(f"{len(set(train.shop_id) - set(test.shop_id))} shops only in train.")

print(f"{len(train.item_cnt_day[train.item_cnt_day < 0])} negative items_cnt_day")

print(f"We need to predict clipped into [0, 20] range item_cnt_month column")
# Fixing outliers

train = train[train.item_price<48000]

train = train[train.item_cnt_day<1000]
# Item count per day (target) visualization



plt.figure(figsize=(20,5))

fig, ax = plt.subplots(figsize=(20,5))

g = sns.boxplot(train.item_cnt_day, palette="Set3", ax=ax)

plt.show()
# Costs of each product

plt.figure(figsize=(20,5))

fig, ax = plt.subplots(figsize=(20,5))

g = sns.boxplot(train.item_price, palette="Set3", ax=ax)

plt.show()
print_info(cats, "Categories")
cats
# Create category, subcategory features

cats.loc[32, "item_category_name"] = 'Карты оплаты - (Кино, Музыка, Игры)'

cats['split'] = cats['item_category_name'].str.split('-')

cats['type'] = cats['split'].map(lambda x: x[0].strip())

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

# if subtype is nan then type

cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats = cats[['item_category_id','type_code', 'subtype_code']]

cats
print_info(shops, "Shops")
shops
# Fixing some names

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]
# Histogram of shops

xticks = [set(train.shop_id.unique()).union(test.shop_id.unique())]

plt.figure(figsize=(20,10))

fig, ax = plt.subplots(figsize=(20,10))

g = sns.distplot(train.shop_id, kde=False, label="Train", ax=ax)

sns.distplot(test.shop_id, kde=False, label="Test")

plt.legend()

plt.show()
print_info(items, 'Items')
items.head()
train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
# Make matrix of all possible combinations of shops and items.

matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train[train.date_block_num==i]

    matrix.append(np.array(list(product([i], 

                                sales.shop_id.unique(), 

                                sales.item_id.unique())), 

                                dtype='int16'))



matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['item_cnt_month'] = (matrix['item_cnt_month']

                                .fillna(0)

                                .clip(0,20) # NB clip target here

                                .astype(np.float16))
test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True) # 34 month
train.head()
matrix.head()
# Add Shops, Items, Categories features

matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')

matrix = pd.merge(matrix, items, on=['item_id'], how='left')

matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')

matrix['city_code'] = matrix['city_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['type_code'] = matrix['type_code'].astype(np.int8)

matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

matrix = matrix.drop(['item_name'], axis=1)
# I used here smaller lag (second). Reason: RAM. But it allows me achieve lower score.



# lookback_range = list(range(1, 33 + 1)) # Takes lots of RAM

lookback_range = list(range(1, 7 + 1))

new_features = []



# Add lag features



# Add previous shop/item sales as feature

# How much item sell in past time wrt to each shop

for diff in tqdm(lookback_range):

    feature_name = 'prev_shopitem_sales_' + str(diff)

    mx2 = matrix.copy()

    mx2.loc[:, 'date_block_num'] += diff

    mx2.rename(columns={'item_cnt_month': feature_name}, inplace=True)

    matrix = matrix.merge(mx2[['shop_id', 'item_id', 'date_block_num', feature_name]], on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')

    matrix[feature_name] = matrix[feature_name].fillna(0)

    new_features.append(feature_name)



# Add previous item sales as feature

# How much item sell in past time

groups = matrix.groupby(by = ['item_id', 'date_block_num'])

for diff in tqdm(lookback_range):

    feature_name = 'prev_item_sales_' + str(diff)

    result = groups.agg({'item_cnt_month':'mean'})

    result = result.reset_index()

    result.loc[:, 'date_block_num'] += diff

    result.rename(columns={'item_cnt_month': feature_name}, inplace=True)

    matrix = matrix.merge(result, on = ['item_id', 'date_block_num'], how = 'left')

    matrix[feature_name] = matrix[feature_name].fillna(0)

    new_features.append(feature_name)



groups = matrix.groupby(by = ['item_id', 'date_block_num'])

for diff in tqdm(lookback_range):

    feature_name = '_prev_item_sales_' + str(diff)

    result = groups.agg({'item_cnt_month':'mean'})

    result = result.reset_index()

    result.loc[:, 'date_block_num'] += diff

    result.rename(columns={'item_cnt_month': feature_name}, inplace=True)

    matrix = matrix.merge(result, on = ['item_id', 'date_block_num'], how = 'left')

    matrix[feature_name] = matrix[feature_name].fillna(0)

    new_features.append(feature_name)  
# This part of code may be a little bit unclear, but it has very simple logic:

# First: count unique values in chosen categorical column

# Second: divide each unique value by lengh of column (here we have a mean)

# Third: create column with mapped column <-> mean values



me_cols = ['shop_id', 'item_id', 'city_code', 'item_category_id', 'type_code', 'subtype_code']

for cl in me_cols:

    me_col_name = "me_" + cl

    matrix.loc[:, me_col_name] = matrix[cl].map(matrix[cl].value_counts().apply(lambda x: x/len(matrix[cl])))
matrix.head()
# Reducing of memory usage

import gc

del items, shops, cats, train



matrix = reduce_mem_usage(matrix)

test = reduce_mem_usage(test)
X_train = matrix[matrix.date_block_num <= 33].drop(['item_cnt_month'], axis=1)

y = matrix[matrix.date_block_num <= 33]['item_cnt_month']

X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del matrix

gc.collect()
# Before we start to fit models, we need to preprocess our dataset (Not necessary for forests).

x_stsc = StandardScaler().fit(pd.concat([X_train, X_test]))

X_train = x_stsc.transform(X_train)

X_test = x_stsc.transform(X_test)
cv = KFold(3, shuffle=True)
def linear_model(

                X,

                X_test,

                y,

                model,

                params,

                folds,

                columns=None):



    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):



        print(f'Fold {fold_n + 1} started at {time.ctime()}')



        result_dict = {}



        # out-of-fold predictions on train data

        oof = np.zeros(len(X))



        # averaged predictions on test data

        prediction = np.zeros(len(X_test))



        # list of scores on folds

        scores = []

        feature_importance = pd.DataFrame()



        X_train, X_valid = X[train_index], X[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]



        model.fit(X_train, y_train)



        y_pred_valid = model.predict(X_valid)

        y_pred = model.predict(X_test)



        oof[valid_index] = y_pred_valid.reshape(-1, )

        

        scores.append(rmse(y_valid, y_pred_valid))



        prediction += y_pred

        

        gc.collect()

    

    prediction /= folds.n_splits



    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))



    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores



    return result_dict
linreg_best_params= {

    'n_jobs': -1

}

ridge_best_params = {

    'alpha': 0.7,

    'solver': 'lsqr',

    'random_state': RANDOM_SEED

}

lr_model = LinearRegression(**linreg_best_params)

ridge_model = Ridge(**ridge_best_params)



linreg_results = linear_model(X_train, X_test, y,

                             lr_model,

                             linreg_best_params,

                             cv)



ridge_results = linear_model(X_train, X_test, y,

                             ridge_model,

                             linreg_best_params,

                             cv)

gc.collect()
def random_forest_regressor(

                X,

                X_test,

                y,

                params,

                folds,

                columns=None):



    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):



        print(f'Fold {fold_n + 1} started at {time.ctime()}')



        result_dict = {}



        # out-of-fold predictions on train data

        oof = np.zeros(len(X))



        # averaged predictions on test data

        prediction = np.zeros(len(X_test))



        # list of scores on folds

        scores = []

        feature_importance = pd.DataFrame()





        if type(X) == np.ndarray:

            X_train, X_valid = X[train_index], X[valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]







        model = RandomForestRegressor(**params)



        model.fit(X_train, y_train)



        y_pred_valid = model.predict(X_valid)

        y_pred = model.predict(X_test)



        oof[valid_index] = y_pred_valid.reshape(-1, )

        

        scores.append(rmse(y_valid, y_pred_valid))



        prediction += y_pred

        gc.collect()

    prediction /= folds.n_splits



    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))



    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    gc.collect()

    return result_dict



# Here I use smaller number of estimators. 4 instead 20. I decided to do this, wrt to number of cores in notebook.

# rf_best_params= {

#  'bootstrap': True,

#  'max_depth': 70,

#  'max_features': 'auto',

#  'min_samples_leaf': 4,

#  'min_samples_split': 10,

#  'n_estimators': 4,

#  'n_jobs': -1,

#  'verbose': 2

# }



rf_best_params= {

 'bootstrap': True,

 'max_depth': 40,

 'max_features': 'auto',

 'min_samples_leaf': 4,

 'min_samples_split': 10,

 'n_estimators': 4,

 'n_jobs': -1,

 'verbose': 2

}



rf_results = random_forest_regressor(X_train, X_test, y, 

                                     rf_best_params,

                                     cv)
params= {

    'num_leaves': 64,

    "max_depth": 8,

    'max_bin': 32,

    "iterations": 200,

    "data_random_seed":0,

    'n_estimators': 1000

}

lgb_results = trainer(X_train, X_test, y, params, cv)
oof_preds = np.array([ridge_results['oof'],

                      linreg_results['oof'],

                      rf_results['oof'],

                      lgb_results['oof']

                     ]).T



preds = np.array([ridge_results['prediction'],

                  linreg_results['prediction'],

                  rf_results['prediction'],

                  lgb_results['prediction']

                 ]).T
# Read all generated predictions



lvl2_model = LinearRegression()

lvl2_model.fit(oof_preds, y)

lvl2_model.score(oof_preds, y)

lvl2_result = lvl2_model.predict(preds)
submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": lvl2_result

})

submission['item_cnt_month'] = submission['item_cnt_month'].clip(0, 20)

# submission.to_csv('lvl2.1.csv',index=False)

submission.to_csv('submission.csv',index=False)
# !pip install pytorch-ignite
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

# from ignite.metrics import RootMeanSquaredError, Loss

# from ignite.handlers import ModelCheckpoint, EarlyStopping

# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

# os.environ["CUDA_VISIBLE_DEVICES"]="2"



# class FSPNet(nn.Module):

#     def __init__(self, input_dim):

#         super(FSPNet, self).__init__()

        

#         self.l1 = nn.Linear(input_dim, 512)

#         self.l2 = nn.Linear(512, 1024)

#         self.l3 = nn.Linear(1024, 256)

#         self.l4 = nn.Linear(256, 1)

        

#     def forward(self, x):

#         x = F.relu(self.l1(x))

#         x = F.relu(self.l2(x))

#         x = F.relu(self.l3(x))

#         x = self.l4(x)

#         return x

    

# class FSPDataset(Dataset):



#     def __init__(self, X, y):

#         self.X = X

#         self.y = y

        

#     def __len__(self):

#         return len(self.y)

    

#     def __getitem__(self, idx):

#         return torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float()



# class FSPDataset_test(Dataset):

#     def __init__(self, X):

#         self.X = X

    

#     def __len__(self):

#         return len(self.X)

    

#     def __getitem__(self, idx):

#         return torch.tensor(self.X[idx]).float()
# def mlp_trainer(

#                 X,

#                 X_test,

#                 y,

#                 folds,

#                 epochs=15):



#     for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):



#         print(f'Fold {fold_n + 1} started at {time.ctime()}')



#         result_dict = {}



#         # out-of-fold predictions on train data

#         oof = np.zeros(len(X))



#         # averaged predictions on test data

#         prediction = np.zeros(len(X_test))



#         # list of scores on folds

#         scores = []

        

#         y = y.values

#         if type(X) == np.ndarray:

#             X_train, X_valid = X[train_index], X[valid_index]

#             y_train, y_valid = y[train_index], y[valid_index]

#         else:

            

#             X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

#             y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]





#         train_dataset = FSPDataset(X_train, y_train)

#         valid_dataset = FSPDataset(X_valid, y_valid)

        

#         train_loader = DataLoader(train_dataset, batch_size=512, 

#                                   shuffle=True, num_workers=5)

#         valid_loader = DataLoader(valid_dataset, batch_size=512,

#                                   shuffle=True, num_workers=5)

        

#         # init model and move to CPU or GPU

#         model = FSPNet(X_train.shape[1])

#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         print(device)

#         model.to(device)

        

#         # Init optimizers

#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#         criterion = nn.MSELoss()

        

#         trainer = create_supervised_trainer(model, 

#                                             optimizer, 

#                                             criterion, 

#                                             device=device)

        

#         metrics = {

#             'RMSE':RootMeanSquaredError(),

#             'MSELoss':Loss(criterion)



#         }

#         evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

#         training_history = {'RMSE':[],'loss':[]}

#         validation_history = {'RMSE':[],'loss':[]}

#         last_epoch = []

        

#         RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

        

#         def score_function(engine):

#             val_loss = engine.state.metrics['MSELoss']

#             return -val_loss



#         handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)

#         evaluator.add_event_handler(Events.COMPLETED, handler)

        

#         @trainer.on(Events.EPOCH_COMPLETED)

#         def log_training_results(trainer):

#             evaluator.run(train_loader)

#             metrics = evaluator.state.metrics

#             accuracy = metrics['RMSE']

#             loss = metrics['MSELoss']

#             last_epoch.append(0)

#             training_history['RMSE'].append(accuracy)

#             training_history['loss'].append(loss)

#             print("Training Results - Epoch: {}  Avg RMSE: {:.2f} Avg MSELoss: {:.2f}"

#                   .format(trainer.state.epoch, accuracy, loss))



#         def log_validation_results(trainer):

#             evaluator.run(valid_loader)

#             metrics = evaluator.state.metrics

#             accuracy = metrics['RMSE']

#             loss = metrics['MSELoss']

#             validation_history['RMSE'].append(accuracy)

#             validation_history['loss'].append(loss)

#             print("Validation Results - Epoch: {}  Avg RMSE: {:.2f} Avg MSELoss: {:.2f}"

#                   .format(trainer.state.epoch, accuracy, loss))



#         trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

        

#         print('Training started...')

#         trainer.run(train_loader, max_epochs=epochs)

                

        

        

#         y_pred_valid = model.forward(torch.tensor(X_valid).float().to('cpu'))

        

#         y_pred = model.forward(torch.tensor(X_test).float().to('cpu'))



#         oof[valid_index] = y_pred_valid.detach().numpy().reshape(-1, )

        

#         scores.append(rmse(y_valid, y_pred_valid.detach().numpy()))



#         prediction += y_pred



#     prediction /= folds.n_splits



#     print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))



#     result_dict['oof'] = oof

#     result_dict['prediction'] = prediction

#     result_dict['scores'] = scores
# mlp_trainer(X_train,

#             X_test, y,

#             cv, epochs=1)
