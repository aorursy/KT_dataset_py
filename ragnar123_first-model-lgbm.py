import numpy as np 

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import gc

warnings.filterwarnings('ignore')

import lightgbm as lgb

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_squared_error, f1_score

from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def read_data():

    print('Reading training, testing and submission data...')

    train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

    test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

    submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time':str})

    print('Train set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    return train, test, submission



train, test, submission = read_data()
# concatenate data

batch = 50

total_batches = 14

train['set'] = 'train'

test['set'] = 'test'

data = pd.concat([train, test])

for i in range(int(total_batches)):

    data.loc[(data['time'] > i * batch) & (data['time'] <= (i + 1) * batch), 'batch'] = i + 1

train = data[data['set'] == 'train']

test = data[data['set'] == 'test']

del data
def preprocess(train, test):

    

    pre_train = train.copy()

    pre_test = test.copy()

    

    batch1 = pre_train[pre_train["batch"] == 1]

    batch2 = pre_train[pre_train["batch"] == 2]

    batch3 = pre_train[pre_train["batch"] == 3]

    batch4 = pre_train[pre_train["batch"] == 4]

    batch5 = pre_train[pre_train["batch"] == 5]

    batch6 = pre_train[pre_train["batch"] == 6]

    batch7 = pre_train[pre_train["batch"] == 7]

    batch8 = pre_train[pre_train["batch"] == 8]

    batch9 = pre_train[pre_train["batch"] == 9]

    batch10 = pre_train[pre_train["batch"] == 10]

    batch11 = pre_test[pre_test['batch'] == 11]

    batch12 = pre_test[pre_test['batch'] == 12]

    batch13 = pre_test[pre_test['batch'] == 13]

    batch14 = pre_test[pre_test['batch'] == 14]

    batches = [batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10, batch11, batch12, batch13, batch14]

    

    for batch in batches:

        for feature in ['signal']:

            # some random rolling features

            for window in [50, 100, 1000, 5000, 10000, 25000]:

                # roll backwards

                batch[feature + 'mean_t' + str(window)] = batch[feature].shift(1).rolling(window).mean()

                batch[feature + 'std_t' + str(window)] = batch[feature].shift(1).rolling(window).std()

                batch[feature + 'min_t' + str(window)] = batch[feature].shift(1).rolling(window).min()

                batch[feature + 'max_t' + str(window)] = batch[feature].shift(1).rolling(window).max()

                min_max = (batch[feature] - batch[feature + 'min_t' + str(window)]) / (batch[feature + 'max_t' + str(window)] - batch[feature + 'min_t' + str(window)])

                batch['norm_t' + str(window)] = min_max * (np.floor(batch[feature + 'max_t' + str(window)]) - np.ceil(batch[feature + 'min_t' + str(window)]))

                

#                 # roll forward

#                 batch[feature + 'mean_t' + str(window) + '_lead'] = batch[feature].shift(- window - 1).rolling(window).mean()

#                 batch[feature + 'std_t' + str(window) +'_lead'] = batch[feature].shift(- window - 1).rolling(window).std()

#                 batch[feature + 'min_t' + str(window) + '_lead'] = batch[feature].shift(- window - 1).rolling(window).min()

#                 batch[feature + 'max_t' + str(window) + '_lead'] = batch[feature].shift(- window - 1).rolling(window).max()

#                 min_max = (batch[feature] - batch[feature + 'min_t' + str(window) + '_lead']) / (batch[feature + 'max_t' + str(window) + '_lead'] - batch[feature + 'min_t' + str(window) + '_lead'])

#                 batch['norm_t' + str(window) + '_lead'] = min_max * (np.floor(batch[feature + 'max_t' + str(window) + '_lead']) - np.ceil(batch[feature + 'min_t' + str(window) + '_lead']))

                

    pre_train = pd.concat([batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10])

    pre_test = pd.concat([batch11, batch12, batch13, batch14])

    

    del batches, batch1, batch2, batch3, batch4, batch5, batch6, batch7, batch8, batch9, batch10, batch11, batch12, batch13, batch14, train, test, min_max

    

    return pre_train, pre_test



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        if col!='open_channels':

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

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df





def scale_fillna(pre_train, pre_test):

    features = [col for col in pre_train.columns if col not in ['open_channels', 'set', 'time', 'batch']]

    pre_train = pre_train.replace([np.inf, -np.inf], np.nan)

    pre_test = pre_test.replace([np.inf, -np.inf], np.nan)

    pre_train.fillna(0, inplace = True)

    pre_test.fillna(0, inplace = True)

#     scaler = StandardScaler()

#     pre_train[features] = scaler.fit_transform(pre_train[features])

#     pre_test[features] = scaler.transform(pre_test[features])

    return pre_train, pre_test



# feature engineering

pre_train, pre_test = preprocess(train, test)

# reduce memory usage

pre_train = reduce_mem_usage(pre_train, verbose=True)

pre_test = reduce_mem_usage(pre_test, verbose=True)

# scaling and filling missing values (this is not required for boosting algorithms, nevertheless i wanted to try and check)

pre_train, pre_test = scale_fillna(pre_train, pre_test)

del train, test

gc.collect()
def run_lgb(pre_train, pre_test, usefull_features, params):

    

    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    target = 'open_channels'

    oof_pred = np.zeros(len(pre_train))

    y_pred = np.zeros(len(pre_test))

    feature_importance = pd.DataFrame()

    

    # train a baseline model and record the weighted cohen kappa score 

    for fold, (tr_ind, val_ind) in enumerate(kf.split(pre_train, pre_train[target])):

        print('Fold {}'.format(fold + 1))

        x_train, x_val = pre_train[usefull_features].iloc[tr_ind], pre_train[usefull_features].iloc[val_ind]

        y_train, y_val = pre_train[target][tr_ind], pre_train[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train)

        val_set = lgb.Dataset(x_val, y_val)

        

        model = lgb.train(params, train_set, num_boost_round = 1000, early_stopping_rounds = 50, 

                         valid_sets = [train_set, val_set], verbose_eval = 100)

        

        oof_pred[val_ind] = model.predict(x_val)

        

        y_pred += model.predict(pre_test[usefull_features]) / kf.n_splits

        

        # get fold importance df

        fold_importance = pd.DataFrame({'features': usefull_features})

        fold_importance['fold'] = fold + 1

        fold_importance['importance'] = model.feature_importance()

        feature_importance = pd.concat([feature_importance, fold_importance])

        

    # round predictions

    rmse_score = np.sqrt(mean_squared_error(pre_train[target], oof_pred))

    print('Our oof rmse score is: ', rmse_score)

    # want to clip and then round predictions

    oof_pred = np.round(np.clip(oof_pred, 0, 10)).astype(int)

    y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)

    cohen_score = cohen_kappa_score(pre_train[target], oof_pred, weights = 'quadratic')

    f1 = f1_score(pre_train[target], oof_pred, average = 'macro')

    print('Our oof cohen kappa score is: ', cohen_score)

    print('Our oof f1_macro score is: ', f1)

    

    # plot feature importance

    fi_mean = feature_importance.groupby(['features'])['importance'].mean().reset_index()

    fi_mean.sort_values('importance', ascending = False, inplace = True)

    plt.figure(figsize = (12, 14))

    sns.barplot(x = fi_mean['importance'], y = fi_mean['features'])

    plt.xlabel('Importance', fontsize = 13)

    plt.ylabel('Feature', fontsize = 13)

    plt.tick_params(axis = 'x', labelsize = 11)

    plt.tick_params(axis = 'y', labelsize = 11)

    plt.title('Light Gradient Boosting Feature Importance (5 KFold)')

    plt.show()

    

    

    return oof_pred, y_pred, feature_importance





# define hyperparammeter (some random hyperparammeters)

params = {'learning_rate': 0.1, 

          'feature_fraction': 0.75, 

          'bagging_fraction': 0.75,

          'bagging_freq': 1,

          'n_jobs': -1, 

          'seed': 50,

          'metric': 'rmse'

        }







# define the features for training

features = [col for col in pre_train.columns if col not in ['open_channels', 'set', 'time', 'batch']]



oof_pred, y_pred, feature_importance = run_lgb(pre_train, pre_test, features, params)
submission.open_channels = y_pred

submission.to_csv("submission.csv",index=False)