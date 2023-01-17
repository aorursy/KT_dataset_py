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



from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score

import gc
def read_data():

    print('Reading train.csv file....')

    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission



def get_time(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    return df



def get_object_columns(df, columns):

    df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()

    df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')

    df.columns = list(df.columns)

    df.fillna(0, inplace = True)

    return df



def get_numeric_columns(df, column):

    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'std']})

    df.fillna(0, inplace = True)

    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_std']

    return df



def get_numeric_columns_2(df, agg_column, column):

    df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'std']}).reset_index()

    df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])

    df.fillna(0, inplace = True)

    df.columns = list(df.columns)

    return df



def get_correct_incorrect(df):

    df = df.groupby(['title'])['num_correct', 'num_incorrect'].agg({'num_correct': ['mean', 'std'], 'num_incorrect': ['mean', 'std']}).reset_index()

    df.columns = ['title', 'num_correct_mean', 'num_correct_std', 'num_incorrect_mean', 'num_incorrect_std']

    return df



def preprocess(train, test, train_labels):

    # columns for feature engineering

    numerical_columns = ['game_time']

    categorical_columns = ['type', 'world']



    reduce_train = pd.DataFrame({'installation_id': train['installation_id'].unique()})

    reduce_train.set_index('installation_id', inplace = True)

    reduce_test = pd.DataFrame({'installation_id': test['installation_id'].unique()})

    reduce_test.set_index('installation_id', inplace = True)

    

    for i in numerical_columns:

        reduce_train = reduce_train.merge(get_numeric_columns(train, i), left_index = True, right_index = True)

        reduce_test = reduce_test.merge(get_numeric_columns(test, i), left_index = True, right_index = True)

    

    for i in categorical_columns:

        reduce_train = reduce_train.merge(get_object_columns(train, i), left_index = True, right_index = True)

        reduce_test = reduce_test.merge(get_object_columns(test, i), left_index = True, right_index = True)



    for i in categorical_columns:

        for j in numerical_columns:

            reduce_train = reduce_train.merge(get_numeric_columns_2(train, i, j), left_index = True, right_index = True)

            reduce_test = reduce_test.merge(get_numeric_columns_2(test, i, j), left_index = True, right_index = True)

            

    reduce_train.reset_index(inplace = True)

    reduce_test.reset_index(inplace = True)

    

    print('Our training set have {} rows and {} columns'.format(reduce_train.shape[0], reduce_train.shape[1]))

    

    # get the mode of the title

    labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))

    # merge target

    labels = train_labels[['installation_id', 'title', 'accuracy_group']]

    # merge with correct incorrect

    corr_inc = get_correct_incorrect(train_labels)

    labels = labels.merge(corr_inc, how = 'left', on = 'title')

    # replace title with the mode

    labels['title'] = labels['title'].map(labels_map)

    # get title from the test set

    reduce_test['title'] = test.groupby('installation_id').last()['title'].reset_index(drop = True)

    # merge with correct incorrect

    reduce_test = reduce_test.merge(corr_inc, how = 'left', on = 'title')

    # map title

    reduce_test['title'] = reduce_test['title'].map(labels_map)

    # join train with labels

    reduce_train = labels.merge(reduce_train, on = 'installation_id', how = 'left')

    print('We have {} training rows'.format(reduce_train.shape[0]))

    # align datasets

    categoricals = ['title']

    reduce_train = reduce_train[[col for col in reduce_test.columns] + ['accuracy_group']]

    return reduce_train, reduce_test, categoricals



# function to perform features selection, not calling it to optimize time, best features are in a list in the next cell

def run_feature_selection(reduce_train, reduce_test):

    kf = KFold(n_splits=10, random_state = 42)

    all_columns = ['num_correct_mean', 'num_correct_std', 'num_incorrect_mean', 'num_incorrect_std', 

                     'game_time_mean', 'game_time_sum', 'game_time_std', 'Activity', 'Assessment', 'Clip', 'Game', 'CRYSTALCAVES', 

                     'MAGMAPEAK', 'NONE', 'TREETOPCITY', ('game_time', 'mean', 'Activity'), 

                     ('game_time', 'mean', 'Assessment'), ('game_time', 'mean', 'Clip'), ('game_time', 'mean', 'Game'),  

                     ('game_time', 'std', 'Activity'), ('game_time', 'std', 'Assessment'), ('game_time', 'std', 'Clip'), 

                     ('game_time', 'std', 'Game'), ('game_time', 'sum', 'Activity'), ('game_time', 'sum', 'Assessment'), 

                     ('game_time', 'sum', 'Clip'), ('game_time', 'sum', 'Game'), ('game_time', 'mean', 'CRYSTALCAVES'), 

                     ('game_time', 'mean', 'MAGMAPEAK'), ('game_time', 'mean', 'NONE'), ('game_time', 'mean', 'TREETOPCITY'), 

                     ('game_time', 'std', 'CRYSTALCAVES'), ('game_time', 'std', 'MAGMAPEAK'), ('game_time', 'std', 'NONE'), 

                     ('game_time', 'std', 'TREETOPCITY'), ('game_time', 'sum', 'CRYSTALCAVES'), 

                     ('game_time', 'sum', 'MAGMAPEAK'), ('game_time', 'sum', 'NONE'), ('game_time', 'sum', 'TREETOPCITY')]

    target = 'accuracy_group'

    oof_pred = np.zeros((len(reduce_train), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train)):

        print('Fold {}'.format(fold + 1))

        x_train, x_val = reduce_train[all_columns].iloc[tr_ind], reduce_train[all_columns].iloc[val_ind]

        y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train, categorical_feature = categoricals)

        val_set = lgb.Dataset(x_val, y_val, categorical_feature = categoricals)



        params = {

                'learning_rate': 0.01,

                'metric': 'multiclass',

                'objective': 'multiclass',

                'num_classes': 4,

                'feature_fraction': 0.75,

                'subsample': 0.75,

                'n_jobs': -1,

            }



        model = lgb.train(params, train_set, num_boost_round = 100000, early_stopping_rounds = 100, 

                          valid_sets=[train_set, val_set], verbose_eval = 100)

        oof_pred[val_ind] = model.predict(x_val)

    # using cohen_kappa because it's the evaluation metric of the competition

    loss_score = cohen_kappa_score(reduce_train[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')

    score = loss_score

    best_features = all_columns.copy()

    for i in all_columns:

        oof_pred = np.zeros((len(reduce_train), 4))

        features = [x for x in best_features if x not in [i]]

        print('Evaluating {} column'.format(i))

        for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train)):

            print('Fold {}'.format(fold + 1))

            x_train, x_val = reduce_train[features].iloc[tr_ind], reduce_train[features].iloc[val_ind]

            y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]

            train_set = lgb.Dataset(x_train, y_train, categorical_feature = categoricals)

            val_set = lgb.Dataset(x_val, y_val, categorical_feature = categoricals)



            model = lgb.train(params, train_set, num_boost_round = 100000, early_stopping_rounds = 100, 

                              valid_sets=[train_set, val_set], verbose_eval = 100)

            oof_pred[val_ind] = model.predict(x_val)

        loss_score = cohen_kappa_score(reduce_train[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')

        if loss_score > score:

            print('Feature {} is useless'.format(i))

            best_features.remove(i)

            score = loss_score

        else:

            print('Feature {} is usefull'.format(i))

        gc.collect()

    print('The best features are: ', best_features + 'title')



    return best_features + 'title'



train, test, train_labels, specs, sample_submission = read_data()

reduce_train, reduce_test, categoricals = preprocess(train, test, train_labels)

#best_features = run_feature_selection(reduce_train, reduce_test)
# best features extracted from run_feature_selection function

usefull_features = ['num_correct_mean', 'num_correct_std', 'num_incorrect_mean', 'num_incorrect_std',

'game_time_mean', 'game_time_sum', 'Activity', 'Clip', 'Game', 'CRYSTALCAVES', 'NONE',

'TREETOPCITY', ('game_time', 'mean', 'Clip'), ('game_time', 'mean', 'Game'), 

('game_time', 'std', 'Assessment'), ('game_time', 'std', 'Clip'), ('game_time', 'std', 'Game'),

('game_time', 'sum', 'Activity'), ('game_time', 'sum', 'Clip'), ('game_time', 'sum', 'Game'),

('game_time', 'mean', 'NONE'), ('game_time', 'mean', 'TREETOPCITY'), ('game_time', 'std', 'CRYSTALCAVES'),

('game_time', 'std', 'MAGMAPEAK'), ('game_time', 'std', 'NONE'), ('game_time', 'std', 'TREETOPCITY'),

('game_time', 'sum', 'CRYSTALCAVES'), ('game_time', 'sum', 'MAGMAPEAK'), 

('game_time', 'sum', 'NONE'), 'title']
def run_lgb(reduce_train, reduce_test, usefull_features):

    kf = KFold(n_splits=10)

    target = 'accuracy_group'

    oof_pred = np.zeros((len(reduce_train), 4))

    y_pred = np.zeros((len(reduce_test), 4))

    for fold, (tr_ind, val_ind) in enumerate(kf.split(reduce_train)):

        print('Fold {}'.format(fold + 1))

        x_train, x_val = reduce_train[usefull_features].iloc[tr_ind], reduce_train[usefull_features].iloc[val_ind]

        y_train, y_val = reduce_train[target][tr_ind], reduce_train[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train, categorical_feature=categoricals)

        val_set = lgb.Dataset(x_val, y_val, categorical_feature=categoricals)



        params = {

            'learning_rate': 0.01,

            'metric': 'multiclass',

            'objective': 'multiclass',

            'num_classes': 4,

            'feature_fraction': 0.75,

            'subsample': 0.75,

            'n_jobs': -1

        }



        model = lgb.train(params, train_set, num_boost_round = 100000, early_stopping_rounds = 100, 

                          valid_sets=[train_set, val_set], verbose_eval = 100)

        oof_pred[val_ind] = model.predict(x_val)

        y_pred += model.predict(reduce_test[usefull_features]) / 10

    loss_score = cohen_kappa_score(reduce_train[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')

    print('Our oof cohen kappa score is: ', loss_score)

    return y_pred



def predict(reduce_test, sample_submission, y_pred):

    reduce_test = reduce_test.reset_index()

    reduce_test = reduce_test[['installation_id']]

    reduce_test['accuracy_group'] = y_pred.argmax(axis = 1)

    sample_submission.drop('accuracy_group', inplace = True, axis = 1)

    sample_submission = sample_submission.merge(reduce_test, on = 'installation_id')

    sample_submission.to_csv('submission.csv', index = False)

    print(sample_submission['accuracy_group'].value_counts(normalize = True))

y_pred = run_lgb(reduce_train, reduce_test, usefull_features)

predict(reduce_test, sample_submission, y_pred)