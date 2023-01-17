"""

Public LB: 0.50456

"""

from collections import Counter, defaultdict



import lightgbm as lgb

import ml_metrics

import numpy as np

import pandas as pd

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

import optuna





def parse_host_verifications(df):

    raw_value_list = []

    for val in df['host_verifications'].tolist():

        values = eval(val)

        if values is not None:

            raw_value_list.append(Counter(values))

        else:

            raw_value_list.append({})



    vectorizer = DictVectorizer(sparse=False)

    X = vectorizer.fit_transform(raw_value_list)

    for idx, col in enumerate(vectorizer.feature_names_):

        df[f'host_verifications_{col}'] = X[:, idx]
def load_data():

    df_trn = pd.read_csv('/kaggle/input/tmu-inclass-competition/train.csv')

    df_tst = pd.read_csv('/kaggle/input/tmu-inclass-competition/test.csv')

    df = pd.concat([df_trn, df_tst], sort=False)

    original_train_size = len(df_trn)

    y_train = df.iloc[:original_train_size]['price'].values



    # calendar_updated の処理  'a week ago' などの文字列を日数の int (この場合は7) に変換

    repl_day = lambda m: m.group(1)

    repl_week = lambda m: str(int(m.group(1)) * 7)

    repl_month = lambda m: str(int(m.group(1)) * 30)

    df.calendar_updated = df.calendar_updated.str.replace('today', '0')

    df.calendar_updated = df.calendar_updated.str.replace('yesterday', '1')

    df.calendar_updated = df.calendar_updated.str.replace(r'^(\d)\sday(.+)ago$', repl_day, regex=True)

    df.calendar_updated = df.calendar_updated.str.replace('a week ago', '7')

    df.calendar_updated = df.calendar_updated.str.replace(r'^(\d)\sweek(.+)ago$', repl_week, regex=True)

    df.calendar_updated = df.calendar_updated.str.replace('a month ago', '30')

    df.calendar_updated = df.calendar_updated.str.replace(r'^(\d{1,2})\smonth(.+)ago$', repl_month, regex=True)

    df.calendar_updated = df.calendar_updated.str.replace('never', '10000')

    df.calendar_updated = df.calendar_updated.apply(lambda x: int(x))



    # host_response_time の処理

    df.host_response_time = df.host_response_time.str.replace('within an hour', '1')

    df.host_response_time = df.host_response_time.str.replace('within a few hours', '3')

    df.host_response_time = df.host_response_time.str.replace('within a day', '24')

    df.host_response_time = df.host_response_time.str.replace('a few days or more', '120')

    df.host_response_time = df.host_response_time.fillna('58').apply(lambda x: int(x))





    repl = lambda m: m.group(1)

    df.host_response_rate = df.host_response_rate.str.replace(r'^(\d+)%$', repl, regex=True)

    df.host_response_rate = df.host_response_rate.fillna('80').apply(lambda x: int(x))



    df.host_since = df.host_since.str.replace('-', '/')

    df.loc[:, 'host_since_year'] = df.host_since.fillna('2020/01/01').apply(

        lambda x: int(x.split('/')[0]))

    df.loc[:, 'host_since_month'] = df.host_since.fillna('2020/01/01').apply(

        lambda x: int(x.split('/')[1]))

    df.loc[:, 'host_since_day'] = df.host_since.fillna('2020/01/01').apply(

        lambda x: int(x.split('/')[2]))



    # Parse host_verifications

    parse_host_verifications(df)



    # Baseline features

    categorical_cols = []

    cols = []

    for col in df.columns:

        if col in ['listing_id', 'price', 'host_since', 'host_acceptance_rate', 'square_feet', 'picture_url']:

            continue



        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(df[col].mean())

        else:

            df[col] = df[col].factorize()[0]

            categorical_cols.append(col)

        cols.append(col)



    return df, df_trn, df_tst, y_train, cols, categorical_cols
def cv(trial):

    np_random_seed = trial.suggest_int("np_random_seed", 0, 1000)

    np.random.seed(np_random_seed)

    # params['boosting_type'] = trial.suggest_categorical('boosting_type', ['gbdt', 'rf', 'dart', 'goss'])

    # params['objective'] = trial.suggest_categorical('objective', ['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie'])

    params['num_leaves'] = trial.suggest_int("num_leaves", 1, 1000)

    params['num_iterations'] = trial.suggest_int("num_iterations", 1, 1000)

    params['learning_rate'] = trial.suggest_uniform("learning_rate", 1e-7, 1e-1)

    params['tree_learner'] = trial.suggest_categorical('tree_learner', ['serial', 'feature', 'data', 'voting'])

    params['seed'] = trial.suggest_int("seed", 0, 1000)

    # params['max_depth'] = trial.suggest_int("max_depth", -1, 2000)

    # params['min_data_in_leaf'] = trial.suggest_int("min_data_in_leaf", 0, 2000)

    # params['min_sum_hessian_in_leaf'] = trial.suggest_uniform("min_sum_hessian_in_leaf", 0, 1)



    params['bagging_fraction'] = trial.suggest_uniform("bagging_fraction", 0.001, 0.999)

    params['bagging_freq'] = trial.suggest_int("bagging_freq", 1, 2000)

    params['bagging_seed'] = trial.suggest_int("bagging_seed", 0, 2000)



    params['feature_fraction'] = trial.suggest_uniform("feature_fraction", 0, 1)

    params['feature_fraction_bynode'] = trial.suggest_uniform("feature_fraction_bynode", 0, 1)

    params['feature_fraction_seed'] = trial.suggest_int("feature_fraction_seed", 0, 2000)



    fit_params['num_boost_round'] = trial.suggest_int("num_boost_round", 0, 2000)

    fit_params['verbose_eval'] = trial.suggest_int("verbose_eval", 0, 2000)



    df, df_trn, df_tst, y_train, cols, categorical_cols = load_data()

    original_train_size = y_train.shape[0]

    X_train = df.iloc[:original_train_size][cols].values



    val_score_list = []

    kf = KFold(n_splits=3, random_state=11, shuffle=True)

    for idx_valtrn, idx_valtst in kf.split(X_train):

        X_valtrn, X_valtst = X_train[idx_valtrn], X_train[idx_valtst]

        y_valtrn, y_valtst = y_train[idx_valtrn], y_train[idx_valtst]



        lgb_valtrn = lgb.Dataset(X_valtrn, np.log1p(y_valtrn),

                                 feature_name=cols,

                                 categorical_feature=categorical_cols)

        lgb_eval = lgb.Dataset(X_valtst, np.log1p(y_valtst),

                               reference=lgb_valtrn,

                               feature_name=cols,

                               categorical_feature=categorical_cols)



        fit_params['valid_sets'] = lgb_eval

        clf = lgb.train(params, lgb_valtrn, **fit_params)



        y_pred = np.expm1(clf.predict(X_valtst,

                                      num_iteration=clf.best_iteration))

        val_score = ml_metrics.rmsle(y_pred, y_valtst)

        print(f'RMSLE: {val_score:.6f}')

        val_score_list.append(val_score)



    avg_val_score = np.mean(val_score_list)

    print(f'Avg-RMSLE: {avg_val_score:.6f}')

    return avg_val_score
def main():

    np.random.seed(np_random_seed)

    df, df_trn, df_tst, y_train, cols, categorical_cols = load_data()

    original_train_size = y_train.shape[0]



    X_train = df.iloc[:original_train_size][cols].values

    X_test = df.iloc[original_train_size:][cols].values

    # Early stopping のための validation split を作成

    X_valtrn, X_valtst, y_valtrn, y_valtst = train_test_split(

        X_train, y_train, test_size=0.1, random_state=11)



    lgb_valtrn = lgb.Dataset(X_valtrn, np.log1p(y_valtrn),

                             feature_name=cols,

                             categorical_feature=categorical_cols)

    lgb_eval = lgb.Dataset(X_valtst, np.log1p(y_valtst),

                           reference=lgb_valtrn,

                           feature_name=cols,

                           categorical_feature=categorical_cols)



    fit_params['valid_sets'] = lgb_eval

    clf = lgb.train(params, lgb_valtrn, **fit_params)



    y_pred = np.expm1(clf.predict(X_valtst, num_iteration=clf.best_iteration))

    print('RMSLE: {:.6f}'.format(ml_metrics.rmsle(y_pred, y_valtst)))



    y_pred = np.expm1(clf.predict(X_test, num_iteration=clf.best_iteration))

    df_tst.loc[:, 'price'] = y_pred

    df_tst[['listing_id', 'price']].to_csv('./optuna.csv', index=False)
if __name__ == '__main__':

    # LightGBM parameters

    # https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst#core-parameters

    params = defaultdict(int)

    params['task'] = 'train'

    params['boosting_type'] = 'gbdt'

    params['objective'] = 'regression'

    params['metric'] = 'rmse'

    params['num_leaves'] = 60

    params['learning_rate'] = 0.1

    params['feature_fraction'] = 1.0

    # params['bagging_fraction'] = 1.0

    params['verbose'] = -1

    params['num_threads'] = 10  # 使用する CPU の個数

    # https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst#learning-control-parameters

    fit_params = defaultdict(int)

    fit_params['num_boost_round'] = 8

    fit_params['verbose_eval'] = 8

    fit_params['early_stopping_rounds'] = 3

    

    study = optuna.create_study()

    study.optimize(cv, n_trials=1)

    print(study.best_params)

    print(study.best_value)

    # cv() 今回は optuna に食わせる時に使うだけなので、そのまま使う必要はない。

    np_random_seed = study.best_params['np_random_seed']

    main()