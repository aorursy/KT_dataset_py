%matplotlib inline

import pandas as pd

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 99)

pd.set_option('display.max_rows', 99)

import os

import numpy as np

from matplotlib import pyplot as plt

from tqdm import tqdm

import plotly.express as px

from sklearn import metrics

import datetime as dt

from functools import partial

import lightgbm as lgb
start_time = dt.datetime.now()
VERSION = 11

COMP_DIR = '../input/covid19-global-forecasting-week-5/'

GEO_PATH = '../input/covid19belugaw5/geo.csv'

CLOSEST_PATH = '../input/covid19belugaw5/closest.csv'





WORK_DIR = '/kaggle/working/'

LOG_FILE = f'{WORK_DIR}paropt_lgb_v{VERSION}.csv'

FIMP_FILE = f'{WORK_DIR}fimp_lgb_v{VERSION}.csv'

FEATURE_FILE_PATH = f'{WORK_DIR}features_v{VERSION}.csv'

PREDS_PATH = f'{WORK_DIR}predictions_v{VERSION}/'

if not os.path.exists(PREDS_PATH):

    os.makedirs(PREDS_PATH)
os.listdir(WORK_DIR)

os.listdir(COMP_DIR)
TRAIN_START = '2020-03-31'

TRAIN_END = '2020-05-11'

CV = 5

RELOAD_FEATURES = True

PRECISION = 2





def w5_loss(preds, data, q=0.5):

    y_true = data.get_label()

    weights = data.get_weight()



    diff = (y_true - preds) * weights

    gt_is_higher = np.sum(diff[diff >= 0] * q)

    gt_is_lower = np.sum(- diff[diff < 0] * (1 - q))



    return 'w5', (gt_is_higher + gt_is_lower) / len(preds), False
def process_train():

    train = pd.read_csv(COMP_DIR + 'train.csv')

    train = train.fillna('')

    train['Location'] = train.Country_Region + '-' + train.Province_State + '-' + train.County

    train = train.drop(columns=['Id', 'Country_Region', 'Province_State', 'County'])



    pop = train.groupby('Location')[['Population', 'Weight']].mean().reset_index()

    pop['cv'] = np.random.randint(0, CV, len(pop))



    confirmed = train.loc[train.Target == 'ConfirmedCases',

                          ['Location', 'Date', 'TargetValue']].copy()

    confirmed.columns = ['Location', 'Date', 'Confirmed']

    fatalities = train.loc[train.Target == 'Fatalities',

                           ['Location', 'Date', 'TargetValue']].copy()

    fatalities.columns = ['Location', 'Date', 'Fatalities']

    targets = pd.merge(confirmed, fatalities, on=['Location', 'Date'])

    targets = targets.merge(pop, on='Location')



    targets['DateTime'] = pd.to_datetime(targets.Date)

    targets['DaysTillEnd'] = (targets.DateTime.max() - targets.DateTime).dt.days + 1

    targets['DayOfWeek'] = targets.DateTime.dt.weekday

    targets['USCounty'] = 1 * (~targets.Location.str.endswith('-'))



    targets.Confirmed = targets.Confirmed.clip(0, None)

    targets.Fatalities = targets.Fatalities.clip(0, None)

    print(targets.max())

    return targets
def extract_timeseries_features(single_location):

    df = single_location.copy()

    df = df.sort_values(by='Date')

    for target in ['Confirmed', 'Fatalities']:

        df[f'{target}CumSum'] = df[target].cumsum()

        for dow in range(7):

            df[f'{target}DOW{dow}'] = ((df.DayOfWeek == dow) * df[target]).cumsum() / (df[target].cumsum() + 1)

        for k in [3, 7, 14, 21]:

            df[f'{target}RollingMean{k}'] = df[target].rolling(k).mean()

            df[f'{target}RollingMean{k}PerK'] = df[target].rolling(k).mean() / df.Population * 10000

        df[f'{target}RollingStd21'] = df[target].rolling(21).std().round(0)

        df[f'{target}DaysSince10'] = (df[f'{target}CumSum'] > 10).cumsum()

        df[f'{target}DaysSince100'] = (df[f'{target}CumSum'] > 100).cumsum()



        df[f'{target}RollingMeanDiff2w'] = df[f'{target}RollingMean7'] / (df[f'{target}RollingMean14'] + 1) - 1

        df[f'{target}RollingMeanDiff3w'] = df[f'{target}RollingMean7'] / (df[f'{target}RollingMean21'] + 1) - 1



    df['DeathRate'] = 100 * df.FatalitiesCumSum.clip(0, None) / (df.ConfirmedCumSum.clip(0, None) + 1)

    df['DeathRateRolling3w'] = 100 * df.FatalitiesRollingMean7.clip(0, None) / (

            df.ConfirmedRollingMean21.clip(0, None) + 1)

    df['ConfirmedPerK'] = 1000 * df.ConfirmedCumSum.clip(0, None) / df.Population

    df['FatalitiesPerK'] = 1000 * df.FatalitiesCumSum.clip(0, None) / df.Population

    return df
def get_nearby_features(features, rank):

    closest = pd.read_csv(CLOSEST_PATH)



    to_aggregate = ['ConfirmedCumSum',

                    'ConfirmedRollingMean21',

                    'ConfirmedRollingMean14',

                    'ConfirmedRollingMean7',



                    'FatalitiesCumSum',

                    'FatalitiesRollingMean21',

                    'FatalitiesRollingMean14',

                    'FatalitiesRollingMean7']



    subset = features[['Date', 'Location', 'Population'] + to_aggregate].copy()

    subset = subset.rename(columns={'Location': 'Location_1'})



    nearby = features[['Date', 'Location']].merge(closest[closest.Rank <= rank], on='Location')

    nearby = nearby.merge(subset, on=['Date', 'Location_1'])



    nearby_sum = nearby.groupby(['Date', 'Location']).sum()

    nearby_mean = nearby.groupby(['Date', 'Location'])[['distance']].mean().round(0)

    for c in to_aggregate:

        nearby_sum[f'Nearby{rank}{c}'] = 1000 * nearby_sum[c] / nearby_sum['Population']



    nearby_features = pd.merge(nearby_sum, nearby_mean, on=['Date', 'Location'])

    nearby_features = nearby_features.rename(columns={'distance_y': f'Nearby{rank}Distance'})

    return nearby_features[[f for f in nearby_features.columns if f.startswith('Nearby')]]
targets = process_train()



if os.path.exists(FEATURE_FILE_PATH) and RELOAD_FEATURES:

    features = pd.read_csv(FEATURE_FILE_PATH)

else:

    features = []

    for loc, df in tqdm(targets.groupby('Location')):

        df = extract_timeseries_features(df)

        features.append(df)

    features = pd.concat(features)



    geo = pd.read_csv(GEO_PATH)

    features = features.merge(geo, on='Location')

    for rank in [5, 10, 20]:

        nearby_features = get_nearby_features(features, rank)

        features = features.merge(nearby_features, on=['Date', 'Location'])



    to_log = ['ConfirmedCumSum', 'Population', 'FatalitiesCumSum']

    for c in to_log:

        features.loc[:, c] = np.log(features[c].values + 1).round(2)



    features.loc[:, 'DeathRate'] = features.loc[:, 'DeathRate'].clip(0, 50)

    features.loc[:, 'DeathRateRolling3w'] = features.loc[:, 'DeathRateRolling3w'].clip(0, 50)



    round_1_digit = [

        'ConfirmedRollingMean21', 'ConfirmedRollingMean14', 'ConfirmedRollingMean7', 'ConfirmedRollingMean3',

    ]

    for c in round_1_digit:

        features.loc[:, c] = features[c].round(1)



    features = features.round(PRECISION)

    features.to_csv(FEATURE_FILE_PATH, index=False)



features = features[features.Location != 'US--']

# Remove Public LB Future

features = features[features.Date < TRAIN_END]

print(f'Features: {features.shape}')

print(f'Features: {features.count()}')
def apply_lgb(features, target, q, params, k, num_round=1000):

    features['TARGET'] = features.groupby('Location')[target].shift(-k)

    do_not_use = [

                     'Location', 'Date', 'TARGET', 'Weight', 'cv', 'DaysTillEnd', 'DateTime'

                 ] + ['Confirmed', 'Fatalities']

    feature_names = [f for f in features.columns if f not in do_not_use]



    print(features.columns)

    print(len(feature_names), feature_names)



    train = features.loc[(~features.TARGET.isna()) & (features.Date > TRAIN_START)]

    test = features.loc[(features.TARGET.isna()) & (features.Date > TRAIN_START)]

    print(train.shape, test.shape)



    test.loc[:, 'PREDICTION'] = 0

    train.loc[:, 'PREDICTION'] = 0

    feature_importances = []

    for cv in range(CV):

        tr = train[train.cv != cv]

        val = train[train.cv == cv]



        train_set = lgb.Dataset(tr[feature_names], label=tr.TARGET, weight=tr.Weight / tr.DaysTillEnd ** 0.2)

        valid_set = lgb.Dataset(val[feature_names], label=val.TARGET, weight=val.Weight / val.DaysTillEnd ** 0.2)



        model = lgb.train(params, train_set, num_round, valid_sets=[train_set, valid_set],

                          early_stopping_rounds=50, feval=partial(w5_loss, q=q))



        train.loc[train.cv == cv, 'PREDICTION'] = model.predict(val[feature_names])

        test.loc[test.cv == cv, 'PREDICTION'] = model.predict(test.loc[test.cv == cv, feature_names])



        fimp = pd.DataFrame({'f': feature_names, 'imp': model.feature_importance()})

        feature_importances.append(fimp)



    _, error, _ = w5_loss(

        train.PREDICTION,

        lgb.Dataset(train[feature_names], label=train.TARGET, weight=train.Weight / train.DaysTillEnd ** 0.2),

        q

    )

    feature_importances = pd.concat(feature_importances)

    feature_importances = feature_importances.groupby('f').sum().reset_index().sort_values(by='imp', ascending=False)

    feature_importances['target'] = target

    feature_importances['k'] = k

    feature_importances['q'] = q



    train_preds = train[['Date', 'Location', 'PREDICTION']]

    test_preds = test[['Date', 'Location', 'PREDICTION']]

    test_preds['target'] = target

    test_preds['k'] = k

    test_preds['q'] = q

    return error, train_preds, test_preds, feature_importances
for i in range(1):

    for target in ['Confirmed', 'Fatalities']:

        for k in range(1, 15):

            for q in [0.05, 0.5, 0.95]:

                params = dict(

                    objective='quantile',

                    alpha=q,

                    metric='custom',

                    max_depth=np.random.choice([6, 8, 10, 15, 20]),

                    learning_rate=np.random.choice([0.025, 0.05, 0.1]),

                    feature_fraction=np.random.choice([0.5, 0.6, 0.7, 0.8]),

                    bagging_freq=np.random.choice([2, 3, 5]),

                    bagging_fraction=np.random.choice([0.7, 0.8]),

                    min_data_in_leaf=np.random.choice([5, 10]),

                    num_leaves=np.random.choice([127, 255]),

                    verbosity=0,

                    n_jobs=4

                )



                num_round = 10



                start = dt.datetime.now()

                error, train_preds, test_preds, feature_importances = apply_lgb(

                    features, target, q, params, k, num_round=num_round)

                print(error)

                print(train_preds.shape, test_preds.shape)



                preds_file_name = PREDS_PATH + f'preds_{target}_{q}_{k}_{error.round(5)}.csv'

                test_preds.to_csv(preds_file_name, index=False)



                end = dt.datetime.now()

                print('Finished', end, (end - start).seconds, 's')



                result = [

                             target, q, k, error, (end - start).seconds

                         ] + list(params.values())

                columns = [

                              'target', 'q', 'k', 'error', 'train_time'

                          ] + list(params.keys())

                result_df = pd.DataFrame([result], columns=columns)

                if os.path.exists(LOG_FILE):

                    result_df.to_csv(LOG_FILE, index=False, sep=';', mode='a', header=False)

                else:

                    result_df.to_csv(LOG_FILE, index=False, sep=';')



                if os.path.exists(FIMP_FILE):

                    feature_importances.to_csv(FIMP_FILE, index=False, mode='a', header=False)

                else:

                    feature_importances.to_csv(FIMP_FILE, index=False)
end_time = dt.datetime.now()

print('Finished', end_time, (end_time - start_time).seconds, 's')