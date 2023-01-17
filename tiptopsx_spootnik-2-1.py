import datetime

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

from tqdm.notebook import tqdm

import warnings





def smape(predict, test):

    return np.mean(2 * np.abs(test - predict) / (np.abs(test) + np.abs(predict))) * 100





# Удаляем лишние измерения, когда моделируем движение спутника

# Процесс поиска косяков в датасете не вошел в этот ноутбук

def get_sat(sat_id):

    sat = df.loc[df['sat_id'] == sat_id].copy()

    sat.drop(sat[sat['timedelta'] == 1].index, inplace=True)

    return sat





# Определяем период по экстремумам

def simple_period(sat_id):

    def get_loc_extr(arr):

        loc_max_ind_ = np.where((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]))

        loc_min_ind_ = np.where((arr[1:-1] < arr[:-2]) & (arr[1:-1] < arr[2:]))

        return loc_max_ind_, loc_min_ind_



    sat = get_sat(sat_id)

    train = sat.loc[sat['type'] == 'train']

    x_loc_max, x_loc_min = get_loc_extr(np.array(train['x']))

    y_loc_max, y_loc_min = get_loc_extr(np.array(train['y']))

    z_loc_max, z_loc_min = get_loc_extr(np.array(train['z']))

    return int(np.concatenate([

        x_loc_max[0][1:] - x_loc_max[0][:-1], x_loc_min[0][1:] - x_loc_min[0][:-1],

        y_loc_max[0][1:] - y_loc_max[0][:-1], y_loc_min[0][1:] - y_loc_min[0][:-1],

        z_loc_max[0][1:] - z_loc_max[0][:-1], z_loc_min[0][1:] - z_loc_min[0][:-1],

    ]).mean())





plt.rcParams['figure.figsize'] = [16, 8]

df = pd.read_csv('/kaggle/input/sputnik/train.csv')

df['datetime'] = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f"), df['epoch']))

df['timedelta'] = np.zeros(len(df.index))

df['timedelta'].values[1:] = (df['datetime'].values[1:] - df['datetime'].values[:-1]) / np.timedelta64(1, 'ms')

df['error'] = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)





# Регрессия из лекции, работает быстро и неплохо (SMAPE ~11)

def regressed(train, test, period, target):

    sat_df = pd.DataFrame({target: pd.concat((train, test), axis=0)[target].to_numpy()})

    features = []

    for period_mult in range(int(np.ceil(len(test) / period)), min(len(test), int(len(train) / period))):

        sat_df["lag_period_{}".format(period_mult)] = sat_df[target].shift(period_mult * period)

        features.append("lag_period_{}".format(period_mult))

    sat_df['lagf_mean'] = sat_df[features].mean(axis=1)

    features.extend(['lagf_mean'])

    train_df = sat_df[:-len(test)].dropna()

    test_df = sat_df[-len(test):][features]

    lin_reg = LinearRegression()

    lin_reg.fit(train_df.drop(target, axis=1), train_df[target])

    result = lin_reg.predict(test_df)

    return result





# Доработки напильником: отдельная регрессия для каждого периода

def regressed(train, test, period, target):

    result = []

    sat_df = pd.DataFrame({target: pd.concat((train, test), axis=0)[target].to_numpy()})

    features = []

    for period_mult in range(1, int(np.ceil(len(train) / period)) - 1):

        sat_df["lag_period_{}".format(period_mult)] = sat_df[target].shift(period_mult * period)

        features.append("lag_period_{}".format(period_mult))

    features.extend(['lagf_mean'])

    for i in range(int(np.ceil(len(test) / period))):

        sat_df['lagf_mean'] = sat_df[features[i:-1]].mean(axis=1)

        train_df = sat_df[:-len(test)][[target] + features[i:]].dropna().copy()

        train_df['lagf_mean'] = train_df[features[i:]].mean(axis=1)

        if -len(test) + (i + 1) * period >= 0:

            test_df = sat_df[-len(test) + i * period:][features[i:]]

        else:

            test_df = sat_df[-len(test) + i * period:-len(test) + (i + 1) * period][features[i:]]

        lin_reg = LinearRegression()

        lin_reg.fit(train_df.drop(target, axis=1), train_df[target])

        result.extend(lin_reg.predict(test_df))

    result.extend([0] * (len(test) - len(result)))

    return result





warnings.filterwarnings('ignore')

for sat_id in tqdm(np.unique(df['sat_id'])):

    sat = get_sat(sat_id)

    train_ = sat.loc[sat['type'] == 'train']

    test_ = sat.loc[sat['type'] == 'test']

    period_ = simple_period(sat_id)



    for target_ in ['x', 'y', 'z']:

        sat.loc[sat['type'] == 'test', target_] = regressed(train_, test_, period_, target_)



    pred = sat.loc[sat['type'] == 'test']

    pred['error'] = np.linalg.norm(pred[['x', 'y', 'z']].values - pred[['x_sim', 'y_sim', 'z_sim']].values, axis=1)



    for k in ['error', 'x', 'y', 'z']:

        df.loc[(df['sat_id'] == sat_id) & (df['type'] == 'test') & (df['timedelta'] != 1), k] = pred[k]

warnings.filterwarnings('default')

for index, row in df.loc[(df['timedelta'] == 1) & (df['type'] == 'test')].iterrows():

    if np.isnan(df['error'][index]):

        df['error'][index] = np.linalg.norm([df['x'][index - 1] - df['x_sim'][index],

                                             df['y'][index - 1] - df['y_sim'][index],

                                             df['z'][index - 1] - df['z_sim'][index]])

df.loc[df['type'] == 'test'][['id', 'error']].to_csv('submission.csv', index=False)