import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats.mstats import gmean
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
train.head(3)
better_column_names = {
    'Дата рейса': 'flight_date',
    'Рейс': 'flight_number',
    'А/П отправл': 'airport_from',
    'А/П прибыт': 'airport_to',
    'Номер ВС': 'aircraft_id',
    'Время отправления по расписанию': 'dt_departure',
    'Время прибытия по расписанию': 'dt_arrival'
}
target_column_name = 'Задержка отправления в минутах'
train_y = train.rename(columns={
    target_column_name: 'delay'
})['delay']
train = train[list(better_column_names)].rename(columns=better_column_names)
test = test[list(better_column_names)].rename(columns=better_column_names)
def generate_features(df):
    df['flight_date'] = pd.to_datetime(df['flight_date'])
    df['dt_departure'] = pd.to_datetime(df['dt_departure'])
    df['dt_arrival'] = pd.to_datetime(df['dt_arrival'])
    df['time_slot'] = df['dt_departure'].dt.hour * 60 + df['dt_departure'].dt.minute
    return df

train = generate_features(train)
test = generate_features(test)
def generate_mean_encoding(column, target):
    column_name = column.name
    return pd.concat([column, target],
                     axis='columns').groupby(column_name).mean()[target.name]
def get_features(df, train):
    airport_from_ME = generate_mean_encoding(train['airport_from'], train_y)
    airport_to_ME = generate_mean_encoding(train['airport_to'], train_y)
    aircraft_id_ME = generate_mean_encoding(train['aircraft_id'], train_y)
    flight_number_ME = generate_mean_encoding(train['flight_number'], train_y)
    time_slot_ME = generate_mean_encoding(train['time_slot'], train_y)
    aircraft_id_VC = train['aircraft_id'].value_counts()
    
    return pd.concat([
        df['airport_from'].apply(lambda x: airport_from_ME.get(x, airport_from_ME.median())),
        df['airport_to'].apply(lambda x: airport_to_ME.get(x, airport_to_ME.median())),
        df['aircraft_id'].apply(lambda x: aircraft_id_ME.get(x, aircraft_id_ME.median())),
        df['flight_number'].apply(lambda x: flight_number_ME.get(x, flight_number_ME.median())),
        df['aircraft_id'].apply(lambda x: aircraft_id_VC.get(x, aircraft_id_VC.median())),
        df['time_slot'].apply(lambda x: time_slot_ME.get(x, time_slot_ME.median())),
        df['flight_date'].dt.weekday,
        df['dt_departure'].dt.hour * 60 + df['dt_departure'].dt.minute,
        (df['dt_arrival'] - df['dt_departure']).dt.seconds
    ], axis='columns')
predictions = []

for random_state in tqdm(range(0, 2)):
    folds = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in folds.split(train):
        x_t, _ = train.iloc[train_index], train.iloc[test_index]
        y_t, _ = train_y.iloc[train_index], train_y.iloc[test_index]
        
        train_features = get_features(x_t, train=x_t)
        test_features = get_features(test, train=x_t)
        regressor = ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, max_depth=7,
                                        random_state=random_state)
        regressor.fit(train_features, y_t)
        predictions.append(regressor.predict(test_features))
        
test_y_pred = gmean(np.array(predictions), axis=0)
submission = pd.read_csv('../input/sample_submission.csv')
submission['Задержка отправления в минутах'] = test_y_pred
submission.to_csv('extra_trees_ensemble_submission.csv', index=None)