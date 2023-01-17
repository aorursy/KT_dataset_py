import numpy as np

import pandas as ps

import lightgbm as lgb

from sklearn.linear_model import Ridge, Lasso, LinearRegression

from sklearn.model_selection import TimeSeriesSplit

from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler
def to_time(num: int) -> str:

    return datetime.utcfromtimestamp(num / 1000).strftime('%Y-%m-%d %H:%M:%S')



def _year(num: int) -> int:

    return int(datetime.utcfromtimestamp(num / 1000).strftime('%Y'))



def _month(num: int) -> int:

    return int(datetime.utcfromtimestamp(num / 1000).strftime('%m'))



def _day(num: int) -> int:

    return int(datetime.utcfromtimestamp(num / 1000).strftime('%d'))



def _hour(num: int) -> int:

    return int(datetime.utcfromtimestamp(num / 1000).strftime('%H'))



def _minutes(num: int) -> int:

    return int(datetime.utcfromtimestamp(num / 1000).strftime('%M'))



def _seconds(num: int) -> int:

    return int(datetime.utcfromtimestamp(num / 1000).strftime('%S'))
def transform_df(path_to_train_df: str, path_to_test_df: str) -> ps.DataFrame:

    train_data = ps.read_csv(path_to_train_df)

    test_data = ps.read_csv(path_to_test_df)

    sensors_list = [f'sensor#{i}' for i in range(12)]

    

    for data in (train_data, test_data):

        data['year'] = data['timestamp'].apply(_year)

        data['month'] = data['timestamp'].apply(_month)

        data['day'] = data['timestamp'].apply(_day)



        data['hour'] = data['timestamp'].apply(_hour)

        data['minutes'] = data['timestamp'].apply(_minutes)

        data['seconds'] = data['timestamp'].apply(_seconds)

        

#         data['timestamp'] = (data['timestamp'] / 1000).astype(int)

        data['mean_sensors'] = data[sensors_list].mean(axis=1)

        data['std_sensors'] = data[sensors_list].std(axis=1)

        data.drop(columns=['timestamp'], inplace=True)

    

    return train_data, test_data



train_df, test_df = transform_df('../input/train.csv', '../input/test.csv')

print(train_df.shape, test_df.shape)
train_df.head()
test_df.head()
X = train_df.drop(columns=['oil_extraction', 'Id'])

y = train_df['oil_extraction'].values

split_method = TimeSeriesSplit(n_splits=5)
X_test = test_df.drop(columns=['Id'])

print(X_test.shape)
def gen_oof_preds(model):

    oof_preds = np.zeros(len(X_test))

    mses = []

    maes = []



    for train_idx, val_idx in split_method.split(X, y):

        X_train, y_train = X.values[train_idx], y[train_idx]

        X_val, y_val = X.values[val_idx], y[val_idx]



        fold_model = model().fit(X_train, y_train)



        fold_preds = fold_model.predict(X_val)

        fold_mse = mean_squared_error(y_val, fold_preds)

        print(f'MSE - {fold_mse}', flush=True)

        mses.append(fold_mse)



        fold_mae = mean_absolute_error(y_val, fold_preds)

        print(f'MAE - {fold_mae}\n', flush=True)

        maes.append(fold_mae)



        oof_preds += fold_model.predict(X_test.values) / split_method.n_splits



    print('MSE:', np.mean(mses), '+-', np.std(mses))

    print('MAE:', np.mean(maes), '+-', np.std(maes))

    return oof_preds
linreg_preds = gen_oof_preds(LinearRegression)
lasso_preds = gen_oof_preds(Lasso)
ridge_preds = gen_oof_preds(Ridge)
submission = ps.DataFrame({

    'Id': test_df['Id'].values,

    'Expected': (lasso_preds + linreg_preds + ridge_preds) / 3

})

submission.to_csv('submission.csv', index=False)

submission.head()