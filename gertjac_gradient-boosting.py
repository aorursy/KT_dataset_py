import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor



hashit = lambda v: abs(hash(v)) % 10000



df_train = pd.read_csv('../input/train.csv', parse_dates=['REPORTED_DATE'])

df_test = pd.read_csv('../input/test.csv', parse_dates=['REPORTED_DATE'], index_col=0)



for dfi in [df_train, df_test]:

    dfi['weekday'] = dfi.REPORTED_DATE.dt.weekday

    dfi['buurt'] = list(map(hashit, dfi.NEIGHBORHOOD_ID))

    dfi['delict'] = list(map(hashit, dfi.OFFENSE_CATEGORY_ID))



model = GradientBoostingRegressor()

model.fit(df_train[['weekday', 'buurt', 'delict']], df_train['N'])

df_test['gb_pred'] = model.predict(df_test[['weekday', 'buurt', 'delict']])



df_test[['gb_pred']].rename({'gb_pred':'predicted'}, axis=1).to_csv('gradient_boosting.csv')