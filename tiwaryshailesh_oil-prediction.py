import pandas as pd

import numpy as np

%matplotlib inline



from sklearn.metrics import mean_squared_log_error

from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
data = pd.read_csv("../input/sachitdataset/data.csv")

data.head()
melt = data.melt(id_vars='MD', var_name='SO', value_name='OilVol')

melt['MD'] = melt['MD']

melt['SO'] = melt['SO'].str.extract('(\d+)', expand=False).astype(int)

melt = melt.sort_values(['SO', 'MD'])

melt.head()
melt2 = melt.copy()

melt2['Last-Vol'] = melt2.groupby(['MD'])['OilVol'].shift()

melt2['Last_Oil_Diff'] = melt2.groupby(['MD'])['Last-Vol'].diff()

melt2 = melt2.dropna()

melt2
def rmsle(ytrue, ypred):

    return np.sqrt(mean_squared_log_error(ytrue, ypred))



mean_error = []

for week in range(7,11):

    train = melt2[melt2['SO'] < week]

    val = melt2[melt2['SO'] == week]

    

    p = val['Last-Vol'].values

    

    error = rmsle(val['OilVol'].values, p)

    print('SO %d - Error %.5f' % (week, error))

    mean_error.append(error)

print('Mean Error = %.5f' % np.mean(mean_error))
melt4 = melt.copy()

melt4['Last-Vol'] = melt4.groupby(['MD'])['OilVol'].shift()

melt4['Last_Oil_Diff'] = melt4.groupby(['MD'])['Last-Vol'].diff()

melt4['Last-1_Vol'] = melt4.groupby(['MD'])['OilVol'].shift(2)

melt4['Last-1_Oil_Diff'] = melt4.groupby(['MD'])['Last-1_Vol'].diff()

melt4['Last-2_Vol'] = melt4.groupby(['MD'])['OilVol'].shift(3)

melt4['Last-2_Oil_Diff'] = melt4.groupby(['MD'])['Last-2_Vol'].diff()

melt4 = melt4.dropna()

melt4.head()
mean_error = []

for week in range(7,11):

    train = melt4[melt4['SO'] < week]

    val = melt4[melt4['SO'] == week]

    

    xtr, xts = train.drop(['OilVol'], axis=1), val.drop(['OilVol'], axis=1)

    ytr, yts = train['OilVol'].values, val['OilVol'].values

    

    mdl = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)

    mdl.fit(xtr, ytr)

    

    p = mdl.predict(xts)

    

    error = rmsle(yts, p)

    print('Week %d - Error %.5f' % (week, error))

    mean_error.append(error)

print('Mean Error = %.5f' % np.mean(mean_error))
val.loc[:, 'Prediction'] = np.round(p)

val.plot.scatter(x='Prediction', y='OilVol', figsize=(15,10), title='Prediction vs Sales', 

                 ylim=(0,40), xlim=(0,40))