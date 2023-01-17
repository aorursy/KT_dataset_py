import numpy as np

import pandas as pd

from pandas.core.nanops import nanmean



import matplotlib.pyplot as plt



#import statsmodels.api as sm

from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings('ignore')



from datetime import datetime as Date, timedelta
sample = pd.read_csv('/kaggle/input/sputnik/sub.csv')

sample
coord = pd.read_csv('/kaggle/input/sputnik/train.csv')

coord.epoch = pd.to_datetime(coord.epoch,format='%Y-%m-%dT%H:%M:%S')

coord.index = coord.epoch

coord.drop('epoch', axis = 1, inplace = True)



coord['year'] = coord.index.year

coord['month'] = coord.index.month

coord['hour'] = coord.index.hour

coord['dayofweek'] = coord.index.dayofweek



coord
from copy import deepcopy as dcpy

cpy = dcpy(coord)

by_sat = [] #отдельно по спутникам



L = np.unique(coord.sat_id).size

assert L == 600



for i in range(L):

    sat = cpy[cpy.sat_id == i]

    cpy = cpy[cpy.sat_id != i]



    L = len(sat.index)

    

    #удаляем лишние измерения, сделанные с интервалом в несколько секунд

    for j in range(L - 1, 0, -1):

        cur, prev = sat.index[j], sat.index[j - 1] 

        if cur - prev < timedelta(0, 10):

            sat = sat[sat.index != cur]

    

    sat.drop('sat_id', axis = 1, inplace = True)

    

    by_sat.append(sat)

    

by_sat[0]
sat = by_sat[0]

ind = sat.type == 'train'
sat[ind].x.plot(figsize = (15, 6), title = 'x = x(t) (train)')
sat[ind].x_sim.plot(figsize = (15, 6), title = 'xs = xs(t) (train)')
sat[ind].y.plot(figsize = (15, 6), title = 'y = y(t) (train)')
sat[ind].y_sim.plot(figsize = (15, 6), title = 'ys = ys(t) (train)')
sat[ind].z.plot(figsize = (15, 6), title = 'z = z(t) (train)')
sat[ind].z_sim.plot(figsize = (15, 6), title = 'zs = zs(t) (train)')
ind = sat.type == 'test'
sat[ind].x_sim.plot(figsize = (15, 6), title = 'xs = xs(t) (test)')
sat[ind].y_sim.plot(figsize = (15, 6), title = 'ys = ys(t) (test)')
sat[ind].z_sim.plot(figsize = (15, 6), title = 'zs = zs(t) (test)')
def find_season(series):

    val = series.values

    L = len(val)

    prev = None

    periods = np.array([])

    

    for i in range(1, L - 1):

        if val[i] > max(val[i - 1], val[i + 1]):

            if prev is None:

                prev = i

            else:

                periods = np.append(periods, i - prev)

                prev = i

                

    return np.mean(periods).astype(int)
coord.index = coord.id

coord.drop('id', axis = 1, inplace = True)

coord
def predict_coord(satel, coord, season):

    to_drop = ['x', 'y' ,'z', 'id']

    sim = '{}_sim'.format(coord)

    

    to_drop.remove(coord)

    

    sat = satel.drop(to_drop, axis = 1)

    features = []

    for i in range(1,100):

        sat["lag{}".format(i)] = sat[sim].shift(i * season)

        features.append("lag{}".format(i))



    sat['lag_mean'] = sat[features].mean(axis = 1)



    train = sat[sat.type == 'train'].drop(['type'], axis = 1).fillna(method = 'ffill')

    train = train.fillna(method = 'bfill')

    train = train.fillna(0)

        

    test = sat[sat.type == 'test'].drop([coord, 'type'], axis = 1).fillna(method = 'ffill')

    test = test.fillna(method = 'bfill')

    test = test.fillna(0)

    

    model = LinearRegression()

    model.fit(train.drop(coord, axis = 1), train[coord])

    

    return model.predict(test)
cnt = 0

for sat in by_sat:

    xseason = find_season(sat.x_sim)

    yseason = find_season(sat.y_sim)

    zseason = find_season(sat.z_sim)



    ind = sat[sat.type == 'test'].id

    

    coord.x.loc[ind] = predict_coord(sat, 'x', xseason)

    coord.y.loc[ind] = predict_coord(sat, 'y', yseason)

    coord.z.loc[ind] = predict_coord(sat, 'z', zseason)

    

    print('Satellite #{} coordinates computed'.format(cnt)) 

    cnt += 1
coord.fillna(method = 'ffill', inplace = True) #заполняем все NaN значениями с предыдущих строк (чуть более ранними измерениями)
coord['error'] = np.linalg.norm(coord[['x', 'y', 'z']].values - coord[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
res = pd.DataFrame( { 'id' : coord[coord.type == 'test'].index, 'error' : coord[coord.type == 'test'].error.values })

res
res.to_csv('/kaggle/working/error.csv', index = False)