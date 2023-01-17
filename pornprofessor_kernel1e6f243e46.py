import numpy as np

import pandas as pd 

from tqdm.notebook import tqdm

import os

from matplotlib import pyplot as plt

import datetime

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from pandas.core.nanops import nanmean as pd_nanmean



from sklearn.metrics import mean_absolute_error
os.listdir()
data = pd.read_csv('../input/sputnik/train.csv',  parse_dates=['epoch'])
def error(df):

    return np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
sat_id = 0

def get_sputnik(sat_id):

    st_0_train = data[data.sat_id==sat_id][data.type=='train']

    st_0_test= data[data.sat_id==sat_id][data.type=='test']

    return st_0_train, st_0_test
data['error'] = error(data)
data
st_0_train ,st_0_test = get_sputnik(sat_id)

#plt.plot(st)

st_0_train ,st_0_test = get_sputnik(sat_id)

len(st_0_train)

# warnings.filterwarnings('ignore')

# out2 = pd.DataFrame(columns=['id','error'])

# for гi in tqdm(data.sat_id.unique()):

#     train, test = get_sputnik(i)

#     fit1 = ExponentialSmoothing(np.asarray(error(train)),seasonal_periods=24,trend='additive',seasonal='additive',).fit()

#     forecast = pd.Series(fit1.forecast(len(test)))

#     tmp = {'id' : test.id.values, 'error':forecast}

#     tmp2=pd.DataFrame(tmp)

#     out2 = out.append(tmp2)

    
%%time

#linear



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

out2 = pd.DataFrame(columns=['id','error'])

for sat_id in tqdm(data.sat_id.unique()):

    sat = data[data.sat_id==sat_id]

    sat_tmp = sat[['epoch','error','type']].rename(columns={'error': 'target'})

    

    target = sat[['id','error']][sat.type=='test']

    #sat[''] = error(sat)



    features = []

    for i in range(int(len(sat_tmp)*0.1)):

        sat_tmp['lag_{}'.format(i)]= sat_tmp.target.shift(int(len(sat_tmp[sat_tmp.type == 'test'])) + i)

        features.append('lag_{}'.format(i))

            

    model = LinearRegression(n_jobs=7)

    train_df = sat_tmp[sat_tmp.type == 'train'][features + ['target']].dropna()

    test_df = sat_tmp[sat_tmp.type == 'test'][features]

    model.fit(train_df[features], train_df['target'])

    forecast = model.predict(test_df)

    target['error'] = forecast

    tmp = {'id' : target.id.values, 'error':target.error}

    tmp2=pd.DataFrame(tmp)

    out2 = out2.append(tmp2)
len(out2.id.unique())
%%time

#linear

out = pd.DataFrame(columns=['id','error'])

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

out = pd.DataFrame(columns=['id','error'])

for sat_id in tqdm(data.sat_id.unique()):

    sat = data[data.sat_id==sat_id]

    sat_x = sat[['epoch','x','type']].rename(columns={'x': 'target'})

    sat_y = sat[['epoch','y','type']].rename(columns={'y': 'target'})

    sat_z = sat[['epoch','z','type']].rename(columns={'z': 'target'})

    target = sat[['id','x','y','z']][sat.type=='test']

    #sat[''] = error(sat)

    for sat_tmp, coor in [(sat_x,'x'), (sat_y,'y'), (sat_z,'z')]:

        features = []

        for i in range(int(len(sat_tmp)*0.1)):

            sat_tmp['lag_{}'.format(i)]= sat_tmp.target.shift(int(len(sat_tmp[sat_tmp.type == 'test'])) + i)

            features.append('lag_{}'.format(i))

            

        model = LinearRegression(n_jobs=7)

        train_df = sat_tmp[sat_tmp.type == 'train'][features + ['target']].dropna()

        test_df = sat_tmp[sat_tmp.type == 'test'][features]

        model.fit(train_df[features], train_df['target'])

        forecast = model.predict(test_df)

        target[coor] = forecast

    target['error'] = np.linalg.norm(target[['x', 'y', 'z']].values - sat[['x_sim', 'y_sim', 'z_sim']][sat.type == 'test'].values, axis=1) 

    tmp = {'id' : target.id.values, 'error':target.error}

    tmp2=pd.DataFrame(tmp)

    out = out.append(tmp2)
print(len(out.id.unique()),len(out2.id.unique()))
out['error']=(out['error']+out2['error'])/2.
out.to_csv('sub.csv',index=False)
plt.plot(sat.epoch, sat.error)

plt.plot(sat[sat.type=='test'].epoch,target.error)


train = st_0_train[:-100]

test = st_0_train[-100:]

fit1 = ExponentialSmoothing(np.asarray(error(train)) ,seasonal_periods=24 ,trend='add', seasonal='add',).fit()

forecast = pd.Series(fit1.forecast(len(test)))

forecast.index = test.index
test.index




plt.plot(forecast)
from matplotlib.pyplot import figure

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

ax1=plt.subplot(311)

ax1.plot(st_0_train.epoch, st_0_train.x)

ax1.plot(st_0_test.epoch, st_0_test.x_sim)

ax2=plt.subplot(312)

ax2.plot(st_0_train.epoch, st_0_train.x_sim)

ax2.plot(st_0_test.epoch, st_0_test.x_sim)

ax3=plt.subplot(313)

ax3.plot(st_0_train.epoch, error(st_0_train))
rcParams['figure.figsize'] = 12, 7

result = sm.tsa.seasonal_decompose(error(st_0_train), model='additive',period=24)

result.plot()

plt.show()