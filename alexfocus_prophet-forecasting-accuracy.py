import gc

from tqdm import tqdm

from tqdm._tqdm import trange

import numpy as np

import pandas as pd

from pylab import rcParams

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from scipy.stats import probplot

from fbprophet import Prophet



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")
folder = '/kaggle/input/m5-forecasting-accuracy/'

calendar = pd.read_csv(folder+'calendar.csv')

validation = pd.read_csv(folder+'sales_train_validation.csv')

submission = pd.read_csv(folder+'sample_submission.csv')



submission = submission[submission.id.str.find('validation')!=-1]

validation = validation.merge(submission, on='id', how='left')

validation = validation.drop(['item_id','dept_id','cat_id','store_id','state_id'], axis=1)



valid_cols = ['d_'+str(1913+i) for i in range(1,29)]

validation.columns = validation.columns.tolist()[:-28]+valid_cols

validation.columns
submission
validation
item1 = validation.iloc[0]

item1 = item1.drop('id').T.reset_index().merge(calendar[['d','date']], left_on='index', right_on='d', how='left').drop(['index','d'], axis=1)

item1.columns = ['y', 'ds']

item1.y = item1.y.astype('float')

item1.ds = item1.ds.astype('datetime64')



rcParams['figure.figsize'] = 20, 5

plt.plot(item1.ds, item1.y)
item = validation.iloc[0]

item = item.drop('id').T.reset_index().merge(calendar[['d','date']], left_on='index', right_on='d', how='left').drop(['index','d'], axis=1)

item.columns = ['y', 'ds']

item.y = item.y.astype('float')

item.ds = item.ds.astype('datetime64')

train_item = item.iloc[:-28]

valid_item = item.iloc[-28:]



#ph = Prophet()



# PARAMETROS DEL MODELO

gr ='linear' #'linear' or 'logistic' trend

sm ='multiplicative' #'additive' (default) or 'multiplicative'

iw = 0.8

ds = False #'auto' True or False 

ys = True #'auto' True or False

ws = True #'auto' True or False

fom = 20 #fourier order mensual

p = 10 #horizonte de prediccion (periods)

ph = Prophet(growth= gr,

            seasonality_mode= sm,

            interval_width = iw,

            #changepoint_range = cr,

            #changepoint_prior_scale=0.07, 

            daily_seasonality= ds, 

            yearly_seasonality= ys,

            weekly_seasonality= ws).add_seasonality(

                            name='mensual', period=30.5,fourier_order=fom)



ph.fit(train_item)

forecast = ph.predict(item[['ds']])

figure = ph.plot(forecast)

figure.show()
for i in trange(len(validation)):

    item = validation.iloc[i]

    item_id = item.id

    item = item.drop('id').T.reset_index().merge(calendar[['d','date']], left_on='index', right_on='d', how='left').drop(['index','d'], axis=1)

    item.columns = ['y', 'ds']

    item.y = item.y.astype('float')

    item.ds = item.ds.astype('datetime64')

    train_item = item.iloc[:-28]

    valid_item = item.iloc[-28:]



    # ph = Prophet()

    # PARAMETROS DEL MODELO

    gr ='linear' #'linear' or 'logistic' trend

    sm ='multiplicative' #'additive' (default) or 'multiplicative'

    iw = 0.8

    ds = False #'auto' True or False 

    ys = True #'auto' True or False

    ws = True #'auto' True or False

    fom = 20 #fourier order mensual

    p = 10 #horizonte de prediccion (periods)

    ph = Prophet(growth= gr,

                seasonality_mode= sm,

                interval_width = iw,

                #changepoint_range = cr,

                #changepoint_prior_scale=0.07, 

                daily_seasonality= ds, 

                yearly_seasonality= ys,

                weekly_seasonality= ws).add_seasonality(

                                name='mensual', period=30.5,fourier_order=fom)

    ph.fit(train_item)

    forecast = ph.predict(valid_item[['ds']])

    validation.iloc[i, -28:] = forecast.yhat.tolist()

    break # FIXME
submission_prophet = validation[['id']+valid_cols]

submission_prophet.columns = ['id']+['F'+str(i) for i in range(1,29)]

submission_prophet_eval = submission_prophet.copy()

submission_prophet_eval.id = submission_prophet_eval.id.apply(lambda _id:_id.replace('_validation','_evaluation'))

submission_prophet = pd.concat([submission_prophet, submission_prophet_eval])

submission_prophet
submission_prophet.to_csv('submission_prophet.csv', index=False)