import os

import numpy as np 

import pandas as pd

from sklearn.preprocessing import OneHotEncoder as ohe

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

import time

import joblib



from fastFM.mcmc import FMRegression as FMRmcmc

from fastFM.als import FMRegression as FMRals

from fastFM.sgd import FMRegression as FMRsgd
train = pd.read_csv('../input/copy-of-predict-movie-ratings/train.csv')

train.head(10)
test = pd.read_csv('../input/copy-of-predict-movie-ratings/test.csv')

test.head(10)
x_train = train.drop(['rating','timestamp','ID'],axis=1)

y_train = train[['rating']].values.reshape(-1)

x_test = test.drop(['timestamp','ID'],axis=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
encoder = ohe(handle_unknown='ignore').fit(x_train)

x_train_e = encoder.transform(x_train)

x_val_e = encoder.transform(x_val)

x_test_e = encoder.transform(x_test)
with open('ohe_encoder.sav',mode='wb') as model_f:

        joblib.dump(encoder, model_f)
models = {

    'MCMC' : {

        'clf' : FMRmcmc(n_iter=120, init_stdev=0.2, rank=14)

    },

    'ALS' : {

        'clf' : FMRals(n_iter=120, init_stdev=0.2, rank=14)

    },

    'SGD' : {

        'clf' : FMRsgd(n_iter=26000000, init_stdev=0.2, l2_reg_V=0.1, l2_reg_w=0.001, rank=12, step_size=0.005)

    }

}
for name, items in models.items():

    t_start = time.time()

    if name == 'MCMC':

        print('Skipping MCMC Algorithm on training')

        continue

    print('Using', name, 'Algorithm')

    items['clf'].fit(x_train_e, y_train)

    items['fit_duration'] = time.time() - t_start

    print('Training duration : ', items['fit_duration'])
for name, items in models.items():

    t_start = time.time()

    print('Using', name, 'Algorithm')

    if name == 'MCMC':

        items['prediction'] = items['clf'].fit_predict(x_train_e, y_train, x_val_e)

        items['fit_duration'] = items['predict_duration'] = time.time() - t_start

    else:

        items['prediction'] = items['clf'].predict(x_val_e)

        items['predict_duration'] = time.time() - t_start

    print('Predict duration : ',items['predict_duration'])
for name, items in models.items():

    print('fastFM',name)

    items['mse'] = mse(y_val, items['prediction'])

    items['rmse'] = np.sqrt(items['mse'])

    print('Fit Time     :',items['fit_duration'],'sec')

    print('Predict Time :',items['predict_duration'],'sec')

    print('MSE          :',items['mse'])

    print('RMSE         :',items['rmse'],'\n')
for name, items in models.items():

    t_start = time.time()

    print('Using', name, 'Algorithm')

    if name == 'MCMC':

        items['sub_prediction'] = items['clf'].fit_predict(x_train_e, y_train, x_test_e)

        items['sub_fit_duration'] = items['sub_predict_duration'] = time.time() - t_start

    else:

        items['sub_prediction'] = items['clf'].predict(x_test_e)

        items['sub_predict_duration'] = time.time() - t_start

    print('Predict duration : ',items['sub_predict_duration'])

    items['submission'] = test.drop(['user','movie','timestamp'],axis=1)

    items['submission']['rating'] = items['sub_prediction']

    items['submission'].to_csv('sub_pmr_fastfm_'+name+'.csv', index=False)

    print('Submission has been saved.')
for name in models:

    model_file = "model_"+name+".sav"

    with open(model_file,mode='wb') as model_f:

        joblib.dump(models[name]['clf'], model_f)
with open('all_models.sav',mode='wb') as model_f:

        joblib.dump(models, model_f)