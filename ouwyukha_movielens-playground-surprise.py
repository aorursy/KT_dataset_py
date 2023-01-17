import os

import numpy as np 

import pandas as pd

from sklearn.preprocessing import OneHotEncoder as ohe

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

import time

import joblib



from surprise import Reader, Dataset

from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
train = pd.read_csv('../input/copy-of-predict-movie-ratings/train.csv')

train.head(10)
test = pd.read_csv('../input/copy-of-predict-movie-ratings/test.csv')

test.head(10)
train_dataset = train.drop(['timestamp','ID'],axis=1)

test_dataset = test.drop(['timestamp','ID'],axis=1)
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=0)
x_val, y_val = val_dataset.drop('rating', axis=1), val_dataset[['rating']].values.reshape(-1)
reader = Reader()

data = Dataset.load_from_df(train_dataset, reader)

trainset = data.build_full_trainset()
models = {

    'SVD' : {

        'clf' : SVD(random_state=0)

    },

    #Time limit

    #'SVDpp' : {

    #    'clf' : SVDpp(random_state=0)

    #},

    'SlopeOne' : {

        'clf' : SlopeOne()

    },

    'NMF' : {

        'clf' : NMF(random_state=0)

    },

    'NormalPredictor' : {

        'clf' : NormalPredictor()

    },

    'BaselineOnly' : {

        'clf' : BaselineOnly()

    },

    'CoClustering' : {

        'clf' : CoClustering(random_state=0)

    }

}
for name, items in models.items():

    t_start = time.time()

    print('Using', name, 'Algorithm')

    items['clf'].fit(trainset)

    items['fit_duration'] = time.time() - t_start

    print('Training duration : ', items['fit_duration'])
def predict(x):

    est = curr_model.predict(x.user,x.movie).est

    return est
for name, items in models.items():

    t_start = time.time()

    print('Using', name, 'Algorithm')

    curr_model = items['clf']

    items['prediction'] = x_val.apply(predict, axis=1)

    items['predict_duration'] = time.time() - t_start

    print('Predict duration : ',items['predict_duration'])
for name, items in models.items():

    print('Surprise',name)

    items['mse'] = mse(y_val, items['prediction'])

    items['rmse'] = np.sqrt(items['mse'])

    print('Fit Time     :',items['fit_duration'],'sec')

    print('Predict Time :',items['predict_duration'],'sec')

    print('MSE          :',items['mse'])

    print('RMSE         :',items['rmse'],'\n')
for name, items in models.items():

    t_start = time.time()

    print('Using', name, 'Algorithm')

    curr_model = items['clf']

    items['sub_prediction'] = test_dataset.apply(predict, axis=1)

    items['sub_predict_duration'] = time.time() - t_start

    print('Predict duration : ',items['sub_predict_duration'])

    items['submission'] = test.drop(['user','movie','timestamp'],axis=1)

    items['submission']['rating'] = items['sub_prediction']

    items['submission'].to_csv('sub_pmr_surprise_'+name+'.csv', index=False)

    print('Submission has been saved.')
for name in models:

    if name == 'SVD':

        model_file = "surprise_model_"+name+".sav"

        with open(model_file,mode='wb') as model_f:

            joblib.dump(models[name]['clf'], model_f)
"""

with open('surprise_all_models.sav',mode='wb') as model_f:

        joblib.dump(models, model_f)

"""