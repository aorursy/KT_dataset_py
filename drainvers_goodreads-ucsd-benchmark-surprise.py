import os



def list_all_files_in(dirpath):

    for dirname, _, filenames in os.walk(dirpath):

        for filename in filenames:

            print(os.path.join(dirname, filename))



list_all_files_in('../input')
import pandas as pd

import numpy as np

import surprise as sp

import time, joblib

from IPython.display import display

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import mean_absolute_error as mae

import warnings; warnings.simplefilter('ignore')
train_interactions_df = pd.read_csv('../input/train_Interactions.csv')

train_interactions_df['rating'] = train_interactions_df['rating'].astype(float)



# Prevent errors with NMF

train_interactions_df['rating'] = train_interactions_df['rating'].where(train_interactions_df['rating'] != 0, 1e-9)



train_df, valid_df = train_interactions_df.loc[:190000-1], train_interactions_df.iloc[190000:]

print('Training set columns  :', train_df.shape)

print('Validation set columns:', valid_df.shape)



x_val, y_val = valid_df.drop('rating', axis=1), valid_df['rating'].values.reshape(-1)
reader = sp.Reader(rating_scale=(0, 5))

dataset = sp.Dataset.load_from_df(train_df, reader)

trainset = dataset.build_full_trainset()

full_dataset = sp.Dataset.load_from_df(train_interactions_df, reader)

full_trainset = full_dataset.build_full_trainset()
models = {

    'SVD': {'clf': sp.SVD(random_state=0)},

    'NMF': {'clf': sp.NMF(random_state=0)},

    'CoClustering': {'clf': sp.CoClustering(random_state=0)},

    'SlopeOne': {'clf': sp.SlopeOne()},

    'SVDpp': {'clf': sp.SVDpp(random_state=0)},

    'KNNBasic': {'clf': sp.KNNBasic()},

    'KNNWithMeans': {'clf': sp.KNNWithMeans()},

    'KNNWithZScore': {'clf': sp.KNNWithZScore()},

    'KNNBaseline': {'clf': sp.KNNBaseline()},

    'NormalPredictor': {'clf': sp.NormalPredictor()},

    'BaselineOnly': {'clf': sp.BaselineOnly()}

}
metrics_list = []



def predict(row):

    return bench_model.predict(row['userID'], row['bookID']).est
for name, model in models.items():

    print('Fitting', name, 'model...', end='')

    

    bench_model = model['clf']

    time_start = time.time()

    bench_model.fit(trainset)

    model['fit_time'] = time.time() - time_start



    print(' completed in', model['fit_time'], 'seconds.')



    print('Predicting ratings using', name, 'model...', end='')

    

    time_start = time.time()

    model['predictions'] = x_val.apply(predict, axis=1)

    model['predict_time'] = time.time() - time_start



    print(' completed in', model['predict_time'], 'seconds.')



    pred_mse = mse(y_val, model['predictions'])

    pred_mae = mae(y_val, model['predictions'])



    metrics_list.append({

        'Model': name,

        'Fit time (seconds)': model['fit_time'],

        'Predict time (seconds)': model['predict_time'],

        'MAE': pred_mae,

        'MSE': pred_mse,

        'RMSE': np.sqrt(pred_mse)

    })
metrics_df = pd.DataFrame(metrics_list).set_index('Model')

display(metrics_df)
for name, model in models.items():

    sub_pred = pd.read_csv('../input/pairs_Rating.txt')

    new_df = pd.DataFrame(columns=['userID', 'bookID'])

    split_items = sub_pred['userID-bookID'].str.split('-', n=1, expand=True)

    new_df['userID'], new_df['bookID'] = split_items[0], split_items[1]



    print('Fitting', name, 'model to entire trainset...', end='')

    

    bench_model = model['clf']

    time_start = time.time()

    bench_model.fit(full_trainset)

    fit_time = time.time() - time_start



    print(' completed in', fit_time, 'seconds.')



    print('Predicting ratings using', name, 'for testset...', end='')

    

    time_start = time.time()

    sub_pred['prediction'] = new_df[['userID', 'bookID']].apply(predict, axis=1)

    predict_time = time.time() - time_start

    

    print(' completed in', predict_time, 'seconds.')

    

    filename = 'submission_{}.csv'.format(name)

    sub_pred.to_csv(filename, index=False)

    

    print('Saved predictions to', filename)