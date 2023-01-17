import warnings

warnings.filterwarnings("ignore")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns



from time import localtime

import time

from datetime import datetime, timedelta,date

import gc



from sklearn.decomposition import PCA
RANDOM_SEED = 631

np.random.seed(RANDOM_SEED)
!ls ../input

train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)



price_raw = train['price']

price_raw_log = np.log1p(price_raw)

train.drop('price', axis = 1, inplace=True)
for i in test.columns:

    tyty = test[i].dtype    

    if not tyty == 'int64' and not tyty == 'float64':

        print(f'feature: {i}, type: {str(tyty)}')
def clean_data(dataset):

    dataset['data_y'] = ''

    dataset['data_m'] = ''

    dataset['data_y'] = dataset['date'].apply(lambda x : str(x[:4])).astype(int)

    dataset['data_m'] = dataset['date'].apply(lambda x : str(x[4:6])).astype(int)

    dataset.drop('date', axis=1, inplace=True)

    return dataset



cleaned_train = clean_data(train)

cleaned_test = clean_data(test)
def geogege(data):

    data['zipcode'] = data['zipcode'].astype(str)  

    data['zipcode-3'] = data['zipcode'].apply(lambda x : str(x[2:3])).astype(int)

    data['zipcode-4'] = data['zipcode'].apply(lambda x : str(x[3:4])).astype(int)

    data['zipcode-5'] = data['zipcode'].apply(lambda x : str(x[4:5])).astype(int)

    data['zipcode-34'] = data['zipcode'].apply(lambda x : str(x[2:4])).astype(int)

    data['zipcode-45'] = data['zipcode'].apply(lambda x : str(x[3:5])).astype(int)

    data['zipcode-35'] = data['zipcode'].apply(lambda x : str(x[2:5])).astype(int)

    data.drop('zipcode', axis=1, inplace=True)

    return data



geoge_train = geogege(cleaned_train)

geoge_test = geogege(cleaned_test)
def latlong_pca(trainset, testset):

    pca2 = PCA(n_components=2)

    coord = trainset[['lat','long']]

    coord_test = testset[['lat','long']]

    

    principalComponents_updated = pca2.fit_transform(coord)

    trainset['coord_pca1']= ''

    trainset['coord_pca2']= ''

    trainset['coord_pca1']= principalComponents_updated[:, 0]

    trainset['coord_pca2']= principalComponents_updated[:, 1]



    principalComponents_updated_test = pca2.transform(coord_test)

    testset['coord_pca1']= ''

    testset['coord_pca2']= ''

    testset['coord_pca1']= principalComponents_updated_test[:, 0]

    testset['coord_pca2']= principalComponents_updated_test[:, 1]

    return trainset, testset



pcaed_train, pcaed_test = latlong_pca(geoge_train, geoge_test)
def all_pca(trainset, testset):

    pca1 = PCA(n_components=2)



    principalComponents_updated = pca1.fit_transform(trainset)

    trainset['pca1']= ''

    trainset['pca2']= ''

    trainset['pca1']= principalComponents_updated[:, 0]

    trainset['pca2']= principalComponents_updated[:, 1]



    principalComponents_updated_test = pca1.transform(testset)

    testset['pca1']= ''

    testset['pca2']= ''

    testset['pca1']= principalComponents_updated_test[:, 0]

    testset['pca2']= principalComponents_updated_test[:, 1]

    return trainset, testset



pcaed_train1, pcaed_test1 = all_pca(pcaed_train, pcaed_test)
x_train = pcaed_train1

y_train = price_raw_log

x_test = pcaed_test1



train_columns = list(x_train.columns)



xgb_params_add1 ={

    'seed': RANDOM_SEED,

    'learning_rate': 0.05,

    'max_depth': 5,

    'subsample': 0.9,

    'colsample_bytree': 0.4,

    'silent': True,

    'n_estimators':5000,

    'refit' : True

}
