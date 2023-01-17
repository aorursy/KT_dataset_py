# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/generated-user-data-for-recommendation/file3.csv')
df.head()
df['rating'].value_counts()[:50].plot()
from surprise import Reader

from surprise import SVD

from surprise import Dataset

from surprise.model_selection import cross_validate

from surprise.model_selection import train_test_split

reader = Reader()
data = Dataset.load_from_df(df[['UserId', 'ArticleId_served', 'rating']], reader)
dataset = data.build_full_trainset()

print('Number of users: ',dataset.n_users,'\n')

print('Number of items: ',dataset.n_items)
from surprise import SVD

from surprise import SVDpp

from surprise import NMF

from surprise import NormalPredictor

from surprise import KNNBaseline

from surprise import KNNBasic

from surprise import KNNWithMeans

from surprise import BaselineOnly

from surprise import CoClustering



benchmark = []

# Iterate over all algorithms

for algorithm in [SVD(), SVDpp(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), BaselineOnly(), CoClustering()]:

    # Perform cross validation

    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)

    

    # Get results & append algorithm name

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    benchmark.append(tmp)

    

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')  
from surprise.model_selection import GridSearchCV

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],

              'reg_all': [0.4, 0.6]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)



gs.fit(data)



# best RMSE score

print(gs.best_score['rmse'])



# best RMSE score

print(gs.best_score['mae'])



# combination of parameters that gave the best RMSE score

print(gs.best_params['rmse'])



# combination of parameters that gave the best MAE score

print(gs.best_params['mae'])
svd = SVD(n_factors= 50, reg_all=0.05)

svd.fit(dataset)
svd.predict(2,4)
svd.predict(2,5)
r = df['ArticleId_served'].unique()

len(r)

list_of_articles = r.tolist()

list_of_articles[:5]



rec = []

for i in list_of_articles:

    predicted_rating = svd.predict(2, i)

    rec.append(predicted_rating)

    

rec[:10]    
param_grid = {'bsl_options':{'method': ['als','sgd'],'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],

              'reg_all': [0.4, 0.6]}}



# bsl_options = {'method': ['als','sgd'],'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],

#               'reg_all': [0.4, 0.6]}

bsl_algo = BaselineOnly()
gs = GridSearchCV(BaselineOnly, param_grid, measures=['rmse', 'mae'], cv=3)



gs.fit(data)



# best RMSE score

print(gs.best_score['rmse'])



# combination of parameters that gave the best RMSE score

print(gs.best_params['rmse'])

bsl_options = {'method': 'sgd', 'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.4}

algo = BaselineOnly(bsl_options=bsl_options)



algo.fit(dataset)
algo.predict(2,4)
algo.predict(2,5)
r = df['ArticleId_served'].unique()

len(r)

list_of_articles = r.tolist()

list_of_articles[:5]
rec = []

for i in list_of_articles:

    predicted_rating = algo.predict(2, i)

    rec.append(predicted_rating)
rec[:10]
from surprise import SVD

from surprise import SVDpp

from surprise import SlopeOne

from surprise import NMF

from surprise import NormalPredictor

from surprise import KNNBaseline

from surprise import KNNBasic

from surprise import KNNWithMeans

from surprise import KNNWithZScore

from surprise import BaselineOnly

from surprise import CoClustering



benchmark = []

# Iterate over all algorithms

for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:

    # Perform cross validation

    results = cross_validate(algorithm, data, measures=['RMSE','FCP'], cv=3, verbose=False)

    

    # Get results & append algorithm name

    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    benchmark.append(tmp)

    

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')  

pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_fcp') 