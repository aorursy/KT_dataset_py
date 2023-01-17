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
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans

from surprise.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import NormalPredictor, Reader, Dataset, accuracy, SVD, SVDpp, KNNBasic, CoClustering, SlopeOne
from surprise.model_selection import cross_validate, KFold, GridSearchCV, train_test_split
from surprise import NormalPredictor, Reader, Dataset, accuracy, SVD, SVDpp, KNNBasic, CoClustering, SlopeOne
from surprise.model_selection import cross_validate, KFold, GridSearchCV, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import NormalPredictor, Reader, Dataset, accuracy, SVD, KNNBasic, CoClustering
from surprise.model_selection import cross_validate, KFold, GridSearchCV, train_test_split
print('done')
d=pd.read_csv('/kaggle/input/the-movies-dataset/ratings_small.csv')
del d['timestamp']
d.head()

def surprise_df(data):
    
    scale = (1, 5)
    reader = Reader(rating_scale=scale)

    df = Dataset.load_from_df(data[['movieId',
                                    "userId",
                                    'rating']], reader)
    
    return df
from sklearn.model_selection import train_test_split
train, test = train_test_split(d, test_size=0.2,random_state=19)

train_data = surprise_df(train)
test_data=surprise_df(test)





kf = KFold(n_splits=5, shuffle=True, random_state=19)
def model_framework(train_data):
    #train_set, test_set = train_test_split(train_data, test_size=0.2, random_state=19)
    #store the rmse values for each fold in the k-fold loop 
    svd_rmse, knn_rmse, co_rmse = [],[],[]

    for trainset, testset in kf.split(train_data):
        
        #svd
        svd = SVD(n_factors=30, n_epochs=50,biased=True, lr_all=0.005, reg_all=0.4, verbose=False)
        svd.fit(trainset)
        svd_pred = svd.test(testset)
        svd_rmse.append(accuracy.rmse(svd_pred,verbose=False))
        
        #knn
        knn = KNNBasic(k=40,sim_options={'name': 'cosine', 'user_based': False}, verbose=False) 
        knn.fit(trainset)
        knn_pred = knn.test(testset)
        knn_rmse.append(accuracy.rmse(knn_pred,verbose=False))
        
        #co_clustering
        co = CoClustering(n_cltr_u=2,n_cltr_i=4,n_epochs=20)         
        co.fit(trainset)
        co_pred = co.test(testset)
        co_rmse.append(accuracy.rmse(co_pred,verbose=False))
        
    
    mean_rmses = [np.mean(svd_rmse),
                  np.mean(knn_rmse),
                  np.mean(co_rmse)]
    
    model_names = ['svd','knn','coclustering']
    compare_df = pd.DataFrame(mean_rmses, columns=['RMSE'], index=model_names)
    
    return compare_df
comparison_df = model_framework(train_data)
comparison_df.head()
def gridsearch(data, model, param_grid):
    param_grid = param_grid
    gs = GridSearchCV(model, param_grid, cv=5)
    gs.fit(data)
    
    new_params = gs.best_params['rmse']
    best_score = gs.best_score['rmse']
    
    print("Best score:", best_score)
    print("Best params:", new_params)
    
    return new_params, best_score
co_param_grid = {'n_cltr_u': [2,3,4,5,6 ],
                  'n_epochs': [20],       
                  'n_cltr_i': [2,3,4,5,6]}

co_params, co_score = gridsearch(train_data, CoClustering, co_param_grid)
co=CoClustering(n_cltr_u=2,n_cltr_i=4,n_epochs=20)
cross_validate(co, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)
print('done')
array_rmse=np.zeros([6,6])
train_set, test_set = train_test_split(data, test_size=0.2, random_state=19)
for u in range(1,7):
    for i in range(1,7):
        co=CoClustering(n_cltr_u=u,n_cltr_i=i,n_epochs=20)        
        co.fit(train_set)
        co_pred = co.test(test_set)
        array_rmse[(u-1),(i-1)]=(accuracy.rmse(co_pred,verbose=False))
        print(u, i,accuracy.rmse(co_pred,verbose=False))
df = pd.DataFrame(data=array_rmse, index=["User Cluster=1","User Cluster=2", "User Cluster=3", "User Cluster=4","User Cluster=5","User Cluster=6",], columns=["Item Cluster=1","Item Cluster=2","Item Cluster=3","Item Cluster=4","Item Cluster=5","Item Cluster=6",])
display(df)