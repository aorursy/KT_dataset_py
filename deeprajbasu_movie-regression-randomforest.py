# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

df.head()
df['original_language'] = df['original_language'].apply(lambda x : 1 if x=='en' else 0 )
df['original_language'].unique()
df.rename(columns={"original_language": "English"},inplace=True)
a = df['production_countries'][0]

a
import ast

a = ast.literal_eval(a)

a
df['release_date']=df['release_date'].fillna('1992-09-04')
df['release_date'].isna().sum()
df['release_date']= pd.to_datetime(df['release_date']) 

df['release_date']=df['release_date'].apply(lambda x: int(x.year))

df['release_date'].head()
df
df.drop(['homepage', 'id','keywords','original_title','overview','status','tagline','title','English'], axis=1,inplace=True)
df['production_companies']=df['production_companies'].apply(lambda x: ast.literal_eval(x))



df['production_companies']=df['production_companies'].apply(lambda x: len(x))

df['production_companies'].head()
df['genres']=df['genres'].apply(lambda x: ast.literal_eval(x))



df['genres']=df['genres'].apply(lambda x: len(x))

df['genres'].head()
df['production_countries']=df['production_countries'].apply(lambda x: ast.literal_eval(x))



df['production_countries']=df['production_countries'].apply(lambda x: len(x))

df['production_countries'].head()
df['spoken_languages']=df['spoken_languages'].apply(lambda x: ast.literal_eval(x))

df['spoken_languages']=df['spoken_languages'].apply(lambda x: len(x))

df['spoken_languages'].head()



#df.rename(columns={"spoken_languages": "Number of spoken_languages"},inplace=True)

#df.rename(columns={"production_countries": "Number of countries produced in"},inplace=True)

#df.rename(columns={"production_companies": "Number of producers"},inplace=True)
df['runtime']=df['runtime'].fillna(df['runtime'].mean())
df['popularity']=df['popularity'].apply(lambda x: int(x))

df['runtime']=df['runtime'].apply(lambda x: int(x))

df
df['production_companies']=df['production_companies'].replace(0,1)

df['production_countries']=df['production_countries'].replace(0,1)



quant = 0.0156

df['revenue']=df['revenue'].replace(0,df['revenue'].quantile(quant))

df['budget']=df['budget'].replace(0,df['budget'].quantile(quant))

df['popularity']=df['popularity'].replace(0,df['popularity'].quantile(quant))

df['runtime']=df['runtime'].replace(0,df['runtime'].quantile(quant))



df['spoken_languages']=df['spoken_languages'].replace(0,1)
#df.drop(['runtime'], axis=1,inplace=True)
df.info()
df.columns
X = df.drop(['revenue'],axis=1)

y = df['revenue']
df.describe()
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

X=scaler.fit_transform(X)

#y=scaler.fit_transform(y)


# import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA



# pca = PCA()

# principalComponents = pca.fit_transform(X)



# plt.figure()

# plt.plot(np.cumsum(pca.explained_variance_ratio_))

# plt.xlabel('Number of Components')

# plt.ylabel('Variance (%)') #for each component

# plt.title('Explained Variance')

# plt.show()
#pca = PCA(n_components=5)

#X = pca.fit_transform(X)

from sklearn.ensemble import RandomForestRegressor as forest

clf = forest(max_depth=40,max_features=0.4,n_estimators=45,random_state=42)
from sklearn.model_selection import train_test_split 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
import pickle



filename = 'RandomForest_model.pickle'

pickle.dump(clf, open(filename, 'wb'))



filename_scaler = 'scaler_model.pickle'

pickle.dump(scaler, open(filename_scaler, 'wb'))

##Hyper parameter tuning 



# n_estimators = [int(x) for x in np.linspace(start = 40, stop = 120, num = 10)]

# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 75, num = 10)]

# max_depth.append(None)

# # Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]



# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                }



# from sklearn.model_selection import GridSearchCV

# grid_search = GridSearchCV(estimator=clf,param_grid=random_grid,cv=2,n_jobs =-1,verbose = 3)

# grid_search.fit(X_train, y_train)

# grid_search.best_params_







# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.

# [Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:   49.3s

# [Parallel(n_jobs=-1)]: Done 120 tasks      | elapsed:  6.5min

# [Parallel(n_jobs=-1)]: Done 280 tasks      | elapsed: 15.5min

# [Parallel(n_jobs=-1)]: Done 504 tasks      | elapsed: 26.2min

# [Parallel(n_jobs=-1)]: Done 792 tasks      | elapsed: 33.7min

# /opt/conda/lib/python3.6/site-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.

#   "timeout or by a memory leak.", UserWarning



# [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.

# [Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:   49.3s

# [Parallel(n_jobs=-1)]: Done 120 tasks      | elapsed:  6.5min

# [Parallel(n_jobs=-1)]: Done 280 tasks      | elapsed: 15.5min

# [Parallel(n_jobs=-1)]: Done 504 tasks      | elapsed: 26.2min

# [Parallel(n_jobs=-1)]: Done 792 tasks      | elapsed: 33.7min

# /opt/conda/lib/python3.6/site-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.

#   "timeout or by a memory leak.", UserWarning

# [Parallel(n_jobs=-1)]: Done 1144 tasks      | elapsed: 53.7min
