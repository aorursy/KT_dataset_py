import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
df = pd.read_csv('../input/wine-whitered/Wine_red.csv',sep=';')

print(df.shape)

df.head(3)
df['rating'] = pd.qcut(df.quality,2,labels=['bad','good'])

df.head(3)
(df_train,df_test) = train_test_split(df,train_size=0.8,test_size=0.2,shuffle=True,stratify=df.rating,random_state=0)
df_train.rating.value_counts()
round(1-1105/(1105+174),3)
df_test.rating.value_counts()
round(1-277/(277+43),3)
#select features and target

features_train = df_train.iloc[:,0:-2]

features_test = df_test.iloc[:,0:-2]

targets_train = df_train.iloc[:,-1]

targets_test = df_test.iloc[:,-1]
# standardization

stnd = StandardScaler()

stnd.fit(features_train)

features_train = stnd.transform(features_train)

features_test = stnd.transform(features_test)
features_test.std(axis=0)
features_train.std(axis=0)
knn = KNeighborsClassifier()
K = np.arange(100)+1

grid = {'n_neighbors':K}
knnCV = GridSearchCV(knn,

                     cv=5,

                     param_grid=grid,

                     return_train_score=True

                     ,n_jobs=-1)
knnCV.fit(features_train,targets_train)
knnCV.best_params_
1-knnCV.best_score_.round(3)
# plot results

results = pd.DataFrame()

results['neighbors'] = K

results['train error'] = 1 -knnCV.cv_results_['mean_train_score']

results['valid error'] = 1 -knnCV.cv_results_['mean_test_score']

axl = results.plot.line(x='neighbors',y='train error')

results.plot.line(x='neighbors',y='valid error',ax=axl)
results.plot.line(x='neighbors',y='valid error')
#compute test result



error_test = 1- knnCV.score(features_test,targets_test)

error_test.round(3)