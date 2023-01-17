import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,AdaBoostRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from numpy import sort

from sklearn.metrics import mean_squared_error

import time

from sklearn.model_selection import cross_val_score,KFold

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import VotingRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

df=pd.read_csv("../input/bits-f464-l1/train.csv")
df.head()
df.drop(["id"],axis=1,inplace=True)
mat=df[['a0','a1','a2','a3','a4','a5','a6']].values

mat.shape
df['agent_id']=np.argmax(mat,axis=1)
df['agent_id']
df.drop(['a0','a1','a2','a3','a4','a5','a6'],axis=1,inplace=True)
df.head()
X,y=df[[col for col in df if col!='label']],df['label']
tdf=pd.read_csv("../input/bits-f464-l1/test.csv")
id=tdf['id']
tdf.drop(["id"],axis=1,inplace=True)

mat=tdf[['a0','a1','a2','a3','a4','a5','a6']].values

mat.shape
tdf['agent_id']=np.argmax(mat,axis=1)
tdf.drop(['a0','a1','a2','a3','a4','a5','a6'],axis=1,inplace=True)
tdf.head()
ada=AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=None),n_estimators=100)

training_start = time.perf_counter()

ada.fit(X, y)

training_end = time.perf_counter()

prediction_start = time.perf_counter()

preds_train=ada.predict(X)

prediction_end = time.perf_counter()

mse_ada_train=mean_squared_error(y, preds_train, squared=False)

ada_train_time = training_end-training_start

ada_prediction_time = prediction_end-prediction_start

print("br training prediction mse is: %f" % (mse_ada_train))

print("Time consumed for training: %4.3f seconds" % (ada_train_time))

print("Time consumed for prediction: %6.5f seconds" % (ada_prediction_time))
preds=ada.predict(tdf)

preds
pd.DataFrame({'id':id,'label':preds}).to_csv('sub1.csv',index=False)