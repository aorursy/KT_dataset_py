# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train=train.drop(columns=['ID']).values
test=test.drop(columns=['ID']).values
X_train=train[:,1:]
Y_train=train[:,0]
Y_train=Y_train.reshape((Y_train.shape[0],1))
Y_train.shape
x=MinMaxScaler(feature_range=(-1,1))
y=MinMaxScaler(feature_range=(-1,1))
X_sca_train=x.fit_transform(X_train)
Y_sca_train=y.fit_transform(Y_train)
sca_test=x.transform(test)
pca=PCA(n_components=0.95,svd_solver="full")
X_pca_train=pca.fit_transform(X_sca_train)
pca_test=pca.transform(sca_test)
# from sklearn.tree import DecisionTreeRegressor
# dt=DecisionTreeRegressor()
# dt.fit(X_pca_train,Y_sca_train)
# pred=dt.predict(pca_test)
# pred
np.unique(pred>0,return_counts=True)
# from sklearn.linear_model import LinearRegression
# lr=LinearRegression()
# lr.fit(X_sca_pca_train,Y_sca_train)
# pred=lr.predict(sca_pca_test)
# pred.shape
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=20,max_features="log2",n_jobs=-1)
rf.fit(X_sca_train,Y_sca_train)
pred=rf.predict(sca_test)
pred.shape
pred=pred.reshape((pred.shape[0],1))
pred=y.inverse_transform(pred)
pred
sampledata=pd.read_csv("../input/sample_submission.csv")
sampledata.target=pred
sampledata.to_csv("sample_submission.csv",index=False)

