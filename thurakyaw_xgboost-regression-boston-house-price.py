# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_boston

boston = load_boston()
boston.keys()
boston.data.shape
boston.feature_names
print(boston.DESCR)
data=pd.DataFrame(boston.data)

data.columns = boston.feature_names
data.head()
data['PRICE']=boston.target
data.head()
data.info()
data.describe()
import xgboost as xgb

from sklearn.metrics import mean_squared_error



X,y = data.iloc[:,:-1],data.iloc[:,-1]
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.3,learning_rate=0.1,max_depth=5,alpha=10,n_estimators=10)
xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,preds))

print ('RMSE=%f'% (rmse))
params = {"obective":"reg:linear",'colsample_bytree':0.3,'learning_rate':0.1,'max_depth':5,'alpha':10}

cv_results = xgb.cv(dtrain=data_dmatrix,params = params,nfold=3,num_boost_round=50,early_stopping_rounds=10,metrics ='rmse',as_pandas = True, seed=123)
cv_results.head()
cv_results['test-rmse-mean'].tail(3)
xgb_reg=xgb.train(params= params,dtrain=data_dmatrix,num_boost_round=10)
import matplotlib.pyplot as plt



xgb.plot_tree(xgb_reg,num_trees=0)

plt.rcParams['figure.figsize']=(70,30)

plt.show()
xgb.plot_importance(xgb_reg)

plt.rcParams['figure.figsize']=[10,5]

plt.show()