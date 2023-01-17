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
subs = pd.read_csv(f'/kaggle/input/flkassignment/dataset.csv')

subs.head()
del subs['datetime']

cols = list(subs.columns.values) #Make a list of all of the columns in the df

cols.pop(cols.index('count'))

subs = subs[cols+['count']] 

subs.head()
train =(subs.values[:, :-1])

test = subs.values[:, -1:]
print(train.shape)

print(test.shape)

from sklearn import model_selection





from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV



X_train,X_test,Y_train,Y_test=model_selection.train_test_split(train,test,test_size=0.25,random_state=0,shuffle=False)
import xgboost as xgb

from sklearn.metrics import mean_squared_error



from sklearn import linear_model,tree,svm,ensemble,neighbors

regressors=[

    linear_model.LinearRegression(),#

    linear_model.Ridge(),#

    linear_model.Lasso(),#

    ensemble.GradientBoostingRegressor(),#

    ensemble.RandomForestRegressor(n_estimators=10),#

    ensemble.RandomForestRegressor(n_estimators=20),#

    ensemble.RandomForestRegressor(n_estimators=30),#

    ensemble.RandomForestRegressor(n_estimators=40),#

    ensemble.RandomForestRegressor(n_estimators=50),#

    ensemble.RandomForestRegressor(n_estimators=60),#

    

    xgb.XGBRegressor(n_estimators=60),#

    xgb.XGBRegressor(n_estimators=70),#

    xgb.XGBRegressor(n_estimators=80),#

    xgb.XGBRegressor(n_estimators=90),#

    xgb.XGBRegressor(n_estimators=100),#

 

    neighbors.KNeighborsRegressor(n_neighbors=1),

    neighbors.KNeighborsRegressor(n_neighbors=2),

    neighbors.KNeighborsRegressor(n_neighbors=3),

        

    neighbors.KNeighborsRegressor(n_neighbors=4),

        

    neighbors.KNeighborsRegressor(n_neighbors=5),

        

    neighbors.KNeighborsRegressor(n_neighbors=6),

        

    neighbors.KNeighborsRegressor(n_neighbors=7),

        

    neighbors.KNeighborsRegressor(n_neighbors=8),

        

    neighbors.KNeighborsRegressor(n_neighbors=9),

        

    neighbors.KNeighborsRegressor(n_neighbors=10)#



]
for reg in regressors:

    reg.fit(X_train,Y_train)

    name=reg.__class__.__name__

    print('='*30)

    print(name)

    predictions=reg.predict(X_test)

    rmse=np.sqrt(mean_squared_error(Y_test,predictions))

    print("RMSError is: {}".format(rmse))