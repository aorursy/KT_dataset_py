
import numpy as np
import pandas as pd
#from fancyimpute import KNN

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score,accuracy_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
# from imblearn.over_sampling import SMOTE
# from sklearn.linear_model import LogisticRegression

# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense , Input , Dropout
from keras.wrappers.scikit_learn import KerasRegressor
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
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_train.head()
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.describe()
cat_feat=["F3","F4","F5","F7","F8","F9","F11","F12"]
test_index=df_test['Unnamed: 0']

df_train.drop(['Unnamed: 0','F1', 'F2'], axis = 1, inplace = True)

df_test.drop(['Unnamed: 0','F1', 'F2'], axis = 1, inplace = True)
df_test.nunique()
df_train.nunique()
(df_test.isnull().sum()).sum()
#13 and 15 too much corr

Y_train=df_train['O/P']
Y_train=Y_train.to_numpy()
X_train=df_train.loc[:, df_train.columns != 'O/P']
X_test=df_test.loc[:,df_test.columns!='O/P']
X_train.dtypes
# X_train["F17"]=X_train["F17"].astype(str)
# X_test["F17"]=X_test["F17"].astype(str)
from catboost import CatBoostRegressor,Pool

x=list(df_train.columns)
num_feat=[]
for k in x:
    if k in cat_feat or k=='O/P':
        continue
    else:
        num_feat.append(k)
num_feat
X_trg_norm=(X_train-X_train.min())/(X_train.max()-X_train.min())
X_tsg_norm=(X_test-X_test.min())/(X_test.max()-X_test.min())
Xtrnorm=X_train
Xtsnorm=X_test
Xtrnorm[num_feat]=X_trg_norm[num_feat]
Xtsnorm[num_feat]=X_tsg_norm[num_feat]
Xtrnorm.describe()
Xtsnorm.describe()
cbdat=Pool(data=X_train,label=Y_train,cat_features=cat_feat)
cbr=CatBoostRegressor()
#39.008-->f10 and f17--->lb 0.89
#38.88-->f10-->lb 0.7
#38.5somethign-->no normal all feat-->lb 0.67
#38.48 2000 trees

grid = {'learning_rate': [0.1,0.07,0.12],
        'depth': [9],
        'l2_leaf_reg': [2],
       'iterations':[1500,1700,2000,2200,2500]}

grid_search_result = cbr.grid_search(grid, 
                                       X=cbdat,  
                                       plot=True)
cbr=CatBoostRegressor(depth=9,learning_rate=0.1,l2_leaf_reg=2,iterations=2000)
cbr.fit(cbdat)
res=cbr.predict(X_test)
combo=pd.concat(objs=[X_train,X_test])
combo.describe()
combo.nunique()
combo=pd.get_dummies(data=combo,columns=["F3","F4","F5","F7","F8","F9","F11","F12"],dummy_na=True,drop_first=True)
combo.head()
X_train_dummy=pd.DataFrame(data=combo[0:Y_train.shape[0]])
X_train_dummy.describe()
X_test_dummy=pd.DataFrame(data=combo[Y_train.shape[0]:])
X_test_dummy.describe()
X_train_normalized_full = scale(X_train_dummy)
X_test_normalized=scale(X_test_dummy)
X_train_normalized_full.shape
Xx=X_train_normalized_full.T.T
Xx.shape
from keras.constraints import maxnorm
def create_model(wextra,wwextra,xtra,kk,k=0.5,wc=1):
# create model
    
    
    model = Sequential()
    model.add(Input(shape=(64,)))
    model.add(Dense(xtra,kernel_initializer='glorot_uniform', activation='relu', kernel_constraint=maxnorm(wc)))
    model.add(Dropout(k))
    model.add(Dense(wextra,kernel_initializer='glorot_uniform', activation='relu',kernel_constraint=maxnorm(wc)))
    model.add(Dropout(kk))
    model.add(Dense(wwextra,kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy','mean_squared_error'])
    return model
seed = 7
numpy.random.seed(seed)
model = KerasRegressor(build_fn=create_model, epochs=20, batch_size=200, verbose=5)
weight_constraint = [5]
dropout_rate = [0.1,0.2,0.3]
# dropout_rate2 = [0.1,0.2,0.3]
x=[40,35,30,25,20,15,10]
# z=[10,11,12]
y=[3,4,5,6,7,8]
param_grid = dict(k=dropout_rate, wc=weight_constraint,xtra=x,wextra=y)#,wwextra=z,kk=dropout_rate2)
print(param_grid)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3,verbose=10,scoring='neg_root_mean_squared_error')
grid_result = grid.fit(Xx, Y_train)
grid_result.best_score_
seed = 7
numpy.random.seed(seed)
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=200, verbose=5)
weight_constraint = [5]
dropout_rate = [0.1]
dropout_rate2 = [0.1]
x=[40]
z=[12]
y=[5]
param_grid = dict(k=dropout_rate, wc=weight_constraint,xtra=x,wextra=y,wwextra=z,kk=dropout_rate2)
print(param_grid)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5,verbose=10,scoring='neg_root_mean_squared_error')
grid_result = grid.fit(Xx, Y_train)


X_train, X_test, y_train, y_test = train_test_split(X_train_normalized,Y_train, test_size=0.2, random_state=42)
import lightgbm as lgb
lgbm = lgb.LGBMRegressor()
xgb=XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.1], #so called `eta` value
              'max_depth': [5,6],
              'subsample': [0.7,0.5],
              'colsample_bytree': [0.7,0.5],
              'n_estimators': [1500,1000,2000,2500]}
xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 3,
                        n_jobs = -1,
                        verbose=10,scoring='neg_root_mean_squared_error')

xgb_grid.fit(X_train_normalized_full,
         Y_train)


# -75.319
params={'colsample_bytree': 0.5,
 'learning_rate': 0.1,
 'max_depth': 5,
 'n_estimators': 1500,
 'nthread': 4,
 'objective': 'reg:linear',
 'subsample': 0.7}
xgb.fit(X_train_normalized_full,Y_train,**params)

xgb_grid.best_score_
xgb_grid.best_params_
df_test = df_test.loc[:, 'F3':'F17']
pred = res
#contains old
pred2=pred
print(pred2)
print(pred)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output_2000.csv', index=False)

