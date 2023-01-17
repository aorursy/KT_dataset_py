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
df=pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.shape
df.describe().T
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,fmt=".2f")
plt.show()
plt.figure(figsize=(15,7))
sns.heatmap(df.describe().T,annot=True,fmt=".2f")
plt.show()
df.info()
df.isnull().sum().sum()
dff=df.copy()
from sklearn import preprocessing
lbe=preprocessing.LabelEncoder()
dff['sex']=lbe.fit_transform(dff['sex'])
dff['smoker']=lbe.fit_transform(dff['smoker'])
dff['region']=lbe.fit_transform(dff['region'])
dff.head()
dff.sex.value_counts()
dff.smoker.value_counts()
dff.region.value_counts()
dff.head()
x=dff.drop(['charges'],axis=1)
y=dff['charges']
x.head()
y[0:5]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,
                                              y,
                                              test_size=0.25,
                                              random_state=42)
print('x_train shape:',x_train.shape)
print('x_test shape:',x_test.shape)
print('y_train shape:',y_train.shape)
print('y_test shape:',y_test.shape)
from sklearn.ensemble import GradientBoostingRegressor
#model:
gbm_model=GradientBoostingRegressor()
gbm_model.fit(x_train,y_train)
#predict:
y_pred=gbm_model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
gbm_params={
'learning_rate':[0.001,0.01],
'max_depth':[3,5,8],
'n_estimators':[50,100],
'subsample':[1,0.5]
}
from sklearn.model_selection import GridSearchCV
gbm=GradientBoostingRegressor()
gbm_cv=GridSearchCV(gbm,
                    gbm_params,
                    cv=10,
                    verbose=2).fit(x_train,y_train)
#best parameters:
gbm_cv.best_params_
#final model:
gbm_tuned_model=GradientBoostingRegressor(learning_rate=0.01,
                                         max_depth=3,
                                         n_estimators=100,
                                         subsample=0.5).fit(x_train,y_train)
y_pred=gbm_tuned_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
!pip install xgboost
import xgboost as xgb
from xgboost import XGBRegressor
#model:
xgb_model=XGBRegressor().fit(x_train,y_train)  #model
y_pred=xgb_model.predict(x_test)                #predict
print(np.sqrt(mean_squared_error(y_test,y_pred)))    #error
print(r2_score(y_test,y_pred))                      #score
xgb_params={'colsample_bytree':[0.4,0.5],
          'n_estimators':[50,100],
          'max_depth':[2,3],
          'learning_rate':[0.1,0.5]}
xgb_cv_model=GridSearchCV(xgb_model,
                         xgb_params,
                         cv=10,
                         verbose=2).fit(x_train,y_train)
xgb_cv_model.best_params_
#final model:
xgb_tuned_model=XGBRegressor(colsample_bytree=0.5,
                            learning_rate=0.5,
                            max_depth=3,
                            n_estimators=100).fit(x_train,y_train)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
from catboost import CatBoostRegressor
catb=CatBoostRegressor()
catb_model=catb.fit(x_train,y_train)
y_pred=catb_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
#model tuning:
catb_params={
    'iterations':[40,50],
    'learning_rate':[0.1,0.2],
    'depth':[3,4,5]
}
catb=CatBoostRegressor()
catb_cv_model=GridSearchCV(catb,
                           catb_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=2).fit(x_train,y_train)
catb_cv_model.best_params_
#final model:
catb_tuned_model=CatBoostRegressor(depth=4,
                                  iterations=50,
                                  learning_rate=0.1).fit(x_train,y_train)
y_pred=catb_tuned_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
