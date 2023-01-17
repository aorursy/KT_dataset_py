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



import os

print((os.listdir('../input/')))
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.datasets import make_regression

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)

df_test.drop(['F1','F2'],axis =1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']

train_y = df_train.loc[:, 'O/P']

df_test = df_test.loc[:, 'F3':'F17']
train_X.hist(figsize = (20,15))
df_test.hist(figsize=(20,15))
train_X.head()
df_test.head()
train_X.info()
df_test.info()
label_X_train = train_X.copy()

label_X_test = df_test.copy()
label_X_train.F3.unique()
label_X_test.F3.unique()
label_X_train.F5.unique()
label_X_train.F12.unique()
label_X_train = pd.get_dummies(label_X_train,columns = ['F3'])#,'F5','F7','F9','F12'])

label_X_test = pd.get_dummies(label_X_test, columns = ['F3'])#,'F5','F7','F9','F12'])

label_X_train, label_X_test = label_X_train.align(label_X_test, join='inner', axis=1)
print(label_X_test.shape)

print(label_X_train.shape)
label_X_test.head()
#X_train_X, val_X, y_train_y, val_y = train_test_split(train_X,train_y,test_size = 0.2,random_state = 42)
rf = XGBRegressor(random_state=0,n_estimators=1100,

                          learning_rate=0.06, missing = 0) 

#rf = RandomForestRegressor(n_estimators=50, random_state=43)

#rf = XGBRegressor()

#rf = lgb.LGBMRegressor()
'''parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:squared error'],

              'learning_rate': [.03, 0.05], #so called `eta` value

              'max_depth': [5, 6],

              'min_child_weight': [4],

              'silent': [0],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [250]}



xgb_grid = GridSearchCV(rf,

                        parameters,

                        cv = 3,

                        n_jobs = 5,

                        verbose=True)'''
#xgb_grid.fit(train_X,train_y)
#print(xgb_grid.best_score_)

#print(xgb_grid.best_params_)
#rf.fit(dummy_X,train_y)

#rf.fit(train_X, train_y)

#rf.fit(X_train_X, y_train_y,eval_set=[(val_X, val_y)],early_stopping_rounds=200)
rf.fit(label_X_train, train_y)
importance = rf.feature_importances_

for i,v in enumerate(importance):

    print("Importance of feature F%0d, %0.5f" % (i+1,v))
scores = -1*cross_val_score(rf,label_X_train,train_y,cv=3,scoring = 'neg_root_mean_squared_error')

print(scores)

print(scores.mean())
rf.score(label_X_train, train_y)
predict_p = rf.predict(label_X_test)
print(predict_p)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(predict_p)

result.head()
result.tail()
result.to_csv('output.csv', index=False)