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
from sklearn.metrics import roc_auc_score
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']
train_y = df_train.loc[:, 'O/P']
# df_test = df_test.loc[:, 'F3':'F17']
df_test.head()
train_entries = len(train_X.F3)
test_entries = len(df_test.F3)
print(train_entries , test_entries)
df=pd.concat([train_X , df_test])
df.tail()
print(len(df.F3) , len(df.F6))
cols = ['F3', 'F4' , 'F5' , 'F7' , 'F8' , 'F9' , 'F11' , 'F12']

df = pd.get_dummies(df , columns = cols)


df.head()
print(len(df.F6))
import seaborn as sns
sns.distplot(train_y)
# df.corr(method ='pearson')
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train_me = df[:train_entries]
X_test =df[train_entries:]
print(len(train_y))

# X_test.head()
# train_y.head()
# rf = RandomForestRegressor(n_estimators=50, random_state=43)

X_train, X_val, y_train, y_val = train_test_split(X_train_me, train_y, test_size=0.20, random_state=29)
print(len(X_train) , len(y_train))
import xgboost as xgb
model = xgb.XGBRegressor()
model.fit(train_X, train_y)

# import sklearn
# model = xgb.XGBRegressor(learning_rate =0.1,
#  n_estimators=1000)
# model.fit(X_train , y_train)
# learning_rates = [0.1, 0.2]
# the_estimators = [50*x for x in range(40,51)]
# params = []
# min_acc= 100000
# accuracies=[]
# for l in learning_rates:
#     for n in the_estimators:
#         model = xgb.XGBRegressor(learning_rate =l,n_estimators=n)
#         model.fit(X_train , y_train)
#         pred= model.predict(X_val)
#         acc=sklearn.metrics.mean_squared_error(pred, y_val)
#         print(acc)
#         accuracies.append([acc , l , n])
#         if acc<min_acc:
#             params = [l,n]
#             min_acc= acc
        
import sklearn
model = xgb.XGBRegressor(learning_rate =0.1,n_estimators=1000)
model.fit(X_train , y_train)
pred= model.predict(X_val)
acc=sklearn.metrics.mean_squared_error(pred, y_val)
print(acc)
pred= model.predict(X_val)
print(pred)
import sklearn
sklearn.metrics.mean_squared_error(pred, y_val)
# print(y_val)
df_test = df_test.loc[:, 'F3':'F17']
pred = model.predict(df_test)
# model.fit(X_train_me, train_y)
print(len(X_train_me.F6) , len(train_y))
print(len(df))
# df_test = df_test.loc[:, 'F3':'F17']
print(train_entries , test_entries , len(df.F6))
# print(df[train_entries:])
pred = model.predict(X_test)
print(pred)
result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output.csv', index=False)

