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
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
df_train.head()
df_train.drop(['F1', 'F2','Unnamed: 0'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3':'F17']

train_y = df_train.loc[:, 'O/P']
scaler=StandardScaler().fit(train_X)

train_X=scaler.transform(train_X)
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,test_size=0.1,random_state=43)
test_index=df_test['Unnamed: 0']

df_test = df_test.loc[:, 'F3':'F17']

df_test=scaler.transform(df_test)
model = CatBoostRegressor(

    n_estimators = 10000,random_seed=42)

model.fit(train_X, train_y, eval_set=(test_X, test_y),verbose=1000,plot=True)
pred_train=model.predict(train_X)

pred_test=model.predict(test_X)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(train_y,pred_train))

print(mean_squared_error(test_y,pred_test))
pred = model.predict(df_test)
print(pred)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result.head()
result.to_csv('output.csv', index=False)