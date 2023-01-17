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
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")
data.info()
data.head(5)
data.drop('Unnamed: 0',axis=1,inplace=True)
plt.subplots(figsize=(20,20))

sns.heatmap(data.corr(),annot=True,square=True)
sns.pairplot(data)
cols = data.columns

num_cols = data._get_numeric_data().columns

cat_cols = list(set(cols) - set(num_cols))
data[cat_cols]
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')
encode = ohe.fit_transform(data[cat_cols])

features = ohe.get_feature_names().tolist()

data1 = pd.DataFrame(encode,columns=features)

data1
dataf = data[num_cols].join(data1)
dataf
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
X = dataf.drop('price',axis=1)#.values.reshape(1,-1)

y = dataf['price']#.values.reshape(1,-1)



X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=101)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =10, nthread = -1)
model_xgb.fit(X_train,y_train)
y_pred = model_xgb.predict(X_test)

print("XGB score",model_xgb.score(X_train,y_train))

print('all gives R2 score',r2_score(y_pred,y_test))

print('all gives MSE is:',mean_squared_error(y_test, y_pred))

rms = np.sqrt(mean_squared_error(y_test, y_pred))

print('all gives RMSE is:',rms)

print("-----------------------------------------")