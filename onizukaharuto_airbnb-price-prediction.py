import pandas as pd # データフレームやデータ処理を行うライブラリ

import numpy as np # 数値計算を行うライブラリ

import os # PythonからOSの機能を使用するライブラリ

import joblib # データの保存や並列処理

import matplotlib.pyplot as plt # 可視化

import seaborn as sns # pltをラッパーした可視化

import optuna # パラメータチューニング

%matplotlib inline

sns.set() # snsでpltの設定をラッパー
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
train_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/test.csv', index_col=0)
train_df.info()
target_col = "price"
plt.hist(train_df[target_col], bins=100)
plt.hist(np.log1p(train_df[target_col]))
import scipy.stats as stats

import pylab

stats.probplot(train_df[target_col], dist="norm", plot=pylab)

plt.show()
stats.probplot(np.log1p(train_df[target_col]), dist="norm", plot=pylab)

plt.show()
df = pd.concat([train_df.drop('price', axis=1), test_df])

df
df = pd.concat([df, pd.get_dummies(df['neighbourhood_group'])], axis=1)

df = df.drop('neighbourhood_group', axis=1)
df = pd.concat([df, pd.get_dummies(df['neighbourhood'])], axis=1)

df = df.drop('neighbourhood', axis=1)
df = pd.concat([df, pd.get_dummies(df['room_type'])], axis=1)

df = df.drop('room_type', axis=1)
df['last_review'] = df['last_review'].fillna(0)

df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df = df.drop(['name', 'host_id', 'host_name'], axis=1)

df
nrow, ncol = train_df.shape

price_df = train_df[['price']]

train_df = df[:nrow]

train_df = pd.concat([train_df, price_df], axis=1)

train_df
nrow, ncol = train_df.shape

test_df = df[nrow:]

test_df
X = train_df.drop(['price'], axis=1).to_numpy()

y = np.log1p(train_df['price']).to_numpy()

y_ = train_df['price'].to_numpy()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

len(X_train),len(X_valid),len(y_train),len(y_valid)
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state=0)

dtr.fit(X_train, y_train)

predict = np.expm1(dtr.predict(X_valid))

#predict = dtr.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(random_state=0)

rfr.fit(X_train, y_train)

predict = np.expm1(rfr.predict(X_valid))

#predict = rfr.predict(X_valid)

np.sqrt(mean_squared_error(y_valid,predict))
import optuna

def objective(trial):

    max_depth = trial.suggest_int('max_depth', 1, 30)

    n_estimators = trial.suggest_int('n_estimators',10,300)

    model = RandomForestRegressor(criterion='mse', max_depth=max_depth, n_estimators=n_estimators, random_state=0,n_jobs=-1)

    model.fit(X_train, y_train)

    y_pred = np.expm1(model.predict(X_valid))

    return np.sqrt(mean_squared_error(y_valid, y_pred))



study = optuna.create_study()

study.optimize(objective, n_trials=100)

study.best_params
max_depth = study.best_params['max_depth']

n_estimators = study.best_params['n_estimators']

model = RandomForestRegressor(criterion='mse', max_depth=max_depth, n_estimators=n_estimators, random_state=0,n_jobs=-1)

model.fit(X_train, y_train)

predict = np.expm1(model.predict(X_valid))

np.sqrt(mean_squared_error(y_valid,predict))
model.fit(X,y)
X = test_df.to_numpy()



p = np.expm1(rfr.predict(X))
submit_df = pd.read_csv('/kaggle/input/1056lab-airbnb-price-predction/sampleSubmission.csv')

submit_df['price'] = p

submit_df.to_csv('submission.csv', index=False)