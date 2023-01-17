# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import DataFrame

import datetime

from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold # 追加



from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_trainval = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv')

print(df_trainval.head())
rev = df_trainval['revenue']

del df_trainval['revenue']
df_all = pd.concat([df_trainval,df_test],axis=0)

df_all['Open Date'] = pd.to_datetime(df_all["Open Date"])

df_all['Year'] = df_all['Open Date'].apply(lambda x:x.year)

df_all['Month'] = df_all['Open Date'].apply(lambda x:x.month)

df_all['Day'] = df_all['Open Date'].apply(lambda x:x.day)



le = LabelEncoder()

df_all['City'] = le.fit_transform(df_all['City'])

df_all['City Group'] = df_all['City Group'].map({'Other':0,'Big Cities':1})

df_all["Type"] = df_all["Type"].map({"FC":0, "IL":1, "DT":2, "MB":3})

print(df_all.head())
df_trainval = df_all.iloc[:df_trainval.shape[0]]

df_test = df_all.iloc[df_trainval.shape[0]:]
df_train_col = [col for col in df_trainval.columns if col not in ['Id','Open Date']]
sc = StandardScaler()

ms = MinMaxScaler()
df_trainval_sc = sc.fit_transform(df_trainval[df_train_col])

df_trainval_sc_ms = ms.fit_transform(df_trainval_sc)

print(DataFrame(df_trainval_sc_ms).head())
def gen_cv():

    m_train = np.floor(len(rev)*0.75).astype(int)#このキャストをintにしないと後にハマる

    train_indices = np.arange(m_train)

    test_indices = np.arange(m_train, len(rev))

    yield (train_indices, test_indices)
params_cnt = 20

params = {"C":np.logspace(0,1,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}

gridsearch = GridSearchCV(SVR(kernel="linear"), params, cv=gen_cv(), scoring="r2", return_train_score=True)

gridsearch.fit(df_trainval_sc_ms, rev)

print('The best parameter = ',gridsearch.best_params_)

print('accuracy = ',gridsearch.best_score_)

print()



regr = SVR(kernel="linear", C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
splits = 5

kf = KFold(n_splits=splits,shuffle=True,random_state=0)



mlp = MLPRegressor(activation='relu',

                  solver='adam',

                  batch_size=100,

                  max_iter=2000,

                   hidden_layer_sizes=(16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,)

                  )

rf = RandomForestRegressor(n_estimators=1000,

                           max_depth=30,

                           random_state=0,

                           n_jobs=-1)



rmse_list = []

#models_mlp = []

#models_rf = []

#models_svr = []

for train_index,test_index in kf.split(df_trainval_sc_ms):

    X_train = df_trainval.iloc[train_index]

    Y_train = rev.iloc[train_index]

    X_valid = df_trainval.iloc[test_index]

    Y_valid = rev.iloc[test_index]

    model_mlp = mlp.fit(X_train[df_train_col], Y_train)

    model_rf = rf.fit(X_train[df_train_col], Y_train)

    model_svr = regr.fit(X_train[df_train_col], Y_train)

    models_mlp.append(model_mlp)

    models_rf.append(model_rf)

    models_svr.append(model_svr)

    prediction_rf = rf.predict(X_valid[df_train_col])

    prediction_mlp = mlp.predict(X_valid[df_train_col])

    prediction_regr = regr.predict(X_valid[df_train_col])

    prediction = (prediction_rf + prediction_mlp + prediction_regr) / 3

    val_rmse = mean_absolute_error(Y_valid,prediction)

    print(val_rmse)

    rmse_list.append(val_rmse)

print('average rmse : {0}'.format(sum(rmse_list)/len(rmse_list)))
df_test_sc = sc.transform(df_test[df_train_col])

df_test_sc_ms = ms.fit_transform(df_test_sc)
mlp.fit(df_trainval_sc_ms, rev)

rf.fit(df_trainval_sc_ms, rev)

regr.fit(df_trainval_sc_ms, rev)



prediction_rf = rf.predict(df_test_sc_ms)

prediction_mlp = mlp.predict(df_test_sc_ms)

prediction_regr = regr.predict(df_test_sc_ms)



prediction = (prediction_rf + prediction_mlp + prediction_regr) / 3



submission = DataFrame({'Id':df_test.Id,'Prediction':prediction})
submission.to_csv('./submission191019.csv',index=False)

!ls && pwd