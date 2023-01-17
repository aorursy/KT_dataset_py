# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import xgboost as xgb





# forum rossmann retirada as funçoes pra cauculo rmspe

def ToWeight(y):

    w = np.zeros(y.shape, dtype=float)

    ind = y != 0

    w[ind] = 1./(y[ind]**2)

    return w





def rmspe(yhat, y):

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))

    return rmspe





def rmspe_xg(yhat, y):

    # y = y.values

    y = y.get_label()

    y = np.exp(y) - 1

    yhat = np.exp(yhat) - 1

    w = ToWeight(y)

    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))

    return "rmspe", rmspe





# tratamento de variaveis

def build_features(features, data):

    # remove NaNs

    data.fillna(0, inplace=True)

    data.loc[data.Open.isnull(), 'Open'] = 1

    # adicionar variaveis

    features.extend(['Store', 'CompetitionDistance', 'CompetitionOpenSinceMonth',

                      'CompetitionOpenSinceYear', 'Promo', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear'])

    #testarcombinaçoes

    #features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2'])



    # tratamentogeral

    features.append('StateHoliday')

    data.loc[data['StateHoliday'] == 'a', 'StateHoliday'] = '1'

    data.loc[data['StateHoliday'] == 'b', 'StateHoliday'] = '2'

    data.loc[data['StateHoliday'] == 'c', 'StateHoliday'] = '3'

    data['StateHoliday'] = data['StateHoliday'].astype(float)

    #separando data em ida,mes,ano,dia da semana

    features.append('DayOfWeek')

    features.append('month')

    features.append('day')

    features.append('year')

    data['year'] = data.Date.apply(lambda x: x.split('-')[0])

    data['year'] = data['year'].astype(float)

    data['month'] = data.Date.apply(lambda x: x.split('-')[1])

    data['month'] = data['month'].astype(float)

    data['day'] = data.Date.apply(lambda x: x.split('-')[2])

    data['day'] = data['day'].astype(float)

    #transformar categorica em int

    features.append('StoreType')

    data.loc[data['StoreType'] == 'a', 'StoreType'] = '1'

    data.loc[data['StoreType'] == 'b', 'StoreType'] = '2'

    data.loc[data['StoreType'] == 'c', 'StoreType'] = '3'

    data.loc[data['StoreType'] == 'd', 'StoreType'] = '4'

    data['StoreType'] = data['StoreType'].astype(float)

    #transformar categorica em int

    features.append('Assortment')

    data.loc[data['Assortment'] == 'a', 'Assortment'] = '1'

    data.loc[data['Assortment'] == 'b', 'Assortment'] = '2'

    data.loc[data['Assortment'] == 'c', 'Assortment'] = '3'

    data['Assortment'] = data['Assortment'].astype(float)



print("Carregando dados com pandas")

train = pd.read_csv("../input/dataset_treino.csv",dtype={'StateHoliday':pd.np.string_})

test = pd.read_csv("../input/dataset_teste.csv",dtype={'StateHoliday':pd.np.string_})

store = pd.read_csv("../input/lojas.csv",dtype={'StateHoliday':pd.np.string_})



print("Se Open for NaN tranformar em 1 = aberto")

test.fillna(1, inplace=True)



print("Usar apenas as lojas abertas para treinamento")

train = train[train["Open"] != 0]



print("Join Train e Test com lojas")

train = pd.merge(train, store, on='Store')

test = pd.merge(test, store, on='Store')



features = []



print("Variaveis:")

build_features(features, train)

build_features([], test)

print(features)



depth = 10

eta = 0.02

ntrees = 3000



params = {"objective": "reg:linear",

          "booster": "gbtree",

          "eta": eta,

          "max_depth": depth,

          "subsample": 0.8,

          "colsample_bytree": 0.7,

          "silent": 0,

          "tree_method":"gpu_hist"

          }



print("Trainando a XGBoost model")



tsize = 0.3

X_train, X_test = train_test_split(train, test_size=tsize)



dtrain = xgb.DMatrix(X_train[features], np.log(X_train["Sales"] + 1))

dvalid = xgb.DMatrix(X_test[features], np.log(X_test["Sales"] + 1))

dtest = xgb.DMatrix(test[features])

watchlist = [(dvalid, 'eval'), (dtrain, 'train')]

gbm = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xg)



print("Validadando Modelo:")

train_probs = gbm.predict(xgb.DMatrix(X_test[features]))

indices = train_probs < 0

train_probs[indices] = 0

error = rmspe(np.exp(train_probs) - 1, X_test['Sales'].values)

print('error', error)



print("Efetuando predições com o dataset de teste")

test_probs = gbm.predict(xgb.DMatrix(test[features]))

indices = test_probs < 0

test_probs[indices] = 0

submission = pd.DataFrame({"Id": test["Id"], "Sales": np.exp(test_probs) - 1})



# lojas fechados vendas = 0

def fix_closed(row):

    if test[test['Id'] == row['Id']]['Open'].values[0] == 0:

        return 0

    else:

        return row['Sales']

submission['Sales'] = submission.apply(fix_closed, axis=1)

print ('Variaveis:')

print (features)

print ('Parametros:')

print (params)

print ('Arquivo de saida:')

print ("xgb_d%s_eta%s_ntree%s_diogo_2_.csv" % (str(depth),str(eta),str(ntrees)))

submission.to_csv("xgb_d%s_eta%s_ntree%s_diogo_2_.csv" % (str(depth),str(eta),str(ntrees)) , index=False)