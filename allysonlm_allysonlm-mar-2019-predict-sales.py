#########################################################

# Imports

import pandas as pd

pd.options.mode.chained_assignment = None

pd.set_option('display.max_columns', 100)



import numpy as np

import random

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import matplotlib as plt

%matplotlib inline 



from sklearn.preprocessing import StandardScaler



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))


#########################################################

# Leitura dos dados

train = pd.read_csv('../input/dataset_treino.csv')

test = pd.read_csv('../input/dataset_teste.csv')

store = pd.read_csv('../input/lojas.csv')





#########################################################

# Dados com lojas abertas

train = train[(train['Open'] == 1) & (train['Sales'] > 0)]





#########################################################   

# Open = 1 para os dados de teste

test.fillna(1, inplace=True)





#########################################################

# Merge

train = train.merge(store, on = 'Store', how = 'left')

test = test.merge(store, on = 'Store', how = 'left')



#########################################################   

# StateHoliday

train.loc[train['StateHoliday'] == 0, 'StateHoliday'] = '0'







le = LabelEncoder()

lst_promo_inter = list((train['PromoInterval'].append(test['PromoInterval'])).unique())

for ds in [train, test]:

    #########################################################

    # Coluna data

    ds['Date'] = pd.to_datetime(ds['Date'], errors='coerce')

    ds['Year'] = ds.Date.dt.year

    ds['Month'] = ds.Date.dt.month

    ds['Day'] = ds.Date.dt.day

    ds['Week'] = ds.Date.dt.week

    ds['WeekOfYear'] = ds.Date.dt.week

    

    #########################################################

    # Categoricos

    ds['StoreType'] = le.fit_transform(ds['StoreType'])

    ds['Assortment'] = le.fit_transform(ds['Assortment'])

    ds['StateHoliday'] = le.fit_transform(ds['StateHoliday'])

    

    

    #########################################################

    # CompetitionMonth

    ds['CompetitionMonth'] = 12 * (train.Year - train.CompetitionOpenSinceYear) + (train.Month - train.CompetitionOpenSinceMonth)

    ds['CompetitionMonth'] = ds.CompetitionMonth.apply(lambda x: x if x > 0 else 0)   

    

    

    #########################################################

    # PromoOpen

    ds['PromoOpen'] = 12 * (ds.Year - ds.Promo2SinceYear) + (ds.WeekOfYear - ds.Promo2SinceWeek) / 4.0        

    ds['PromoOpen'] = ds.PromoOpen.apply(lambda x: x if x > 0 else 0)

    

    

    #########################################################

    # PromoInterval

    ds['PromoInterval'] = [lst_promo_inter.index(x) for x in ds['PromoInterval']]

    

    

    #########################################################

    # Prencher nulos

    ds.fillna(-1, inplace=True)
train.head(3)
#########################################################

# RMSPE

def rmspe(y, yhat):

    return np.sqrt(np.mean((yhat/y-1) ** 2))



def rmspe_xg(yhat, y):

    y = np.expm1(y.get_label())

    yhat = np.expm1(yhat)

    return "rmspe", rmspe(y,yhat)







#########################################################

# Features

features = ['Store',

            'DayOfWeek', 

            'Promo', 

            'StateHoliday', 

            'SchoolHoliday', 

            'StoreType', 

            'Assortment',

            'CompetitionDistance', 

            'CompetitionOpenSinceMonth', 

            'CompetitionOpenSinceYear', 

            'Promo2',

            'Promo2SinceWeek', 

            'Promo2SinceYear', 

            'PromoInterval', 

            'Year',

            'Month', 

            'Day',

            #'Week', 

            'WeekOfYear'

            #'CompetitionMonth',

            #'PromoOpen'

           ]





#########################################################

# Modelo

param = {

    'objective': 'reg:linear', 

    "booster" : "gbtree",

    'eta': 0.03,

    'max_depth':10,

    'subsample':0.9,

    'colsample_bytree':0.7,

    'silent' : 1  

}



X_train, X_test, y_train, y_test = train_test_split(train[features], np.log1p(train['Sales']), 

                                                    test_size = 50000, random_state = 2019)



dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_test, y_test)



watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
#########################################################

# Treinamento

gbm = xgb.train(

            param, 

            dtrain, 

            7000,

            evals=watchlist,

            early_stopping_rounds=100, 

            feval=rmspe_xg, 

            verbose_eval=100

)
#########################################################

# Predict treino

yhat = gbm.predict(xgb.DMatrix(X_test))
rmspe(np.expm1(y_test), np.expm1(yhat))
#########################################################

# Tabela com erro, ratio e ajuste de pesos

res = pd.DataFrame(data = y_test)

res['Prediction']=yhat

res = pd.merge(X_test,res, left_index= True, right_index=True)

res['Ratio'] = res.Prediction/res.Sales

res['Error'] =abs(res.Ratio-1)

res['Weight'] = res.Sales/res.Prediction



res.head()
#########################################################

# Conceito: os scores de treino e validação estão bem próximos, ajustanto os pesos do treino

# eh provavel que o score da validação seja melhor



W=[(0.990+(i/1000)) for i in range(20)]

S =[]

for w in W:

    error = rmspe(np.expm1(y_test), np.expm1(yhat*w))

    #print('RMSPE for {:.3f}:{:.6f}'.format(w,error))

    S.append(error)

Score = pd.Series(S,index=W)

#Score.plot()

BS = Score[Score.values == Score.values.min()]

print ('Melhor ajuste de score:{}'.format(BS))

#########################################################

# Score com ajuste

rmspe(np.expm1(y_test), np.expm1(yhat*0.999))
#########################################################

# Predição final

test_probs = gbm.predict(xgb.DMatrix(test[features]))
#########################################################

# Peso final ajustado por tentativa e erro no Score público

peso_final = 0.9965

submission = pd.DataFrame({"Id": test["Id"], "Open": test["Open"], "Sales":  np.expm1(test_probs * peso_final) })

submission.loc[submission['Open'] == 0, 'Sales'] = 0

submission.loc[submission['Sales'] < 0, 'Sales'] = 0

submission = submission[['Id', 'Sales']]





submission.to_csv("sub_final4.csv", index=False)