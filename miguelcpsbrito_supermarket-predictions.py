import time

start_time_all = time.time()



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import matplotlib.pyplot as plt
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train_set=pd.read_csv("../input/train.csv")

#oil_set=pd.read_csv("../input/oil.csv")

#mini_train=train_set

#mini_train.to_csv("mini_train.csv")

#mini_train=pd.read_csv("../working/mini_train.csv")

# Any results you write to the current directory are saved as output.

print('done!')
train_set.head()
size=train_set.shape[0]
from datetime import datetime

stores_set=pd.read_csv("../input/stores.csv")

#STORES ENCODE

#city state type cluster

from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

dstores = defaultdict(LabelEncoder)

df=stores_set.copy()

sto_nbr=pd.DataFrame(df['store_nbr'])

df=df.drop('store_nbr',axis=1)

# Encoding the variable

fite = df.apply(lambda x: dstores[x.name].fit_transform(x))

stores_encoded=pd.merge(sto_nbr,fite, left_index=True, right_index=True)

print(stores_encoded.head())

del stores_set

del fite

del sto_nbr

del df
#ITEMS ENCODE

#family class perishable

items_set=pd.read_csv("../input/items.csv")



items_set.head()

ditems = defaultdict(LabelEncoder)

dfi=items_set.copy()

ite_nbr=pd.DataFrame(dfi['item_nbr'])

dfi=dfi.drop('item_nbr',axis=1)

# Encoding the variable

fite = dfi.apply(lambda x: ditems[x.name].fit_transform(x))

items_encoded=pd.merge(ite_nbr,fite, left_index=True, right_index=True)

print(items_encoded.head())

del items_set

del fite

del ite_nbr

del dfi
from sklearn.metrics import mean_squared_error

from math import sqrt

from xgboost.sklearn import XGBRegressor

max_depth = 3

min_child_weight = 1#10

subsample = 1#0.5

colsample_bytree = 1#0.6

objective = 'reg:linear'

num_estimators = 1000

learning_rate = 0.2

silent=False

#booster='gblinear'

booster='gbtree'

model = XGBRegressor(

                    max_depth=max_depth, 

                    learning_rate=learning_rate, 

                    n_estimators=num_estimators, 

                    silent=silent, 

                    objective=objective, 

                    booster=booster, 

                    n_jobs=1, 

                    nthread=None, 

                    gamma=0, 

                    min_child_weight=min_child_weight, 

                    max_delta_step=0, 

                    subsample=subsample, 

                    colsample_bytree=colsample_bytree, 

                    colsample_bylevel=1, 

                    reg_alpha=0, 

                    reg_lambda=1, 

                    scale_pos_weight=1, 

                    base_score=0.5, 

                    random_state=0, seed=None,  missing=None

)   
start_time = time.time()

size=125497040

inicio=0

interval_size=1254970

i_trainset=interval_size#0#125497040#    testset 3370464

i=0

while i<4:

    i=i+1

    print(i)

    size=size-interval_size

    if size>0:

        mini_train=train_set.iloc[inicio:i_trainset,:]

        inicio=i_trainset

        i_trainset=i_trainset+interval_size

        print("done1")

        mini_train=mini_train.drop('id',axis=1)

        mini_train['Ano'] = mini_train['date'].map(lambda x: (datetime.strptime(x, '%Y-%M-%d')).year)

        mini_train['Mes'] = mini_train['date'].map(lambda x: (datetime.strptime(x, '%Y-%M-%d')).month)

        mini_train['Dia'] = mini_train['date'].map(lambda x: (datetime.strptime(x, '%Y-%M-%d')).day)

        mini_train['onpromotion']=(mini_train.isnull()['onpromotion'])=0

        mini_train.loc[mini_train['onpromotion'] ==False, 'onpromotion'] = 0

        mini_train.loc[mini_train['onpromotion'] ==True, 'onpromotion'] = 1



        t1=mini_train.merge(items_encoded,how='left', left_on=['item_nbr'], right_on = ['item_nbr'])

        t2=t1.merge(stores_encoded,how='left', left_on=['store_nbr'], right_on = ['store_nbr'])

        df=pd.DataFrame(t2)

        del t1

        del t2

        del mini_train

        print("Done4!")

        print("--- %s seconds ---" % (time.time() - start_time))

        print("--- %s seconds all---" % (time.time() - start_time_all))

        #create train set

        #size=df.shape[0]

        X_train = df.loc[:, (df.columns != 'unit_sales') & (df.columns != 'date') ]

        y_train = df.loc[:,(df.columns == 'unit_sales')]

        del df

        print("go training")

        start_time = time.time()

        model.fit(X_train, y_train)  

        print("Done")

        print("--- %s seconds ---" % (time.time() - start_time))

        print("--- %s seconds all---" % (time.time() - start_time_all))

        print("train %d %d " % (inicio , size))

del X_train

del y_train

#del y_train_pred

print("done")
#preparacao do set de test

start_time = time.time()

test_set=pd.read_csv("../input/test.csv")

print(test_set.shape)

test_set['Ano'] = test_set['date'].map(lambda x: (datetime.strptime(x, '%Y-%M-%d')).year)

test_set['Mes'] = test_set['date'].map(lambda x: (datetime.strptime(x, '%Y-%M-%d')).month)

test_set['Dia'] = test_set['date'].map(lambda x: (datetime.strptime(x, '%Y-%M-%d')).day)

print("Done!")

test_set['onpromotion']=(test_set.isnull()['onpromotion'])=0

test_set.loc[test_set['onpromotion'] ==False, 'onpromotion'] = 0

test_set.loc[test_set['onpromotion'] ==True, 'onpromotion'] = 1

print("Done!")

te1=test_set.merge(items_encoded,how='left', left_on=['item_nbr'], right_on = ['item_nbr'])

te2=te1.merge(stores_encoded,how='left', left_on=['store_nbr'], right_on = ['store_nbr'])

test_df=pd.DataFrame(te2)

X_test = test_df.loc[:,(test_df.columns != 'id') & (test_df.columns != 'unit_sales') & (test_df.columns != 'date') ]

del te1

del te2

del test_set

del items_encoded

del stores_encoded

print("Done!")

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

y_test_pred=model.predict(X_test)  

y_test_df=pd.DataFrame(y_test_pred,columns={'unit_sales'})

y_test_df[y_test_df.unit_sales<0]=0

output_set=pd.concat( (test_df,y_test_df),axis=1)[['id','date','store_nbr','item_nbr','onpromotion','unit_sales']]

output_seta=output_set.groupby(['id']).agg({'unit_sales':'sum'})

output_seta.to_csv("output.csv")

del output_set

del output_seta

print('done!')

print("--- %s seconds ---" % (time.time() - start_time))
#### THE END
print(check_output(["ls", "../working"]).decode("utf8"))

print("--- %s seconds ---" % (time.time() - start_time_all))