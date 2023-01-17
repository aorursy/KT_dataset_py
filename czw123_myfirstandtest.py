import numpy as np

import pandas as pd

from subprocess import check_output

#types={'Semana':np.uint8,'Agencia_ID':np.uint16,'Canal_ID':np.uint8,

 #      'Ruta_SAK':np.uint16,'Cliente_ID':np.uint32,'Producto_ID':np.uint16,

#       'Demanda_uni_equil':np.uint32}

types = {'Semana':np.uint8, 'Agencia_ID':np.uint16, 'Canal_ID':np.uint8,

         'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 'Producto_ID':np.uint16,

         'Demanda_uni_equil':np.uint32}

#train=pd.read_csv('../input/train.csv',usecols=types.keys(),dtype=types)

train=pd.read_csv('../input/train.csv',usecols=types.keys(),dtype=types,nrows=1000)

print(train.dtype)

print(train.info(memery_usage=True))



import numpy as np

import pandas as pd

from subprocess import check_output

#types={'Semana':np.uint8,'Agencia_ID':np.uint16,'Canal_ID':np.uint8,

 #      'Ruta_SAK':np.uint16,'Cliente_ID':np.uint32,'Producto_ID':np.uint16,

#       'Demanda_uni_equil':np.uint32}

types = {'Semana':np.uint8, 'Agencia_ID':np.uint16, 'Canal_ID':np.uint8,

         'Ruta_SAK':np.uint16, 'Cliente_ID':np.uint32, 'Producto_ID':np.uint16,

         'Demanda_uni_equil':np.uint32}

#train=pd.read_csv('../input/train.csv',usecols=types.keys(),dtypes=types)

train=pd.read_csv('../input/train.csv',usecols=types.keys(),dtypes=types,nrows=1000)

print(train.dtype)

print(train.info(memery_usage=True))
from subprocess import check_output

import pandas as pd

print(check_output(['ls','.']).decode('utf8'))

submission=pd.read_csv('../input/sample_submission.csv')

print(submission.shape)

print(submission.columns)

print(submission.head(20))
import numpy as np

import pandas as pd

import gc

import xgboost as xgb

import math

from sklearn.cross_validation import train_test_split

from ml_metrics import rmsle

def evalerror(preds, dtrain):



    labels = dtrain.get_label()

    assert len(preds) == len(labels)

    labels = labels.tolist()

    preds = preds.tolist()

    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]

    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5



nrows=10000

train=pd.read_csv('../input/train.csv',nrows=nrows)

test=pd.read_csv('../input/test.csv',nrows=nrows)



print(train.columns)

print(test.columns)

print(test.columns.values)

ids=test['id']

test=test.drop(['id'],axis=1)

y=train['Demanda_uni_equil']

X=train[test.columns.values]



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1000)







params = {}

params['objective'] = "reg:linear"

params['eta'] = 0.05

params['max_depth'] = 5

params['subsample'] = 0.8

params['colsample_bytree'] = 0.6

params['silent'] = True



print ('')



test_preds = np.zeros(test.shape[0])

xg_train = xgb.DMatrix(X_train, label=y_train)

xg_test = xgb.DMatrix(X_test)



watchlist = [(xg_train, 'train')]

num_rounds = 100



xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)

preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)



print ('RMSLE Score:', rmsle(y_test, preds))



fxg_test = xgb.DMatrix(test)

fold_preds = np.around(xgclassifier.predict(fxg_test, ntree_limit=xgclassifier.best_iteration), decimals = 1)

test_preds += fold_preds



submission = pd.DataFrame({'id':ids, 'Demanda_uni_equil': test_preds})

submission.to_csv('submission.csv', index=False)
import numpy as np

import pandas as pd

import os

import gc

nrows=100000

train=pd.read_csv('../input/train.csv',nrows=nrows)

print(train.shape)

print(train.columns)

data=train.copy()

data['target']=data['Demanda_uni_equil']

data.drop(['Demanda_uni_equil'],axis=1,inplace=True)



nCliente_ID = pd.DataFrame(pd.groupby(data,['Cliente_ID','Semana'])['target'].count())

print(nCliente_ID.shape)

print(nCliente_ID.columns)

print(nCliente_ID.head(2))

nCliente_ID = nCliente_ID.reset_index()

print(nCliente_ID.shape)

print(nCliente_ID.columns)

print(nCliente_ID.head(2))

nCliente_ID.rename(columns={'target': 'nCliente_ID'}, inplace=True)

print(nCliente_ID.shape)

print(nCliente_ID.columns)

print(nCliente_ID.head(2))

nCliente_ID = pd.DataFrame(pd.groupby(nCliente_ID,['Cliente_ID'])['nCliente_ID'].mean())

print(nCliente_ID.shape)

print(nCliente_ID.columns)

print(nCliente_ID.head(2))

nCliente_ID = nCliente_ID.reset_index()

print(nCliente_ID.shape)

print(nCliente_ID.columns)

print(nCliente_ID.head(2))

 



data = pd.merge(data, nCliente_ID, 

                            how='left',

                            left_on=['Cliente_ID'], 

                            right_on=['Cliente_ID'],

                            left_index=False, right_index=False, sort=True,

                            suffixes=('_x', '_y'), copy=False) 

print(data.columns)

print(data.head(50))



del nCliente_ID

gc.collect()

print('merge completo nCliente_ID')

print(data.shape[0])
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import datetime

import time

from collections import defaultdict

import gc



def run_solution():

    print('Preparing arrays...')

    f = open("../input/train.csv", "r")

    f.readline()

    total = 0



    client_product_arr = defaultdict(int)

    client_product_arr_count = defaultdict(int)

    client_arr = defaultdict(int)

    client_arr_count = defaultdict(int)

    product_arr = defaultdict(int)

    product_arr_count = defaultdict(int)



    # Calc counts

    avg_target = 0.0

    while 1:

        line = f.readline().strip()

        total += 1



        if total % 10000000 == 0:

            print('Read {} lines...'.format(total))



        if line == '':

            break



        arr = line.split(",")

        week = int(arr[0])

        agency = arr[1]

        canal_id = arr[2]

        ruta_sak = arr[3]

        cliente_id = int(arr[4])

        producto_id = int(arr[5])

        vuh = arr[6]

        vh = arr[7]

        dup = arr[8]

        dp = arr[9]

        target = int(arr[10])

        avg_target += target



        client_product_arr[(cliente_id, producto_id)] += target

        client_product_arr_count[(cliente_id, producto_id)] += 1

        client_arr[cliente_id] += target

        client_arr_count[cliente_id] += 1

        product_arr[producto_id] += target

        product_arr_count[producto_id] += 1



    f.close()

    avg_target /= total

    print('Average target: ', avg_target)

    gc.collect()

    

    print('Generate submission...')

    now = datetime.datetime.now()

    path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

    out = open(path, "w")

    f = open("../input/test.csv", "r")

    f.readline()

    total = 0

    out.write("id,Demanda_uni_equil\n")



    index_both = 0

    index_client = 0

    index_product = 0

    index_empty = 0



    while 1:

        line = f.readline().strip()

        total += 1



        if total % 10000000 == 0:

            print('Write {} lines...'.format(total))



        if line == '':

            break



        arr = line.split(",")

        id = arr[0]

        week = int(arr[1])

        agency = arr[2]

        canal_id = arr[3]

        ruta_sak = arr[4]

        cliente_id = int(arr[5])

        producto_id = int(arr[6])



        out.write(str(id) + ',')

        if (cliente_id, producto_id) in client_product_arr:

            val = client_product_arr[(cliente_id, producto_id)]/client_product_arr_count[(cliente_id, producto_id)]

            out.write(str(val))

            index_both += 1

        elif cliente_id in client_arr:

            val = client_arr[cliente_id]/client_arr_count[cliente_id]

            out.write(str(val))

            index_client += 1

        elif producto_id in product_arr:

            val = product_arr[producto_id]/product_arr_count[producto_id]

            out.write(str(val))

            index_product += 1

        else:

            out.write(str(avg_target))

            index_empty += 1

        out.write("\n")



    print('Both: {}'.format(index_both))

    print('Client: {}'.format(index_client))

    print('Product: {}'.format(index_product))

    print('Empty: {}'.format(index_empty))



    out.close()

    f.close()



start_time = time.time()

#run_solution()

print("Elapsed time overall: %s seconds" % (time.time() - start_time))
print(check_output(["ls", "."]).decode("utf8"))