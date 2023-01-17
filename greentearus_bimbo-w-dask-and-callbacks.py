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
import dask.dataframe as dd

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
'''

import pandas as pd

cliente_tabla = pd.read_csv("../input/grupo-bimbo-inventory-demand/cliente_tabla.csv")

producto_tabla = pd.read_csv("../input/grupo-bimbo-inventory-demand/producto_tabla.csv")

sample_submission = pd.read_csv("../input/grupo-bimbo-inventory-demand/sample_submission.csv")

test = pd.read_csv("../input/grupo-bimbo-inventory-demand/test.csv")

town_state = pd.read_csv("../input/grupo-bimbo-inventory-demand/town_state.csv")

train = pd.read_csv("../input/grupo-bimbo-inventory-demand/train.csv")

'''
FULL_train = dd.read_csv('../input/grupo-bimbo-inventory-demand/train.csv')

FULL_test = dd.read_csv('../input/grupo-bimbo-inventory-demand/test.csv')
#+assumption: same cliend can order same product on from the same agency on the same week by different routes = > we assume it's the same amount regardless of route. 

#so we drop duplicates irregardless of wether they are meaningful or erroneous
import dask

dask.config.set(scheduler='threads')
TRAIN_Semana_3 = FULL_train.loc[FULL_train['Semana'] == 3, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])

TRAIN_Semana_4 = FULL_train.loc[FULL_train['Semana'] == 4, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])

TRAIN_Semana_5 = FULL_train.loc[FULL_train['Semana'] == 5, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])

TRAIN_Semana_6 = FULL_train.loc[FULL_train['Semana'] == 6, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])

TRAIN_Semana_7 = FULL_train.loc[FULL_train['Semana'] == 7, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])

TRAIN_Semana_8 = FULL_train.loc[FULL_train['Semana'] == 8, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])

TRAIN_Semana_9 = FULL_train.loc[FULL_train['Semana'] == 9, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])
TEST_Semana_10 = FULL_test.loc[FULL_test['Semana'] == 10, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Semana']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])

TEST_Semana_11 = FULL_test.loc[FULL_test['Semana'] == 11, ['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Semana']].drop_duplicates(subset=['Agencia_ID', 'Producto_ID', 'Cliente_ID'])
PRED_Semana_10 = TEST_Semana_10.merge(TRAIN_Semana_3, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_10 = PRED_Semana_10.rename(columns={'Demanda_uni_equil': 'D_3'})

PRED_Semana_10 = PRED_Semana_10.persist()
print(len(TEST_Semana_10))

print(len(PRED_Semana_10)) #==len(tester) or len(TEST_Semana_10) True!
PRED_Semana_10 = PRED_Semana_10.merge(TRAIN_Semana_4, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_10 = PRED_Semana_10.rename(columns={'Demanda_uni_equil': 'D_4'})

PRED_Semana_10 = PRED_Semana_10.persist()
PRED_Semana_10 = PRED_Semana_10.merge(TRAIN_Semana_5, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_10 = PRED_Semana_10.rename(columns={'Demanda_uni_equil': 'D_5'})

PRED_Semana_10 = PRED_Semana_10.persist()
PRED_Semana_10 = PRED_Semana_10.merge(TRAIN_Semana_6, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_10 = PRED_Semana_10.rename(columns={'Demanda_uni_equil': 'D_6'})

PRED_Semana_10 = PRED_Semana_10.persist()
PRED_Semana_10 = PRED_Semana_10.merge(TRAIN_Semana_7, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_10 = PRED_Semana_10.rename(columns={'Demanda_uni_equil': 'D_7'})

PRED_Semana_10 = PRED_Semana_10.persist()
PRED_Semana_10 = PRED_Semana_10.merge(TRAIN_Semana_8, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_10 = PRED_Semana_10.rename(columns={'Demanda_uni_equil': 'D_8'})

PRED_Semana_10 = PRED_Semana_10.persist()
PRED_Semana_10 = PRED_Semana_10.merge(TRAIN_Semana_9, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_10 = PRED_Semana_10.rename(columns={'Demanda_uni_equil': 'D_9'})

PRED_Semana_10 = PRED_Semana_10.persist()
print(len(TEST_Semana_10))

print(len(PRED_Semana_10)) 
def predict(df, cols, result):

    

    #put all the shit into return statement otheriwise it

    #doesn't work properly(recursively does each statement several

    #times before proceeding to the ext or i dunno some shit)

    return df.assign(result=df[cols].mean(axis=1).fillna(7).astype(int)) #we use round(0) vs .astype(int) coz we wanna preserve NaN information << revisited, non-relevant now





cols = ['D_3', 'D_4', 'D_5', 'D_6', 'D_7', 'D_8', 'D_9']

#cols = ['D_3', 'D_4']

result = 'D_10'

#result = 'M_34'

PRED_Semana_10_DONE = PRED_Semana_10.map_partitions(predict, cols, result)

PRED_Semana_10_DONE = PRED_Semana_10_DONE.rename(columns={'result': result})

PRED_Semana_10_DONE
print(len(TEST_Semana_10))

print(len(PRED_Semana_10)) 

print(len(PRED_Semana_10_DONE)) 
PRED_Semana_11 = TEST_Semana_11.merge(PRED_Semana_10_DONE.drop(['Semana'], axis=1), how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID'], suffixes=('_1', '+1') )

PRED_Semana_11 = PRED_Semana_11.persist()
print(len(TEST_Semana_11))

print(len(PRED_Semana_11)) 
cols = ['D_3', 'D_4', 'D_5', 'D_6', 'D_7', 'D_8', 'D_9', 'D_10']

result = 'D_11'

PRED_Semana_11_DONE = PRED_Semana_11.map_partitions(predict, cols, result)

PRED_Semana_11_DONE = PRED_Semana_11_DONE.rename(columns={'result': result})

PRED_Semana_11_DONE
print(len(TEST_Semana_11))

print(len(PRED_Semana_11)) 

print(len(PRED_Semana_11_DONE)) 
CUT_PRED_Semana_10_DONE = PRED_Semana_10_DONE[['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Semana', 'D_10']]

CUT_PRED_Semana_10_DONE = CUT_PRED_Semana_10_DONE.rename(columns={'D_10': 'Demanda_uni_equil'})



CUT_PRED_Semana_11_DONE = PRED_Semana_11_DONE[['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Semana', 'D_11']]

CUT_PRED_Semana_11_DONE = CUT_PRED_Semana_11_DONE.rename(columns={'D_11': 'Demanda_uni_equil'})



FINAL_CUT_PRED = CUT_PRED_Semana_10_DONE.append(CUT_PRED_Semana_11_DONE)

                                              

#FULL_pred = FULL_test.merge(FINAL_CUT_PRED, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Semana'], suffixes=('_1', '+1') )



#FULL_pred
Base = FULL_test

Base.tail()
FULL_pred = Base.merge(FINAL_CUT_PRED, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Semana'], suffixes=('_1', '+1') )

#FULL_pred = Base.join(FINAL_CUT_PRED, how='left', on=['Agencia_ID', 'Producto_ID', 'Cliente_ID', 'Semana'])

FULL_pred
#FULL_pred.set_index('index')
#len(FULL_pred.dropna()) #.isnull().plot()
print(len(FULL_test))

print(len(FULL_pred))
FINAL_pred = FULL_pred[['id', 'Demanda_uni_equil']] #.persist() pls don't

#FINAL_pred = FULL_pred[['Demanda_uni_equil']] #.persist() pls don't

FINAL_pred
print(len(FULL_test))

print(len(FULL_pred))

print(len(FINAL_pred))
FULL_test.head()
panda = FINAL_pred.compute()
panda
panda = panda.reset_index()
panda
panda = panda.sort_values(by='id')
panda
panda.to_csv('submission_integral_indexed.csv', index=False)