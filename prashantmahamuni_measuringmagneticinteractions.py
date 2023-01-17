import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import gc

import os



import matplotlib.pyplot as plt

import seaborn as sns



import lightgbm as lgb

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

print(os.listdir("../input"))
train_original = pd.read_csv("../input/train.csv")

structures_original = pd.read_csv("../input/structures.csv")

test_original = pd.read_csv("../input/test.csv")
train_original.head()
structures_original.head()
test_original.head()
structures_original[structures_original['molecule_name'] == 'dsgdb9nsd_000015']
moleculeCount = structures_original.groupby(by=['molecule_name','atom'])[['atom']].count()

moleculeCount.rename(columns={'atom':'count'},inplace = True)

moleculeCount = moleculeCount.unstack(fill_value=0)

moleculeCount = moleculeCount['count'].reset_index()



moleculeCount.head()
moleculeCount[moleculeCount['molecule_name'] == 'dsgdb9nsd_000015']
structures = pd.DataFrame.merge(structures_original,moleculeCount

                               ,how='inner'

                               ,left_on = ['molecule_name'] 

                               ,right_on = ['molecule_name']

                              )



structures.head()
tmp_merge = pd.DataFrame.merge(train_original,structures

                               ,how='left'

                               ,left_on = ['molecule_name','atom_index_0'] 

                               ,right_on = ['molecule_name','atom_index']

                              )



tmp_merge = tmp_merge.merge(structures

                ,how='left'

                ,left_on = ['molecule_name','atom_index_1'] 

                ,right_on = ['molecule_name','atom_index']

               )



tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)

tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' , 'scalar_coupling_constant' , 

                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']



train = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,

           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O', 'scalar_coupling_constant']]

train.sort_values(by=['id','molecule_name'],inplace=True)

train.reset_index(inplace=True,drop=True)



tmp_merge = None



train.head()
tmp_merge = pd.DataFrame.merge(test_original,structures

                               ,how='inner'

                               ,left_on = ['molecule_name','atom_index_0'] 

                               ,right_on = ['molecule_name','atom_index']

                              )

tmp_merge = tmp_merge.merge(structures

                ,how='inner'

                ,left_on = ['molecule_name','atom_index_1'] 

                ,right_on = ['molecule_name','atom_index']

               )



tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)

tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' ,  

                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']





test = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,

           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1', 'C','F','H','N','O']]





test.sort_values(by=['id','molecule_name'],inplace=True)

test.reset_index(inplace=True,drop=True)



tmp_merge = None



test.head()


train_original = None

del train_original

structures_original = None

del structures_original

test_original = None

del test_original

structures = None

del structures

gc.collect()
train['dist'] = np.linalg.norm(train[['x_0', 'y_0', 'z_0']].values - train[['x_1', 'y_1', 'z_1']].values, axis=1)

test['dist'] = np.linalg.norm(test[['x_0', 'y_0', 'z_0']].values - test[['x_1', 'y_1', 'z_1']].values, axis=1)



train.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)

test.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)
train['type'] = pd.Categorical(train['type'])

train['atom_nm_1'] = pd.Categorical(train['atom_nm_1'])

test['type'] = pd.Categorical(test['type'])

test['atom_nm_1'] = pd.Categorical(test['atom_nm_1'])
train.head()
test.head()
X = train[['atom_0' ,  'atom_1' , 'type', 'atom_nm_1', 'C' ,  'F' ,  'H' ,  'N' ,  'O' , 'dist' ]]



y = train['scalar_coupling_constant']
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.4, random_state=420)
lgb_train = lgb.Dataset(X_train,y_train,free_raw_data=True)

lgb_eval = lgb.Dataset(X_test,y_test,free_raw_data=True)
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'learning_rate': 0.05,

    'num_leaves': 50, 

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 0,

    'num_boost_round':5000,

    'reg_alpha': 0.1,

    'reg_lambda': 0.3,

'early_stopping_rounds':5

         }
gbm = lgb.train(

    params,

    lgb_train,



    valid_sets=lgb_eval

)
y_predict = gbm.predict(X_test)

mse = np.sqrt(metrics.mean_squared_error(y_predict,y_test))



print('Mean Squared Error is : '+str(mse))
submission_df = pd.DataFrame(columns=['id', 'scalar_coupling_constant'])

submission_df['id'] = test['id']

submission_df['scalar_coupling_constant'] = gbm.predict(test[['atom_0' ,  'atom_1' , 'type', 'atom_nm_1', 'C' ,  'F' ,  'H' ,  'N' ,  'O' , 'dist' ]])

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)