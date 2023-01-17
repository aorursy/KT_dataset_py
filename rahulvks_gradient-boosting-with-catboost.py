# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier, Pool

from catboost import CatBoostClassifier

import os

from catboost import cv

import shap

# load JS visualization code to notebook

shap.initjs()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")

train = pd.read_csv("../input/cat-in-the-dat/train.csv")

test = pd.read_csv("../input/cat-in-the-dat/test.csv")
##Data Prepare 



# dictionary to map the feature

bin_dict = {'T':1, 'F':0, 'Y':1, 'N':0}



# Maping the category values in our dict

train['bin_3'] = train['bin_3'].map(bin_dict)

train['bin_4'] = train['bin_4'].map(bin_dict)

test['bin_3'] = test['bin_3'].map(bin_dict)

test['bin_4'] = test['bin_4'].map(bin_dict)







dummies = pd.concat([train, test], axis=0, sort=False)

print(f'Shape before dummy transformation: {dummies.shape}')

dummies = pd.get_dummies(dummies, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],\

                          prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)

print(f'Shape after dummy transformation: {dummies.shape}')

train, test = dummies.iloc[:train.shape[0], :], dummies.iloc[train.shape[0]:, :]

del dummies

#print(train.shape)

#print(test.shape)





# Importing categorical options of pandas

from pandas.api.types import CategoricalDtype 



# seting the orders of our ordinal features

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 

                                     'Master', 'Grandmaster'], ordered=True)

ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',

                                     'Boiling Hot', 'Lava Hot'], ordered=True)

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

def transformingOrdinalFeatures(df, ord_1, ord_2, ord_3, ord_4):    

    df.ord_1 = df.ord_1.astype(ord_1)

    df.ord_2 = df.ord_2.astype(ord_2)

    df.ord_3 = df.ord_3.astype(ord_3)

    df.ord_4 = df.ord_4.astype(ord_4)



    # Geting the codes of ordinal categoy's 

    df.ord_1 = df.ord_1.cat.codes

    df.ord_2 = df.ord_2.cat.codes

    df.ord_3 = df.ord_3.cat.codes

    df.ord_4 = df.ord_4.cat.codes

    

    return df

train = transformingOrdinalFeatures(train, ord_1, ord_2, ord_3, ord_4)

test = transformingOrdinalFeatures(test, ord_1, ord_2, ord_3, ord_4)



#print(train.shape)

#print(test.shape)



def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



train = date_cyc_enc(train, 'day', 7)

test = date_cyc_enc(test, 'day', 7) 



train = date_cyc_enc(train, 'month', 12)

test = date_cyc_enc(test, 'month', 12)



#print(train.shape)

#print(test.shape)





import string



# Then encode 'ord_5' using ACSII values



# Option 1: Add up the indices of two letters in string.ascii_letters

train['ord_5_oe_add'] = train['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

test['ord_5_oe_add'] = test['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))



# Option 2: Join the indices of two letters in string.ascii_letters

train['ord_5_oe_join'] = train['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))

test['ord_5_oe_join'] = test['ord_5'].apply(lambda x:float(''.join(str(string.ascii_letters.find(letter)+1) for letter in x)))



# Option 3: Split 'ord_5' into two new columns using the indices of two letters in string.ascii_letters, separately

train['ord_5_oe1'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))

test['ord_5_oe1'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[0])+1))



train['ord_5_oe2'] = train['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))

test['ord_5_oe2'] = test['ord_5'].apply(lambda x:(string.ascii_letters.find(x[1])+1))



for col in ['ord_5_oe1', 'ord_5_oe2', 'ord_5_oe_add', 'ord_5_oe_join']:

    train[col]= train[col].astype('float64')

    test[col]= test[col].astype('float64')



#print(train.shape)

#print(test.shape)



from sklearn.preprocessing import LabelEncoder



high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



for col in high_card_feats:    

    train[f'hash_{col}'] = train[col].apply( lambda x: hash(str(x)) % 5000 )

    test[f'hash_{col}'] = test[col].apply( lambda x: hash(str(x)) % 5000 )

                                               

    if train[col].dtype == 'object' or test[col].dtype == 'object': 

        lbl = LabelEncoder()

        lbl.fit(list(train[col].values) + list(test[col].values))

        train[f'le_{col}'] = lbl.transform(list(train[col].values))

        test[f'le_{col}'] = lbl.transform(list(test[col].values))



#print(train.shape)

#print(test.shape)





col = high_card_feats + ['ord_5','day', 'month']

                    # + ['hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9']

                    # + ['le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9']

train.drop(col, axis=1, inplace=True)

test.drop(col, axis=1, inplace=True)



#print(train.shape)

#print(test.shape)

train['target'].value_counts()
train.head(2)
X_train = train.drop('target', axis=1).values

y_train = train['target'].values

X_test = test.values



X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.4, 

                                                                random_state=17,stratify=y_train)



SEED = 17



model = CatBoostClassifier (iterations=500,learning_rate=0.5, custom_loss=['AUC','Accuracy'],

                            early_stopping_rounds=100,eval_metric='AUC')



model.fit(X_train_part, y_train_part,

        eval_set=(X_valid, y_valid),

        use_best_model=False,

        verbose=20,

        plot=True,early_stopping_rounds=20);
print("Tree Count wiht early stopping ", (model.tree_count_))

train_pool = Pool(data=X_train,label=y_train)

validation_pool = Pool(data=X_valid, label=y_valid)



params = {'loss_function' : 'Logloss',

         'iterations': 100,

         'custom_loss': ['AUC','Accuracy'],

         'learning_rate': 0.5,

         }





cv_data = cv(params= params,

             pool= train_pool,

             shuffle=True,

             fold_count=5,

             partition_random_seed=0,

             verbose=False,

             plot=True,)
cv_data.head(10)
best_value = np.min(cv_data['test-Logloss-mean'])

best_iter = np.argmin(cv_data['test-Logloss-mean'])



print('Best validation score without stratified: {:.4f} {:.4f} on step {} '

      .format(best_value, cv_data['test-Logloss-std'][best_iter],best_iter))
cv_data = cv(params= params,

             pool= train_pool,

             shuffle=True,

             fold_count=5,

             stratified=True,

             partition_random_seed=0,

             verbose=False,

             plot=True)
best_value = np.min(cv_data['test-Logloss-mean'])

best_iter = np.argmin(cv_data['test-Logloss-mean'])



print('Best validation score with stratified: {:.4f} {:.4f} on step {} '

      .format(best_value, cv_data['test-Logloss-std'][best_iter],best_iter)) 
'''from sklearn.model_selection import GridSearchCV





#param_grid = {"learning_rate": [0.001,0.01,0.5],"depth":[3,1,2,6,4,5,7,8,9,10]}



param_grid = {"depth":[3,1,2,6,4,5,7,8,9,10],

          "iterations":[50],

          "learning_rate":[0.03,0.001,0.01,0.1,0.2,0.3], 

          "l2_leaf_reg":[3,1,5,10,100],

          "border_count":[32,5,10,20,50,100,200],

          #"ctr_border_count":[5,10,20,100,200],

          "thread_count":[4]}







clf  = CatBoostClassifier(

            verbose=False,eval_metric='AUC',early_stopping_rounds=100)



grid_search = GridSearchCV(clf , param_grid=param_grid, cv = 3)

results = grid_search.fit(X_train,y_train)

results.best_estimator_.get_params()'''
model_with_early_stop = CatBoostClassifier(eval_metric='AUC',

                                          iterations=100,learning_rate=0.5,

                                          early_stopping_rounds=20)



model_with_early_stop.fit(train_pool,eval_set = validation_pool,

                         verbose = False, plot=True)
print(model_with_early_stop.tree_count_)
model = CatBoostClassifier(iterations=200,learning_rate=0.01)

model.fit(train_pool,verbose=50)
print(model.predict(X_test))
#Probality of being 0 or 1

print(model.predict_proba(X_test))
raw_pred = model.predict(

    X_test, prediction_type='RawFormulaVal'

)
print(raw_pred)
metrics = model.eval_metrics(data= validation_pool,

                            metrics = ['Logloss','AUC'],

                            ntree_start = 0,

                            ntree_end = 0,

                            eval_period =1,

                            plot=True)
tunned_model = CatBoostClassifier(eval_metric='AUC',

iterations=500,

learning_rate=0.03,

depth=6,

l2_leaf_reg=3,

random_strength=1,

bagging_temperature=1)



tunned_model.fit(

train_pool,verbose=False, eval_set = validation_pool,plot=True)
model_valid_predict = tunned_model.predict_proba(X_valid)[:, 1]
roc_auc_score(y_valid, model_valid_predict)
cb_model = CatBoostClassifier(iterations=500,

                             learning_rate=0.05,

                             depth=10,

                             eval_metric='AUC',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             od_wait=20,use_best_model=True)

cb_model.fit(

train_pool,verbose=False, eval_set = validation_pool,plot=True)
model_valid_predict = cb_model.predict_proba(X_valid)[:, 1]
roc_auc_score(y_valid, model_valid_predict)
model_test_predict = cb_model.predict(test)
sub = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")

submission = pd.DataFrame({'id': sub["id"], 'target': model_test_predict})

submission.to_csv('submission_1.csv', index=False)