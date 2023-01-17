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
def label_encoder(df):
    assert sum(df.dtypes == 'object') >= 1, 'dataframe not have a values of type object'
    #se saca la lista de columnas de tipo object
    list_columns_object = df.dtypes[df.dtypes == 'object'].index
    for index in list_columns_object:
        #se cambian los tipos a category para poder aplicar el .cat.codes de pandas
        df.loc[:,index] = df.loc[:,index].astype('category').cat.codes
        
    return df
data_train = pd.read_csv('../input/application_train.csv')
data_test = pd.read_csv('../input/application_test.csv')
print ('% de ceros: ', np.mean(data_train.TARGET==0))
print ('% de unos: ', np.mean(data_train.TARGET==1))
data_train_1 = data_train.loc[data_train.loc[data_train.index,'TARGET']==1]
data_train_0 = data_train.loc[data_train.loc[data_train.index,'TARGET']==0]

data_train_1.shape, data_train_0.shape
index_0 = np.random.choice(data_train_0.index,data_train_1.shape[0],False)
data_train_0 = data_train_0.loc[index_0]
data_train_1.shape, data_train_0.shape
data_train = pd.concat((data_train_0,data_train_1)).sort_index()
data_train.shape, data_test.shape
data_train = data_train.set_index(data_train.SK_ID_CURR)
data_test = data_test.set_index(data_test.SK_ID_CURR)

data_train = data_train.drop('SK_ID_CURR',axis=1)
data_test = data_test.drop('SK_ID_CURR',axis=1)
pre_app = pd.read_csv('../input/previous_application.csv')
pre_app = pre_app.set_index(pre_app.SK_ID_CURR.values)
pre_app = pre_app.drop('SK_ID_CURR',axis=1)
pre_app = label_encoder(pre_app)
pre_app = pre_app.groupby(pre_app.index).mean()
pre_app = pre_app.drop('SK_ID_PREV',axis=1)
data_train_merge = pd.merge(data_train, pre_app, left_index=True, right_index=True, how='left')
data_test_merge = pd.merge(data_test, pre_app, left_index=True, right_index=True, how='left')

data_train_merge.shape, data_test_merge.shape
bureau = pd.read_csv('../input/bureau.csv')
bureau.shape
bureau = bureau.set_index(bureau.columns[0])
bureau = bureau.drop('SK_ID_BUREAU', axis=1)
bureau.head()
bureau = label_encoder(bureau)
bureau = bureau.groupby(bureau.index).mean()
bureau.shape
data_train_merge = pd.merge(data_train_merge, bureau, left_index=True, right_index=True, how='left')
data_test_merge = pd.merge(data_test_merge, bureau, left_index=True, right_index=True, how='left')

data_train_merge.shape, data_test_merge.shape
credict_cb = pd.read_csv('../input/credit_card_balance.csv')
credict_cb = credict_cb.set_index(credict_cb.columns[1])
credcit_cb = credict_cb.drop('SK_ID_PREV',axis=1)
print(credict_cb.shape)
credict_cb = label_encoder(credict_cb)
credict_cb = credict_cb.groupby(credict_cb.index).mean()
credict_cb.shape
data_train_merge = pd.merge(data_train_merge, credict_cb, left_index=True, right_index=True, how='left')
data_test_merge = pd.merge(data_test_merge, credict_cb, left_index=True, right_index=True, how='left')

data_train_merge.shape, data_test_merge.shape
posh_cb = pd.read_csv('../input/POS_CASH_balance.csv')
posh_cb = posh_cb.set_index(posh_cb.columns[1])
posh_cb = posh_cb.drop('SK_ID_PREV',axis=1)
posh_cb.shape
posh_cb = label_encoder(posh_cb)
posh_cb = posh_cb.groupby(posh_cb.index).mean()
posh_cb.shape
data_train_merge = pd.merge(data_train_merge, posh_cb, left_index=True, right_index=True, how='left')
data_test_merge = pd.merge(data_test_merge, posh_cb, left_index=True, right_index=True, how='left')

data_train_merge.shape, data_test_merge.shape
#lo mismo para el data_train
col_mean_train = (data_train_merge.loc[:,data_train_merge.count()<data_train_merge.shape[0]].min())
dict_col_mean_train = {}
for i in range(len(col_mean_train)):
    dict_col_mean_train[col_mean_train.index[i]] = col_mean_train[i]
    
#lo mismo para el data_test
col_mean_test = (data_test_merge.loc[:,data_test_merge.count()<data_test_merge.shape[0]].min())
dict_col_mean_test = {}
for i in range(len(col_mean_test)):
    dict_col_mean_test[col_mean_test.index[i]] = col_mean_test[i]
col_mean_train
data_train_merge_t = data_train_merge.fillna(value=dict_col_mean_train)
data_test_merge_t = data_test_merge.fillna(value=dict_col_mean_test)
X_train = data_train_merge_t[[col for col in data_train_merge_t.columns if col!='TARGET']]
y_train = data_train_merge_t.TARGET

X_test = data_test_merge_t

X_train.shape, y_train.shape, X_test.shape
X_train = label_encoder(X_train)
X_test = label_encoder(X_test)
X_train.head()
y_train.head()
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(learning_rate=.5, n_estimators=150, verbose=1)
GBC.fit(X_train, y_train)
GBC.score(X_train, y_train)
predict = GBC.predict_proba(X_test)[:,1]
predict
ls -l
#sin correlación de datos
result = pd.DataFrame(data=predict, index=data_test_merge.index, columns=['TARGET'])
result.index.name = 'SK_ID_CURR'
result.to_csv('../working/submission_test_1.csv')
result.head()

