import numpy as np

import pandas as pd

import os



os.listdir("../input")



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
def merge_train_data():    

    print( "Loading Train Data...")

    train_transaction = pd.read_csv('../input/train_transaction.csv')

    train_identity = pd.read_csv('../input/train_identity.csv')

    

    print( "Shape of train_transaction" , train_transaction.shape )

    train_transaction.head()

    

    print("Shape of train_identity" , train_identity.shape )

    train_identity.head()

    

    print( "Merging..." )

    train_merged = pd.merge( train_transaction , train_identity , on='TransactionID' , how='left' )

    

    del train_transaction 

    del train_identity

    

    return train_merged
train_merged = merge_train_data()

train_merged.shape

train_merged.head()
chk_NaN = train_merged.isnull().sum()

chk_NaN
def merge_test_data():

    print( "Loading Test Data...")

    test_transaction = pd.read_csv('../input/test_transaction.csv')

    test_identity = pd.read_csv('../input/test_identity.csv')

    

    print( "Shape of test_transaction" , test_transaction.shape )

    test_transaction.head()

    

    print("Shape of test_identity" , test_identity.shape )

    test_identity.head()

    

    print( "Merging..." )

    test_merged = pd.merge( test_transaction , test_identity , on='TransactionID' , how='left' )



    del test_transaction

    del test_identity

    return test_merged
test_merged = merge_test_data()

test_merged.shape

test_merged.head()
chk_NaN = test_merged.isnull().sum()

chk_NaN
%matplotlib inline



import seaborn as sns

import matplotlib.pyplot as plt



sns.countplot( x = 'isFraud', data = train_merged )

plt.xticks(rotation=45)

plt.title( 'Target' )

plt.show()
target = train_merged['isFraud']

train_merged = train_merged.drop('isFraud' , axis='columns')
train_merged.shape
rows_count = train_merged.shape[0]



missing_value = pd.DataFrame(columns=['Missing Rate'])

missing_value['Missing Rate'] = train_merged.isnull().sum() / rows_count



high_miss_rate = missing_value[ missing_value['Missing Rate'] >= 0.85 ].index.values.tolist()



print("High Missing Rate(85%) Feature Count : ",len(high_miss_rate))



train_merged = train_merged.drop(high_miss_rate , axis='columns')

train_merged.shape

del high_miss_rate
cols_int = []

cols_float = []

cols_object = []



all_cols_list = train_merged.columns.values.tolist()



for col in all_cols_list:

    if train_merged[col].dtypes == 'int64':

        cols_int.append(col)

    elif train_merged[col].dtypes == 'float64':

        cols_float.append(col)

    elif train_merged[col].dtypes == 'object':

        cols_object.append(col)

    else:

        print('Exception')



print( 'Total train_merged Feature Numbers : ', len(all_cols_list))

print( 'int64 Feature Numbers : ', len(cols_int))

print( 'float64 Feature Numbers : ', len(cols_float))

print( 'object Feature Numbers : ', len(cols_object))
import xgboost as xgb

from xgboost import XGBClassifier
# fit model no training data

model = XGBClassifier( nthread = 4 )

model.fit( train_merged[cols_int + cols_float] , target)
xgb_fea_imp = pd.DataFrame(list(model.get_booster().get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)

xgb_fea_imp.shape

xgb_fea_imp.head(10)
import lightgbm as lgb



train_ds = lgb.Dataset( train_merged[ xgb_fea_imp['feature'].tolist() ] , label = target )



parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_threads' : 4,

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 1

}



model = lgb.train(parameters, train_ds ,num_boost_round=5000)

prob = model.predict( test_merged[ xgb_fea_imp['feature'].tolist() ] )

prob
submission = pd.DataFrame()

submission['TransactionID'] = test_merged['TransactionID']

submission['isFraud'] = prob
submission.head()