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
from sklearn import preprocessing

import xgboost as xgb

from sklearn.model_selection import KFold , GridSearchCV

from sklearn.metrics import roc_auc_score 

import gc

import matplotlib.pyplot as plt

import seaborn as sns
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID' ,nrows = 200000)

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID' )



train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
%%time

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print(train.shape)

print(test.shape)



%%time

Y_train = train['isFraud'].copy()

X_train = train.drop('isFraud' , axis = 1)

X_test = test.copy()

del train_transaction, train_identity, test_transaction, test_identity

del train , test

gc.collect()
print(X_train.shape)

print(X_test.shape)
for i in X_train.columns:

    if X_train[i].dtype == 'object' or X_test[i].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(X_train[i].values) + list(X_test[i].values))

        X_train[i] = lbl.transform(list(X_train[i].values))

        X_test[i] = lbl.transform(list(X_test[i].values))
for i in X_train.columns:

    if X_train[i].dtype == 'object' or X_test[i].dtype == 'object':

        print(i)
%%time 

X_train = reduce_mem_usage(X_train)

X_test = reduce_mem_usage(X_test)
# null_data = pd.DataFrame(X_train.isnull().sum()/X_train.shape[0]*100)

# null_data = pd.DataFrame()

# null_data = pd.concat([pd.DataFrame(X_train.isnull().sum()/X_train.shape[0]*100 ,columns=['train']) ,pd.DataFrame(X_test.isnull().sum()/X_test.shape[0]*100,columns=['test']) ] ,axis = 1).reset_index()

# null_data.head()
# columns_drop = null_data.sort_values(by = 'train'  , ascending = 0).head(100)['index'].values
train_corr = X_train.corr().abs()*100

train_corr = train_corr.where(np.triu(np.ones(train_corr.shape)).astype(np.bool))

train_corr.values[[np.arange(train_corr.shape[0])]*2] = np.nan

print(train_corr.shape)

# display(train_corr.tail(20))

counter =0

columns_drop =[]

train_corr_matrix = train_corr.values

for i in range(1 , train_corr.shape[0] ,1 ):

    for j in range(i , train_corr.shape[0] , 1):

        if train_corr_matrix[i][j] >= 98:

            counter+=1

            columns_drop.append(train_corr.columns[j])

            if counter%20 ==0:

                print('Comman Columns pair reached ... ' , counter)

print(' Total Common Pair Found .... ',counter)
# columns_drop = list(set(columns_drop))

# X_train.drop(columns = columns_drop , inplace = True)

# X_test.drop(columns = columns_drop , inplace = True)
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
Y_train.value_counts()
EPOCHS = 5

y_pred = np.zeros(sample_submission.shape[0])

y_oof = np.zeros(X_train.shape[0])

kf = KFold(n_splits = EPOCHS , shuffle = True)

for x_train_index , x_val_index in kf.split(X_train , Y_train):

    clf = xgb.XGBClassifier(

        n_estimators=500,

        max_depth=4,

        learning_rate=0.005,

        subsample=0.2,

        colsample_bytree = 0.2

    )

    x_tr , x_val = X_train.iloc[x_train_index , :] , X_train.iloc[x_val_index,:]

    y_tr , y_val = Y_train.iloc[x_train_index] , Y_train.iloc[x_val_index]

    clf.fit(x_tr,y_tr)

    y_pred_train = clf.predict_proba(x_val)[:,1]

    y_oof[x_val_index] = y_pred_train

    print('ROC AUC {}'.format(roc_auc_score(y_val, y_pred_train)))

    y_pred+= clf.predict_proba(X_test)[:,1] / EPOCHS



 
col = X_train.columns

importances = clf.feature_importances_

dataframe = pd.DataFrame({'col':col , 'importance':importances})

dataframe = dataframe.sort_values(by=['importance'] ,ascending = False)

dataframe['importance_ratio'] = dataframe['importance']/dataframe['importance'].max()*100

dataframe = dataframe.head(35)

dataframe['col'] = dataframe['col'].apply(lambda x:  str(x))

plt.figure(figsize=(18,12))

plt.barh(dataframe['col'], dataframe['importance_ratio'], color='orange' , align='center' ,linewidth =30 )

plt.yticks(rotation=30)

plt.show()
sample_submission['isFraud'] = y_pred

sample_submission.to_csv('second_simple_xgboost_Fraud_Baseline.csv')