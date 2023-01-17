import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

gc.enable()

import time

import warnings

warnings.filterwarnings("ignore")
print(os.listdir("../input"))

!ls -GFlash  ../input
%%time

# import Dataset to play with it

train_identity= pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

train_transaction= pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

test_identity= pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')

sample_submissions = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

print ("Done!")




print('Shape of Data:')

print(train_transaction.shape)

print(test_transaction.shape)

print(train_identity.shape)

print(test_identity.shape)

print(sample_submissions.shape)







# Creat our train & test dataset

#%%time

tes = pd.merge(test_transaction , test_identity, how='left' , on = 'TransactionID')

test = pd.merge(sample_submissions , tes , how = 'left' , on = 'TransactionID')

train = pd.merge(train_transaction , train_identity, how='left', on = 'TransactionID')

del train_identity,train_transaction,test_identity, test_transaction , tes 
train.info()
test.info()
#Based on this great kernel https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            print("min for this col: ",mn)

            print("max for this col: ",mx)

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist




train, NAlist = reduce_mem_usage(train)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)



train.info()

train.shape
test, NAlist = reduce_mem_usage(test)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
test.info()

test.shape
test = test.drop(columns = ['id_15' , 'id_16' , 'id_23' , 'id_27' , 'id_28' , 'id_29','id_12' , 'id_30' , 'id_31' , 'id_33' , 'id_34','id_12','M5' , 'id_35' , 'id_36' , 'id_37' , 'id_38', 'DeviceType' , 'DeviceInfo'])
train = train.drop(columns = ['id_15' , 'id_16' , 'id_23' , 'id_27' , 'id_28' , 'id_29','id_12' , 'id_30' ,'M5', 'id_31' , 'id_33' , 'id_34' , 'id_35' , 'id_36' , 'id_37' , 'id_38', 'DeviceType' , 'DeviceInfo'])
#Applying Imputer on Test File 

from sklearn.preprocessing import Imputer



test['card6'] = test['card6'].fillna('debit')



test['card4'] = test['card4'].fillna('visa')



test['P_emaildomain'] = test['P_emaildomain'].fillna('gmail.com')



test['R_emaildomain'] = test['R_emaildomain'].fillna('gmail.com')



test['M1'] = test['M1'].fillna('T')



test['M2'] = test['M2'].fillna('T')



test['M3'] = test['M3'].fillna('T')



test['M4'] = test['M4'].fillna('M0')



test['M6'] = test['M6'].fillna('F')



test['M7'] = test['M7'].fillna('F')



test['M8'] = test['M8'].fillna('F')



test['M9'] = test['M9'].fillna('T')

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



X = test.iloc[:,4].values

test.iloc[:,4] = labelencoder.fit_transform(X.astype(str))



X = test.iloc[:,8].values

test.iloc[:,8] = labelencoder.fit_transform(X)



X = test.iloc[:,10].values

test.iloc[:,10] = labelencoder.fit_transform(X)



X = test.iloc[:,15].values

test.iloc[:,15] = labelencoder.fit_transform(X)



X = test.iloc[:,16].values

test.iloc[:,16] = labelencoder.fit_transform(X)



X = test.iloc[:,53].values

test.iloc[:,53] = labelencoder.fit_transform(X)



X = test.iloc[:,42].values

test.iloc[:,42] = labelencoder.fit_transform(X)



X = test.iloc[:,43].values

test.iloc[:,43] = labelencoder.fit_transform(X)



X = test.iloc[:,44].values

test.iloc[:,44] = labelencoder.fit_transform(X)



X = test.iloc[:,45].values

test.iloc[:,45] = labelencoder.fit_transform(X)



X = test.iloc[:,46].values

test.iloc[:,46] = labelencoder.fit_transform(X)



X = test.iloc[:,47].values

test.iloc[:,47] = labelencoder.fit_transform(X)



X = test.iloc[:,48].values

test.iloc[:,48] = labelencoder.fit_transform(X)



X = test.iloc[:,49].values

test.iloc[:,49] = labelencoder.fit_transform(X)



X = test.iloc[:,50].values

test.iloc[:,50] = labelencoder.fit_transform(X)



X = test.iloc[:,51].values

test.iloc[:,51] = labelencoder.fit_transform(X)



X = test.iloc[:,52].values

test.iloc[:,52] = labelencoder.fit_transform(X)
#Removing Nan values from train

train['card4'] = train['card4'].fillna('visa')



train['card6'] = train['card6'].fillna('debit')



train['P_emaildomain'] = train['P_emaildomain'].fillna('gmail.com')



train['R_emaildomain'] = train['R_emaildomain'].fillna('gmail.com')



train['M1'] = train['M1'].fillna('T')



train['M2'] = train['M2'].fillna('T')



train['M3'] = train['M3'].fillna('T')



train['M4'] = train['M4'].fillna('M0')



train['M6'] = train['M6'].fillna('F')



train['M7'] = train['M7'].fillna('F')



train['M8'] = train['M8'].fillna('F')



train['M9'] = train['M9'].fillna('T')



X = train.iloc[:,4].values

train.iloc[:,4] = labelencoder.fit_transform(X)



X = train.iloc[:,8].values

train.iloc[:,8] = labelencoder.fit_transform(X)



X = train.iloc[:,10].values

train.iloc[:,10] = labelencoder.fit_transform(X)



X = train.iloc[:,15].values

train.iloc[:,15] = labelencoder.fit_transform(X)



X = train.iloc[:,16].values

train.iloc[:,16] = labelencoder.fit_transform(X)



X = train.iloc[:,54].values

train.iloc[:,54] = labelencoder.fit_transform(X)



X = train.iloc[:,42].values

train.iloc[:,42] = labelencoder.fit_transform(X)



X = train.iloc[:,43].values

train.iloc[:,43] = labelencoder.fit_transform(X)



X = train.iloc[:,45].values

train.iloc[:,45] = labelencoder.fit_transform(X)



X = train.iloc[:,46].values

train.iloc[:,46] = labelencoder.fit_transform(X)



X = train.iloc[:,44].values

train.iloc[:,44] = labelencoder.fit_transform(X)



X = train.iloc[:,47].values

train.iloc[:,47] = labelencoder.fit_transform(X)



X = train.iloc[:,48].values

train.iloc[:,48] = labelencoder.fit_transform(X)



X = train.iloc[:,49].values

train.iloc[:,49] = labelencoder.fit_transform(X)



X = train.iloc[:,50].values

train.iloc[:,50] = labelencoder.fit_transform(X)



X = train.iloc[:,51].values

train.iloc[:,51] = labelencoder.fit_transform(X)



X = train.iloc[:,52].values

train.iloc[:,52] = labelencoder.fit_transform(X)



X = train.iloc[:,53].values

train.iloc[:,53] = labelencoder.fit_transform(X)
concat = [train , test]

fraud_det = pd.concat(concat)

fraud_det = fraud_det.reset_index()
X_train = fraud_det.iloc[:590540,3:418]

Y_train = fraud_det.iloc[:590540,2]

X_test = fraud_det.iloc[590540:,3:418]

Y_test = fraud_det.iloc[590540:,2]
from sklearn.preprocessing import StandardScaler

independent_scaler = StandardScaler()

X_train_norm = independent_scaler.fit_transform(X_train)

X_test_norm = independent_scaler.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components =2)

X_train_pca = pca.fit_transform(X_train_norm)

X_train_pca =pca.transform(X_train_norm)

X_test_pca = pca.fit_transform(X_test_norm)

X_test_pca =pca.transform(X_test_norm)

pca.explained_variance_ratio_
#Random Forest 

from sklearn.ensemble import RandomForestRegressor

rfc = RandomForestRegressor(n_estimators =5)

rfc.fit(X_train_pca , Y_train)

pred = rfc.predict(X_test_pca)

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(Y_test, pred)))

pred.reshape(-1,1)

sample_submissions['fraud'] = pred

#sample_submissions.head()

sample_submissions = sample_submissions.drop(columns = ['isFraud'])

#sample_submissions.head()

sample_submissions.rename(columns = {'fraud': 'isFraud'} , inplace = True)

rfpca = sample_submissions
rfpca.to_csv('randomforest_pca.csv', index=False)
rfpca.head()
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth = 1, random_state = 0)

dt.fit(X_train_pca, Y_train)

prediction = dt.predict(X_test_pca)

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(Y_test, prediction)))

prediction.reshape(-1,1)

sample_submissions['fraud'] = prediction

#sample_submissions.head()

sample_submissions = sample_submissions.drop(columns = ['isFraud'])

sample_submissions.rename(columns = {'fraud': 'isFraud'} , inplace = True)

dtpca = sample_submissions

#sample_submissions.head()

#dtpca

dtpca.head()
dtpca.to_csv('decisiontree_pca.csv', index=False)
