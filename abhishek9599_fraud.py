import pandas as pd

import numpy as np
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

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

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_id = reduce_mem_usage(pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv'))

train_trn = reduce_mem_usage(pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv'))

test_id = reduce_mem_usage(pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv'))

test_trn = reduce_mem_usage(pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv'))
train = pd.merge(train_trn,train_id,on='TransactionID',how='left',sort=False)
test = pd.merge(test_trn,test_id,on='TransactionID',how='left',sort=False)
tar_train = train['isFraud']

train.drop('isFraud',inplace=True,axis=1)

data = pd.concat([train,test],axis =0,sort=False)
l = data.isna().sum()
n = []

for key,value in zip(data.columns,l):

    if (value/ data.shape[0] > 0.7):

        n.append(key)
data.drop(n,axis=1,inplace=True)
m = list(set(data.columns) - set(data._get_numeric_data().columns))
for i in m:

    data[i] = data[i].fillna(data[i].mode()[0])
data.fillna(data.median(),inplace=True)
train = data.iloc[:train.shape[0],:]

test = data.iloc[train.shape[0]:,:]
from category_encoders import WOEEncoder

woe = WOEEncoder(cols=list(train.columns)[1:])

train = woe.fit_transform(train.iloc[:,1:],tar_train)
TransactionID = test.iloc[:,0]
test = woe.transform(test.iloc[:,1:])
train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
data = reduce_mem_usage(data)
from imblearn.over_sampling import SMOTE 

sm = SMOTE(sampling_strategy = 'auto',random_state=42,n_jobs = -1)

train, tar_train = sm.fit_resample(train, tar_train)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train,tar_train,test_size=0.33,random_state=42)
#from sklearn.metrics import accuracy_score,confusion_matrix

#from sklearn.linear_model import S

#clf = LogisticRegression(n_jobs = -1)

#from xgboost import XGBClassifier

#clf =XGBClassifier(n_jobs=-1,n_estimators=150)

#from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

#clf = RandomForestClassifier(n_estimators=100,n_jobs=-1,criterion='gini')

#clf =GradientBoostingClassifier()

#clf.fit(x_train,y_train)
#y_pred = clf.predict(x_test)

#print(accuracy_score(y_test,y_pred))

#confusion_matrix(y_test,y_pred)
#from sklearn.model_selection import cross_val_score

#sc = cross_val_score(clf,x_train,y_train,cv=4,n_jobs=-1)
#sc.mean(),sc.std()
#clf.fit(train,tar_train)
train.shape,test.shape
#pred = clf.predict_proba(test)[:,1]
#df3 = pd.DataFrame({'TransactionID':TransactionID,'isFraud':pred})

#df3.to_csv('Submit.csv',index=False)
import lightgbm as lgb
lg_train = lgb.Dataset(x_train,y_train)

lg_eval = lgb.Dataset(x_test,y_test,reference=lg_train)
params = {

    "boosting_type": "gbdt",

    'metric': 'auc',

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'objective':'binary',

    'bagging_freq': 5,

    'verbose': 0

}
gbm = lgb.train(params,

                lg_train,

                valid_sets=lg_eval,

                early_stopping_rounds=5,

                num_boost_round=200)
pred_y = gbm.predict(test)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,pred_y)
#sc = lgb.cv(params,lg_train,200, 

 #               nfold = 10, 

  #              early_stopping_rounds = 25,

   #             stratified = True)
df3
df3 = pd.DataFrame({'TransactionID':TransactionID,'isFraud':pred_y})

df3.to_csv('Submit.csv',index=False)