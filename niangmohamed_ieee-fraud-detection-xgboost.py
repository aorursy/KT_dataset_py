import numpy as np # linear algebra

import pandas as pd # data processing, CSV file 



import datetime

import seaborn as sns

import matplotlib.pyplot as plt



import matplotlib.gridspec as gridspec

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GroupKFold

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.metrics import roc_auc_score



import xgboost as xgb



import warnings

warnings.filterwarnings("ignore")



import gc

gc.enable()



import os

os.chdir('/kaggle/input/ieeedatapreprocessing') # Set working directory

print(os.listdir('/kaggle/input/ieeedatapreprocessing'))
%%time

X_train = pd.read_pickle('train_df.pkl')

X_test = pd.read_pickle('test_df.pkl')

print ("Data is loaded!")
print('train_transaction shape is {}'.format(X_train.shape))

print('test_transaction shape is {}'.format(X_test.shape))
X_train.head()
X_test.head()
# NORMALIZE D COLUMNS

for i in [1,2,3,4,5,10,11,15]:

    if i in [1,2,3,5]: continue

    X_train['D'+str(i)] =  X_train['D'+str(i)] - X_train.TransactionDT/np.float32(24*60*60)

    X_test['D'+str(i)] = X_test['D'+str(i)] - X_test.TransactionDT/np.float32(24*60*60) 
# FREQUENCY ENCODE TOGETHER

def encode_FE(df1, df2, cols):

    for col in cols:

        df = pd.concat([df1[col],df2[col]])

        vc = df.value_counts(dropna=True, normalize=True).to_dict()

        vc[-1] = -1

        nm = col+'_FE'

        df1[nm] = df1[col].map(vc)

        df1[nm] = df1[nm].astype('float32')

        df2[nm] = df2[col].map(vc)

        df2[nm] = df2[nm].astype('float32')

        print(nm,', ',end='')

        

# LABEL ENCODE

def encode_LE(col,train=X_train,test=X_test,verbose=True):

    df_comb = pd.concat([train[col],test[col]],axis=0)

    df_comb,_ = df_comb.factorize(sort=True)

    nm = col

    if df_comb.max()>32000: 

        train[nm] = df_comb[:len(train)].astype('int32')

        test[nm] = df_comb[len(train):].astype('int32')

    else:

        train[nm] = df_comb[:len(train)].astype('int16')

        test[nm] = df_comb[len(train):].astype('int16')

    del df_comb; x=gc.collect()

    if verbose: print(nm,', ',end='')



# LABEL ENCODE 2

def encode_LE2(df1,df2,col,verbose=True):

    df_comb = pd.concat([df1[col],df2[col]],axis=0)

    df_comb,_ = df_comb.factorize()

    df1[col] = df_comb[:len(df1)].astype('int32')

    df2[col] = df_comb[len(df1):].astype('int32')

    if verbose: print(col,', ',end='')

        

# GROUP AGGREGATION MEAN AND STD

def encode_AG(main_columns, uids, aggregations=['mean'], train_df=X_train, test_df=X_test, 

              fillna=True, usena=False):

    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS

    for main_column in main_columns:  

        for col in uids:

            for agg_type in aggregations:

                new_col_name = main_column+'_'+col+'_'+agg_type

                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])

                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan

                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(

                                                        columns={agg_type: new_col_name})



                temp_df.index = list(temp_df[col])

                temp_df = temp_df[new_col_name].to_dict()   



                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')

                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')

                

                if fillna:

                    train_df[new_col_name].fillna(-1,inplace=True)

                    test_df[new_col_name].fillna(-1,inplace=True)

                

                print("'"+new_col_name+"'",', ',end='')

                

# COMBINE FEATURES

def encode_CB(col1,col2,df1=X_train,df2=X_test):

    nm = col1+'_'+col2

    df1[nm] = df1[col1].astype(str)+'_'+df1[col2].astype(str)

    df2[nm] = df2[col1].astype(str)+'_'+df2[col2].astype(str) 

    encode_LE(nm,verbose=False)

    print(nm,', ',end='')

    

# GROUP AGGREGATION NUNIQUE

def encode_AG2(main_columns, uids, train_df=X_train, test_df=X_test):

    for main_column in main_columns:  

        for col in uids:

            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)

            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()

            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')

            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')

            print(col+'_'+main_column+'_ct, ',end='')
# TRANSACTION AMT CENTS

X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')

X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
# FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN

encode_FE(X_train,X_test,['addr1','card1','card2','card3','P_emaildomain'])

# COMBINE COLUMNS CARD1+ADDR1, CARD1+ADDR1+P_EMAILDOMAIN

encode_CB('card1','addr1')

encode_CB('card1_addr1','P_emaildomain')

# FREQUENCY ENOCDE

encode_FE(X_train,X_test,['card1_addr1','card1_addr1_P_emaildomain'])

# GROUP AGGREGATE

encode_AG(['TransactionAmt','D10','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'],usena=True)
# ADD MONTH FEATURE

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

X_train['DT_M'] = X_train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

X_train['DT_M'] = (X_train['DT_M'].dt.year-2017)*12 + X_train['DT_M'].dt.month 



X_test['DT_M'] = X_test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))

X_test['DT_M'] = (X_test['DT_M'].dt.year-2017)*12 + X_test['DT_M'].dt.month 
# ADD UID FEATURE

X_train['day'] = X_train.TransactionDT / (24*60*60)

X_train['uid'] = X_train.card1_addr1.astype(str)+'_'+np.floor(X_train.day-X_train.D1).astype(str)



X_test['day'] = X_test.TransactionDT / (24*60*60)

X_test['uid'] = X_test.card1_addr1.astype(str)+'_'+np.floor(X_test.day-X_test.D1).astype(str)

# LABEL ENCODE

encode_LE2(X_train,X_test,'uid',verbose=False)
oof = np.zeros(len(X_train))

preds = np.zeros(len(X_test))
idxT = X_train.index[:4*len(X_train)//5]

idxV = X_train.index[4*len(X_train)//5:]
cols_to_drop = ["TransactionID", "isFraud", "TransactionDT"]

useful_cols = list(X_train.columns)



for col in cols_to_drop:

    while True:

        try:

            useful_cols.remove(col)

        except:

            break
print('NOW USING THE FOLLOWING',len(useful_cols),'FEATURES.')

np.array(useful_cols)
y_train = X_train['isFraud'].copy()
skf = GroupKFold(n_splits=6)



for i, (idxT, idxV) in enumerate( skf.split(X_train, y_train, groups=X_train['DT_M']) ):

    month = X_train.iloc[idxV]['DT_M'].iloc[0]

    

    print('Fold',i,'withholding month',month)

    print(' rows of train =',len(idxT),'rows of holdout =',len(idxV))

    

    xgboost_magic_classifier = xgb.XGBClassifier(

            n_estimators=15000,

            max_depth=20,

            learning_rate=0.02,

            subsample=0.8,

            eval_metric='auc',

            colsample_bytree=0.4,

            missing=-999,

            tree_method='gpu_hist' 

        )   

    

    xgboost_magic_classifier_fit = xgboost_magic_classifier.fit(X_train[useful_cols].iloc[idxT], y_train.iloc[idxT], 

            eval_set=[(X_train[useful_cols].iloc[idxV],y_train.iloc[idxV])],

            verbose=100, early_stopping_rounds=500)

    

    oof[idxV] += xgboost_magic_classifier.predict_proba(X_train[useful_cols].iloc[idxV])[:,1]

    preds += xgboost_magic_classifier.predict_proba(X_test[useful_cols])[:,1]/skf.n_splits

    

    del xgboost_magic_classifier_fit

    x = gc.collect()
print(confusion_matrix(y_train, oof.round()))
print(classification_report(y_train, oof.round()))
feature_imp = pd.DataFrame(sorted(zip(xgboost_magic_classifier.feature_importances_,useful_cols)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:50])

plt.title('XGBoost cross validation Most Important Features')

plt.tight_layout()

plt.show()
submission = pd.read_csv('sample_submission.csv', index_col='TransactionID')

submission.isFraud = preds

submission.head()
plt.hist(submission.isFraud,bins=100)

plt.ylim((0,5000))

plt.title('XGBoost cross validation submission')

plt.show()
submission.to_csv('/kaggle/working/xgboost_cv_submission.csv')