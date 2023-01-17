# data analysis libraries:

import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

import numpy as np, pandas as pd, os, gc

from sklearn.model_selection import GroupKFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline

import gc

import time

from contextlib import contextmanager
@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))
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

def encode_LE(col,train,test,verbose=False):

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

        

# GROUP AGGREGATION MEAN AND STD

# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda

def encode_AG(main_columns, uids, aggregations, train_df, test_df, 

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

def encode_CB(col1,col2,train,test):

    nm = col1+'_'+col2

    train[nm] = train[col1].astype(str)+'_'+train[col2].astype(str)

    test[nm] = test[col1].astype(str)+'_'+test[col2].astype(str) 

    encode_LE(nm,train,test)

# GROUP AGGREGATION NUNIQUE

def encode_AG2(main_columns, uids, train_df, test_df):

    for main_column in main_columns:  

        for col in uids:

            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)

            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()

            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')

            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')

            print(col+'_'+main_column+'_ct, ',end='')
%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

def reduce_mem_usage2(df):

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
def comb_mails(emails,us_emails,train,test):

    for c in ['P_emaildomain', 'R_emaildomain']:

                train[c + '_bin'] = train[c].map(emails)

                test[c + '_bin'] = test[c].map(emails)



                train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])

                test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])



                train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

                test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
def love():

    print('\n'.join([''.join([(' I_Love_Data_Science_'[(x-y) % len('I_Love_Data_Science_')] if ((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <= 0 else ' ') for x in range(-30, 30)]) for y in range(15, -15, -1)]))
def combine():

     with timer('Combining :'):   

        print('Combining Start...')

        # Read train and test data with pd.read_csv():

        train_id= pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")

        test_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

        train_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

        test_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

        train=pd.merge(train_tr, train_id, on = "TransactionID",how='left',left_index=True, right_index=True)

        test=pd.merge(test_tr, test_id, on = "TransactionID",how="left",left_index=True, right_index=True)

        del train_id, train_tr, test_id, test_tr

        test.columns=train.columns.drop("isFraud")

    

        

        



        return train,test
def pre_processing_and_feature_engineering():

    train,test=combine()   

    

    with timer('Preprocessing and Feature Engineering'):

        print('-' * 30)

        print('Preprocessing and Feature Engineering start...')

        print('-' * 10)

        print("After corelation test we can drop this columns")

        drop_col=[]

        train_colmns=train.loc[:,"V1":"V339"].columns

        test_columns=test.loc[:,"V2":"V339"].columns

        for col1,col2 in zip(train_colmns,test_columns):            

                if ((train.loc[:,col1:col2].corr().loc[col2].sum()-1)>0.75) & (train[col1].isna().sum()== train[col2].isna().sum()):

                    print("'"+col2+"'",', ',end='')

                    drop_col.append(col2)

        train=train.drop(drop_col,axis=1)

        test=test.drop(drop_col,axis=1)

        del drop_col

        print("-")

        print("We drop these columns as well because more than 90 percent of these columns are nan")

        for col in train.columns: 

                   if sum(train[col].isnull())/float(len(train.index)) > 0.90:

                    print("'"+col+"'",', ',end='')

                    train=train.drop(col,axis=1)

                    test=test.drop(col,axis=1)

                            

        emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 

              'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',

              'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',     

              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 

              'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',

              'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',

              'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 

              'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 

              'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',

              'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',

              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',

              'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 

              'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 

              'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 

              'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 

              'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 

              'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 

              'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',

              'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}



        us_emails = ['gmail', 'net', 'edu']

        comb_mails(emails,us_emails,train,test)

        print("-")

        print("Creating New Features...")

        na_low=['TransactionAmt' , 'ProductCD' , 'card1' , 'card2' , 'card3' , 'card4' , 'card5' , 'card6' , 'addr1' ,

        'addr2' , 'P_emaildomain' , 'C1' , 'C2' , 'C3' , 'C4' , 'C5' , 'C6' , 'C7' , 'C8' , 'C9' , 'C10' , 

        'C11' , 'C12' , 'C13' , 'C14'  , 'M6' ]

        numeric=train[na_low]._get_numeric_data().columns

        

        encode_FE(train,test,['addr1','card1','card2','card3','P_emaildomain'])

        encode_CB('card1','addr1',train,test)

        train['day'] = train.TransactionDT / (24*60*60)

        train['uid'] = train.card1_addr1.astype(str)+'_'+np.floor(train.day-train.D1).astype(str)



        test['day'] = test.TransactionDT / (24*60*60)

        test['uid'] = test.card1_addr1.astype(str)+'_'+np.floor(test.day-test.D1).astype(str)

        encode_FE(train,test,['uid'])

        encode_AG(numeric, ['uid'],['mean',"std"], train, test)

        categorical_columns=test.columns.drop(test._get_numeric_data().columns)

        categorical_columns=categorical_columns.drop('uid')

        encode_AG2(categorical_columns, ['uid'], train, test)

        # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN

        encode_FE(train,test,['addr1','card1','card2','card3','P_emaildomain'])

        # COMBINE COLUMNS CARD1+ADDR1+P_EMAILDOMAIN

        encode_CB('card1_addr1','P_emaildomain',train, test)

        # FREQUENCY ENOCDE

        encode_FE(train,test,['card1_addr1','card1_addr1_P_emaildomain'])

        # GROUP AGGREGATE

        encode_AG(['TransactionAmt','D9','D11'],['card1','card1_addr1','card1_addr1_P_emaildomain'],['mean','std'],train,test,usena=True)

        del train['uid'], test['uid']

        print("Creating New Features Finished")

        print('-' * 10)

        print('Label Coding and One Hot Encoding Start...')

        categorical_columns=test.columns.drop(test._get_numeric_data().columns)

        from sklearn import preprocessing

        for i in categorical_columns: 

            lbe=preprocessing.LabelEncoder()

            train[i]=lbe.fit_transform(train[i].astype(str))

        for i in categorical_columns:    

            test[i]=lbe.fit_transform(test[i].astype(str))

        for i in categorical_columns:

            if (test[i].max()== train[i].max())&(train[i].max()<8):

                    test = pd.get_dummies(test, columns = [i])

                    train=pd.get_dummies(train, columns = [i])

          

        

                

        print('-' * 10)

        return train,test
def modeling():

    train,test=pre_processing_and_feature_engineering()

    with timer('Machine Learning '):

        #feature_drop=[]

        train_TransactionID= train["TransactionID"]

        test_TransactionID=test["TransactionID"]

        X= train.sort_values('TransactionDT').drop([ 'TransactionDT', 'TransactionID'], axis=1)

        y = train.sort_values('TransactionDT')['isFraud']

        test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)

        del train

        X=X.drop("isFraud", axis=1)

        print('-' * 30)

        print("Droping unusefull Features which were determined end of the  first training ...")

        #for i in feature_drop:

               #X=X.drop([i],axis=1)

                #test=test.drop([i],axis=1)

               #print(i,', ',end='')

        print('-')

        print('-' * 20)

        print('Machine Learning start ... ')

     

        params = {'num_leaves': 491,

              'min_child_weight': 0.03454472573214212,

              'feature_fraction': 0.3797454081646243,

              'bagging_fraction': 0.4181193142567742,

              'min_data_in_leaf': 106,

              'objective': 'binary',

              'max_depth': -1,

              'learning_rate': 0.006883242363721497,

              "boosting_type": "gbdt",

              "bagging_seed": 11,

              "metric": 'auc',

              "verbosity": -1,

              'reg_alpha': 0.3899927210061127,

              'reg_lambda': 0.6485237330340494,

              'random_state': 47}

        folds = TimeSeriesSplit(n_splits=5)



        aucs = list()

        feature_importances = pd.DataFrame()

        feature_importances['feature'] = X.columns



        

        for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):

            

            print('Training on fold {}'.format(fold + 1))



            trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

            val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])

            clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)



            feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()

            aucs.append(clf.best_score['valid_1']['auc'])



            print('Fold {} finished '.format(fold + 1))

        print('-' * 10)

        print('Training has finished.')

        

        print('Mean AUC:', np.mean(aucs))

        print('-' * 10)

        print("final model, set the output as a dataframe and convert to csv file named submission.csv")

        # clf right now is the last model

        best_iter = clf.best_iteration

        clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)

        clf.fit(X, y)

        #set the output as a dataframe and convert to csv file named submission.csv

        predictions = clf.predict_proba(test)[:, 1]

        output = pd.DataFrame({ "TransactionID" : test_TransactionID, "isFraud": predictions })

        output.to_csv('submission_lgbm.csv', index=False)

        print('Feature importances...')

                

        feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)

        feature_importances.to_csv('feature_importances.csv')

        love()

             

        plt.figure(figsize=(16, 16))

        sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

        plt.title('50 TOP feature importance ');

        

       

     

        return 
def main():

    with timer('Full Model Run '):

        print("Full Model Run Start...")

        print('-' * 50)

        modeling() 

        print('-' * 50)
if __name__ == "__main__":

     main()