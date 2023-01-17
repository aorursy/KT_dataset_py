import numpy as np

import pandas as pd

import gc

import time

import datetime

from contextlib import contextmanager

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

from sklearn import metrics, preprocessing



from sklearn.model_selection import KFold # StratifiedKFold

from sklearn.model_selection import GridSearchCV



from sklearn.decomposition import PCA # KernelPCA



import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



from sklearn.preprocessing import LabelEncoder





pd.set_option('display.max_rows', 999)

pd.set_option('display.max_columns',700)

pd.set_option('display.float_format', lambda x: '%.3f' % x)





import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", category=FutureWarning) 

warnings.filterwarnings("ignore", category=UserWarning) 

@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))

    

# Display/plot feature importance

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:100].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(15, 20))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances01.png')



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

def loading_data():

    print('LOADING DATA')

    train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

    train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")



    test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")

    test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")



    # Fix column name 

    fix_col_name = {testIdCol:trainIdCol for testIdCol, trainIdCol in zip(test_identity.columns, train_identity.columns)}

    test_identity.rename(columns=fix_col_name, inplace=True)



    ## Reduce memory

    train_transaction = reduce_mem_usage(train_transaction)

    train_identity = reduce_mem_usage(train_identity)



    test_transaction = reduce_mem_usage(test_transaction)

    test_identity = reduce_mem_usage(test_identity)



    # Merge (transaction-identity)

    train = train_transaction.merge(train_identity, on='TransactionID', how='left')

    test = test_transaction.merge(test_identity, on='TransactionID', how='left')



    #MERGE (X_train - X_test)

    train_test = pd.concat([train, test], ignore_index=True)



    print(f'train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')

    print(f'test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')



    del train_transaction, train_identity, test_transaction, test_identity; x = gc.collect()  

    return train_test
def processing_data(train_test):

    print('PROCESSING DATA')

    drop_col_list = []

    

    # TransactionDT

    START_DATE = '2015-04-22'

    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

    train_test['NewDate'] = train_test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    train_test['NewDate_YMD'] = train_test['NewDate'].dt.year.astype(str) + '-' + train_test['NewDate'].dt.month.astype(str) + '-' + train_test['NewDate'].dt.day.astype(str)

    train_test['NewDate_YearMonth'] = train_test['NewDate'].dt.year.astype(str) + '-' + train_test['NewDate'].dt.month.astype(str)

    train_test['NewDate_Weekday'] = train_test['NewDate'].dt.dayofweek

    train_test['NewDate_Hour'] = train_test['NewDate'].dt.hour

    train_test['NewDate_Day'] = train_test['NewDate'].dt.day

    drop_col_list.extend(["TransactionDT","NewDate"])  ## !!!

    

    # TransactionAMT

    train_test['New_Cents'] = (train_test['TransactionAmt'] - np.floor(train_test['TransactionAmt'])).astype('float32')

    train_test['New_TransactionAmt_Bin'] = pd.qcut(train_test['TransactionAmt'],15)

    

    #cardX

    card_cols = [c for c in train_test if c[0:2] == 'ca']

    for col in ['card2','card3','card4','card5','card6']:

        train_test[col] = train_test.groupby(['card1'])[col].transform(lambda x: x.mode(dropna=False).iat[0])

        train_test[col].fillna(train_test[col].mode()[0], inplace=True)

    

    

    # P_email_domain & R_email_domain 

    train_test.loc[train_test['P_emaildomain'].isin(['gmail.com', 'gmail']),'P_emaildomain'] = 'Google'

    train_test.loc[train_test['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk','yahoo.co.jp', 'yahoo.de', 'yahoo.fr','yahoo.es']), 'P_emaildomain'] = 'Yahoo'

    train_test.loc[train_test['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr']), 'P_emaildomain'] = 'Microsoft'

    train_test.loc[train_test['P_emaildomain'].isin(train_test['P_emaildomain'].value_counts()[train_test['P_emaildomain'].value_counts() <= 500 ].index), 'P_emaildomain'] = "Others"

    train_test['P_emaildomain'].fillna("Unknown", inplace=True)



    train_test.loc[train_test['R_emaildomain'].isin(['gmail.com', 'gmail']),'R_emaildomain'] = 'Google'

    train_test.loc[train_test['R_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk','yahoo.co.jp', 'yahoo.de', 'yahoo.fr','yahoo.es']), 'R_emaildomain'] = 'Yahoo'

    train_test.loc[train_test['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr']), 'R_emaildomain'] = 'Microsoft'

    train_test.loc[train_test['R_emaildomain'].isin(train_test['R_emaildomain'].value_counts()[train_test['R_emaildomain'].value_counts() <= 300 ].index), 'R_emaildomain'] = "Others"

    train_test['R_emaildomain'].fillna("Unknown", inplace=True)



    # DeviceInfo

    train_test['DeviceInfo'] = train_test['DeviceInfo'].fillna('unknown_device').str.lower()

    train_test['DeviceInfo'] = train_test['DeviceInfo'].str.split('/', expand=True)[0]

    

    train_test.loc[train_test['DeviceInfo'].str.contains('SM', na=False), 'DeviceInfo'] = 'Samsung'

    train_test.loc[train_test['DeviceInfo'].str.contains('SAMSUNG', na=False), 'DeviceInfo'] = 'Samsung'

    train_test.loc[train_test['DeviceInfo'].str.contains('GT-', na=False), 'DeviceInfo'] = 'Samsung'

    train_test.loc[train_test['DeviceInfo'].str.contains('Moto G', na=False), 'DeviceInfo'] = 'Motorola'

    train_test.loc[train_test['DeviceInfo'].str.contains('Moto', na=False), 'DeviceInfo'] = 'Motorola'

    train_test.loc[train_test['DeviceInfo'].str.contains('moto', na=False), 'DeviceInfo'] = 'Motorola'

    train_test.loc[train_test['DeviceInfo'].str.contains('LG-', na=False), 'DeviceInfo'] = 'LG'

    train_test.loc[train_test['DeviceInfo'].str.contains('rv:', na=False), 'DeviceInfo'] = 'RV'

    train_test.loc[train_test['DeviceInfo'].str.contains('HUAWEI', na=False), 'DeviceInfo'] = 'Huawei'

    train_test.loc[train_test['DeviceInfo'].str.contains('ALE-', na=False), 'DeviceInfo'] = 'Huawei'

    train_test.loc[train_test['DeviceInfo'].str.contains('-L', na=False), 'DeviceInfo'] = 'Huawei'

    train_test.loc[train_test['DeviceInfo'].str.contains('Blade', na=False), 'DeviceInfo'] = 'ZTE'

    train_test.loc[train_test['DeviceInfo'].str.contains('BLADE', na=False), 'DeviceInfo'] = 'ZTE'

    train_test.loc[train_test['DeviceInfo'].str.contains('Linux', na=False), 'DeviceInfo'] = 'Linux'

    train_test.loc[train_test['DeviceInfo'].str.contains('XT', na=False), 'DeviceInfo'] = 'Sony'

    train_test.loc[train_test['DeviceInfo'].str.contains('HTC', na=False), 'DeviceInfo'] = 'HTC'

    train_test.loc[train_test['DeviceInfo'].str.contains('ASUS', na=False), 'DeviceInfo'] = 'Asus'



    train_test.loc[train_test['DeviceInfo'].isin(train_test['DeviceInfo'].value_counts()[train_test['DeviceInfo'].value_counts() < 1000].index), 'DeviceInfo'] = "Others"



    # V1 - V339

    v_cols = [c for c in train_test if c[0] == 'V']

    v_nan_df = train_test[v_cols].isna()

    nan_groups={}



    for col in v_cols:

        cur_group = v_nan_df[col].sum()

        try:

            nan_groups[cur_group].append(col)

        except:

            nan_groups[cur_group]=[col]

    del v_nan_df; x=gc.collect()



    for nan_cnt, v_group in nan_groups.items():

        train_test['New_v_group_'+str(nan_cnt)+'_nulls'] = nan_cnt

        sc = preprocessing.MinMaxScaler()

        pca = PCA(n_components=2)

        v_group_pca = pca.fit_transform(sc.fit_transform(train_test[v_group].fillna(-1)))

        train_test['New_v_group_'+str(nan_cnt)+'_pca0'] = v_group_pca[:,0]

        train_test['New_v_group_'+str(nan_cnt)+'_pca1'] = v_group_pca[:,1]



    drop_col_list.extend(v_cols)

    

    print('CREATING NEW FEATURES')

        

    train_test['New_card1_card2']=train_test['card1'].astype(str)+'_'+train_test['card2'].astype(str)

    train_test['New_addr1_addr2']=train_test['addr1'].astype(str)+'_'+train_test['addr2'].astype(str)

    train_test['New_card1_card2_addr1_addr2']=train_test['card1'].astype(str)+'_'+train_test['card2'].astype(str)+'_'+train_test['addr1'].astype(str)+'_'+train_test['addr2'].astype(str)



    train_test['New_P_emaildomain_addr1'] = train_test['P_emaildomain'] + '_' + train_test['addr1'].astype(str)

    train_test['New_R_emaildomain_addr2'] = train_test['R_emaildomain'] + '_' + train_test['addr2'].astype(str)

    

    #Aggregation features

    train_test['New_TransactionAmt_to_mean_card1'] = train_test['TransactionAmt'] / train_test.groupby(['card1'])['TransactionAmt'].transform('mean')

    train_test['New_TransactionAmt_to_mean_card4'] = train_test['TransactionAmt'] / train_test.groupby(['card4'])['TransactionAmt'].transform('mean')

    train_test['New_TransactionAmt_to_std_card1'] = train_test['TransactionAmt'] / train_test.groupby(['card1'])['TransactionAmt'].transform('std')

    train_test['TransactionAmt_to_std_card4'] = train_test['TransactionAmt'] / train_test.groupby(['card4'])['TransactionAmt'].transform('std')



    train_test['New_id_02_to_mean_card1'] = train_test['id_02'] / train_test.groupby(['card1'])['id_02'].transform('mean')

    train_test['New_id_02_to_mean_card4'] = train_test['id_02'] / train_test.groupby(['card4'])['id_02'].transform('mean')

    train_test['New_id_02_to_std_card1'] = train_test['id_02'] / train_test.groupby(['card1'])['id_02'].transform('std')

    train_test['New_id_02_to_std_card4'] = train_test['id_02'] / train_test.groupby(['card4'])['id_02'].transform('std')



    train_test['New_D15_to_mean_card1'] = train_test['D15'] / train_test.groupby(['card1'])['D15'].transform('mean')

    train_test['New_D15_to_mean_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('mean')

    train_test['New_D15_to_std_card1'] = train_test['D15'] / train_test.groupby(['card1'])['D15'].transform('std')

    train_test['New_D15_to_std_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('std')



    train_test['New_D15_to_mean_addr1'] = train_test['D15'] / train_test.groupby(['addr1'])['D15'].transform('mean')

    train_test['New_D15_to_mean_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('mean')

    train_test['New_D15_to_std_addr1'] = train_test['D15'] / train_test.groupby(['addr1'])['D15'].transform('std')

    train_test['New_D15_to_std_card4'] = train_test['D15'] / train_test.groupby(['card4'])['D15'].transform('std')

    

    drop_col_list.extend(card_cols)

    

    # Frequency Encoding 

    fe_col_list=["New_TransactionAmt_Bin",'card4','card6','P_emaildomain','R_emaildomain','DeviceType','DeviceInfo']+[c for c in train_test if c[0] == 'M']

    for col in fe_col_list:

        vc = train_test[col].value_counts(dropna=True, normalize=True).to_dict()

        vc[-1] = -1

        nm = 'New_'+col+'_FE'

        train_test[nm] = train_test[col].map(vc)

        train_test[nm] = train_test[nm].astype('float32')



    

    print('DROPING UNNECESSARY FEATURES')

    train_test=train_test.drop(drop_col_list, axis=1)

    

    print('APPLYING LABEL ENCODING TO CATEGORICAL FEATURES')

    for col in train_test.columns:

        if train_test[col].dtype == 'object':

            le = LabelEncoder()

            le.fit(list(train_test[col].astype(str).values))

            train_test[col] = le.transform(list(train_test[col].astype(str).values))

    

    print('REDUCING MEMORY USAGE')

    train_test = reduce_mem_usage(train_test)

    

    print('DATA IS READY TO MODELLING')

    

    return train_test
def modeling(train_test,target):



    train = train_test[train_test[target].notnull()]

    test = train_test[train_test[target].isnull()]



    folds = KFold(n_splits = 10, shuffle = True, random_state = 1001)



    oof_preds = np.zeros(train.shape[0])

    sub_preds = np.zeros(test.shape[0])

    

    feature_importance_df = pd.DataFrame()



    features = [f for f in train.columns if f not in [target,'TransactionID','New_TransactionAmt_Bin','NewDate']]



    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[target])):

        

        start_time = time.time()

        print('Training on fold {}'.format(n_fold + 1))



        X_train, y_train = train[features].iloc[train_idx], train[target].iloc[train_idx]



        X_valid, y_valid = train[features].iloc[valid_idx], train[target].iloc[valid_idx]

        

        params={'learning_rate': 0.01,

        'objective': 'binary',

        'metric': 'auc',

        'num_threads': -1,

        'num_leaves': 256,

        'verbose': 1,

        'random_state': 42,

        'bagging_fraction': 1,

        'feature_fraction': 0.85 }

       

        clf = LGBMClassifier(**params, n_estimators=1000) #categorical_feature = LGBM_cat_col_list



        clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], 

                eval_metric = 'auc', verbose = 200, early_stopping_rounds = 200)



        #y_pred_valid

        oof_preds[valid_idx] = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]

        sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = features

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)





        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_valid, oof_preds[valid_idx]))) 





    print('Full AUC score %.6f' % roc_auc_score(train[target], oof_preds)) #y_pred_valid   



    test[target] = sub_preds

    test[['TransactionID', target]].to_csv("submission_lightgbm2.csv", index= False)



    display_importances(feature_importance_df)

    

    return feature_importance_df
def main():

    with timer("Loading Data"):

        train_test = loading_data()

    

    with timer("Preprocessing Data"):

        train_test = processing_data(train_test)

        

    with timer("Modeling"):

        feat_importance = modeling(train_test ,'isFraud')
if __name__ == "__main__":

    with timer("Full model run"):

        main()