import os
print(os.listdir("../Kaggle_HomeCredit/infiles"))
# import libraries and Load data  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from time import time
import datetime
import missingno as msno
%matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

tstart = time()

#from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier

import gc
import sys
#print(sys.base_prefix)

from sklearn.metrics import auc, roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.metrics import classification_report_imbalanced

#from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
#from scipy.stats import randint
#import scipy.stats as st

import warnings
warnings.filterwarnings("ignore")

LABELS = ["Normal", "Loan Default"]
file_path = '../input/'
train_file_path = file_path + 'application_train.csv'
test_file_path = file_path + 'application_test.csv'
bureau_file_path = file_path + 'bureau.csv'
bureau_balance_file_path = file_path + 'bureau_balance.csv'
credit_card_file_path = file_path + 'credit_card_balance.csv'
installments_payments_file_path = file_path + 'installments_payments.csv'
previous_application_file_path = file_path + 'previous_application.csv'
POS_CASH_balance_file_path = file_path + 'POS_CASH_balance.csv'

def plot_bar_graph(df, feature, feature_label) :
    val_count = df[feature].value_counts()
    fig, ax = plt.subplots(figsize=(9,6))
    sns.set(style="darkgrid")
    sns.barplot(val_count.index, val_count.values, alpha=0.9, ax=ax)
    plt.title('Frequency Distribution ')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(feature_label, fontsize=12)
    plt.show()
    
def plot_pie_graph(df, feature) :
    labels = df[feature].astype('category').cat.categories.tolist()
    counts = df[feature].value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots(figsize=(10,10))
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
    ax1.axis('equal')
    plt.title('Frequency Distribution ')
    plt.show()
    


def cat_data_distribution(df):
    tot_cred_len = len(df)
    for col in df.columns:
        #print(col,  bureau_data[col].unique())
        unique_cnt = len(df[col].unique())
        if unique_cnt <= 25:
            #if unique_cnt <= 10:
            print(df[col].value_counts(), '\n\n Total : ', df[col].value_counts().sum(), ' out of : ', tot_cred_len )
            plot_bar_graph(df, col, col)

### Funtion to identify columns with 'Nan' or missing values

def missing_data_col(df):
    null_col_list = df.columns[df.isna().any()].tolist()
    return null_col_list

### Function to Encode or convert Object type column to  numeric 

def convert_category(df):
    for col in df:
        le=LabelEncoder()
        if df[col].dtype == 'object':
            col_name = df[col].name
            
            # replace or mask the 'Nan' values if any before applying the laberEncoder
            df[col_name] = df[col_name].factorize()[0]
            # Apply LabelEncoder : fit and transform categorical object data types to numeric 
            le.fit(df[col_name])
            df[col_name] = le.transform(df[col_name])
        
    # print(df.head())
    return df

def fill_missing_col(df, col , not_null_list):
    
    #Feature set
    # Split sets into train and test
    train  = df.loc[ (df[col].notnull()==True) ]# known COLUMN values
    test = df.loc[ (df[col].isnull()==True) ]# null or unknown COLUMN values
    
    # missing values columns are stored in a target array
    y = train[col] #.values
    # print('len y ...', y)
    # All the other values are stored in the feature array
    X = train[not_null_list]   #.values[:, 1::]
    # print('len X ...', X)
    if len(df[col].value_counts()) < 25:
        # Create and fit a model
        #model = RandomForestClassifier(n_estimators=20, n_jobs=-1)
        model = XGBClassifier(n_estimators=10, n_jobs=-1, random_state =101)
    else:
        #model = RandomForestRegressor(n_estimators=20, n_jobs=-1)
        model =  XGBRegressor(n_estimators=10, n_jobs=-1, random_state =101)
    
    model.fit(X, y)
    
    # Use the fitted model to predict the missing values
    pred_y = model.predict(test[not_null_list])
    
    # Assign those predictions to the full data set
    df.loc[ (df[col].isnull()), col ] = pred_y 
    
    del train, test, X,y
    gc.collect()
    
    return df

def data_preprocessing (df):
    df = convert_category(df)
    # Get columns with missing values
    null_col_list = missing_data_col(df) 
    
    # Get total column list 
    total_col_list = list(df.columns.values)
    # Get columns without missing values
    not_null_list = list(set(total_col_list) - set(null_col_list)) 
    #not_null_list
    
    #predicting missing values in age using Random Forest
    for col in null_col_list:
        print('Replacing missing value for :', col)
        df = fill_missing_col(df, col, not_null_list)
    
    return df

# load data
train_df = pd.read_csv(train_file_path)
print ("Shape of Application Train data : ", train_df.shape)
test_df = pd.read_csv(test_file_path)
print ("Shape of Application Test data : ",test_df.shape)

# cat_data_distribution(train_df)
train_df = train_df[ train_df['CODE_GENDER'] != 'XNA']
train_df = train_df[ train_df['NAME_FAMILY_STATUS'] != 'Unknown']
train_df = train_df[ train_df['FLAG_MOBIL'] == 1]
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
count_classes = pd.value_counts(train_df['TARGET'], sort = True)
count_classes.plot(kind = 'bar', rot=0, figsize = (10,6))
plt.title("Transaction Target distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Target")
plt.ylabel("Frequency");
# tcred = time()
# create 'TARGET' in test data and assgn value  '-1'
test_df['TARGET'] = -1

# Concatenate train data and test data
credit_df = pd.concat([train_df, test_df], axis = 0, ignore_index=True)
credit_df.shape
# credit_df['TARGET'].value_counts() # '=1' belongs to test data
del train_df, test_df
gc.collect()

# print( "Concat train + test :  {} secs" .format(time() - tcred))
msno.bar(credit_df);
cred_col = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
       #'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK', 
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',  'AMT_REQ_CREDIT_BUREAU_YEAR',
       'APARTMENTS_AVG', 'APARTMENTS_MEDI', 'APARTMENTS_MODE',
       #'BASEMENTAREA_AVG', 'BASEMENTAREA_MEDI', 'BASEMENTAREA_MODE',
       'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'CODE_GENDER',  'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
       'DAYS_REGISTRATION', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'EMERGENCYSTATE_MODE',
       # 'ELEVATORS_AVG', 'ELEVATORS_MEDI','ELEVATORS_MODE',  
       #'ENTRANCES_MEDI', 'ENTRANCES_MODE', 'ENTRANCES_AVG',
       'EXT_SOURCE_1', 'EXT_SOURCE_2',   'EXT_SOURCE_3', 
       'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_EMP_PHONE', 'FLAG_MOBIL',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_PHONE', 'FLAG_WORK_PHONE',
       #'FLOORSMAX_AVG', 'FLOORSMAX_MEDI', 'FLOORSMAX_MODE',
       #'FLOORSMIN_AVG', 'FLOORSMIN_MEDI', 'FLOORSMIN_MODE',
       'FONDKAPREMONT_MODE', 'HOUR_APPR_PROCESS_START', 
       'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION',
       'LIVINGAREA_AVG', 'LIVINGAREA_MEDI',
       'LIVINGAREA_MODE', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE',
       'NAME_TYPE_SUITE', 
       'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'OWN_CAR_AGE',
       'REGION_POPULATION_RELATIVE', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION',
       'REG_REGION_NOT_WORK_REGION', 'SK_ID_CURR', 'TARGET',
       ]
credit_df = credit_df[cred_col]
tt= time()
credit_df = data_preprocessing (credit_df)
print (" Total time for Data Pre Processing :  ({0:.3f} s)\n".format(time() - tt) )
default_loan = credit_df[credit_df.TARGET == 1]
normal_loan = credit_df[credit_df.TARGET == 0]
default_loan.shape
normal_loan.shape
# credit_df = data_preprocessing (credit_df)
# cat_data_distribution(credit_df) 
# Load Bureau Data
bureau_data = pd.read_csv(bureau_file_path)
# bureau_data.head()
# msno.bar(bureau_data);
# Call "cat_data_distribution" funtion to plot all categorical features with 15 or less categories.
# cat_data_distribution(bureau_data)        

plot_pie_graph(bureau_data, 'CREDIT_TYPE')
print ("Shape of Bureau data : ", bureau_data.shape)
# Identify categorical important features and 
# Remove records with less number of nomber of transactions compared to total records. 

bureau_data = bureau_data.loc[(bureau_data['CREDIT_ACTIVE'] != 'Sold') & (bureau_data['CREDIT_ACTIVE'] != 'Bad debt') ]
bureau_data = bureau_data.loc[(bureau_data['CREDIT_TYPE'] == 'Consumer credit') | (bureau_data['CREDIT_TYPE'] == 'Credit card') 
                             | (bureau_data['CREDIT_TYPE'] == 'Car loan') | (bureau_data['CREDIT_TYPE'] == 'Mortgage') ]
msno.bar(bureau_data);
bureau_data = bureau_data[ bureau_data['CREDIT_DAY_OVERDUE'] <=180]
# bureau_data = data_preprocessing (bureau_data)

# Remove features that has less impact on the transations. 
# Here you need your intutions keeping the classification modeling in mind.
bureau_data = bureau_data.drop(['DAYS_CREDIT','CREDIT_DAY_OVERDUE','AMT_CREDIT_MAX_OVERDUE', 'AMT_ANNUITY',
                                'CREDIT_CURRENCY','CNT_CREDIT_PROLONG', 'CREDIT_ACTIVE', 'DAYS_CREDIT_ENDDATE',
                                'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE','CREDIT_TYPE'], axis = 1) 

# List columns with null data
# bureau_data.columns[bureau_data.isna().any()].tolist()

bureau_data = bureau_data.reset_index()
bureau_data = bureau_data.drop(['index'], axis = 1) 
# check the shape of bureau data
# bureau_data.info()
bureau_balance_data = pd.read_csv(bureau_balance_file_path)
print ("Shape of Bureau Balance data : ",bureau_balance_data.shape)
# bureau_balance_data.head()
plot_pie_graph(bureau_balance_data, 'STATUS')
# Discard 'Status' with negligible transaction
bureau_balance_data = bureau_balance_data.loc[(bureau_balance_data['STATUS'] == 'C' ) 
                                              | (bureau_balance_data['STATUS'] == '0')
                                             | (bureau_balance_data['STATUS'] == 'X')
                                              | (bureau_balance_data['STATUS'] == 1)
                                             ]

# Lets create a column "LATEST_MONTH" to get transaction of the latest month in the record or dataset based on "SK_ID_BUREAU". 
# Based on 'LATEST_MONTH' we will select or extract the corresponding status in a new field 'BUREAU_BALANCE_STATUS' 
# and store "Null" value in records other than latest status. 
# Finally we will drop all thoese records with null values. This will gives us records with latest transaction status.

bureau_balance_data['LATEST_MONTH'] = bureau_balance_data.groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].transform(max)
bureau_balance_data['BUREAU_BALANCE_STATUS'] = np.where((bureau_balance_data['MONTHS_BALANCE'] == bureau_balance_data['LATEST_MONTH']) 
                     , bureau_balance_data['STATUS'], np.nan)

# Drop all the rows with 'NaN' values. This will leave only single row for each 'SK_ID_BUREAU'.
bureau_balance_data = bureau_balance_data.dropna()

bureau_balance_data = bureau_balance_data.drop(['LATEST_MONTH', 'BUREAU_BALANCE_STATUS', 'MONTHS_BALANCE'], axis = 1) 
# bureau_balance_data.info()

bureau_result = pd.merge(bureau_data,
                 bureau_balance_data[[ 'SK_ID_BUREAU', 'STATUS']],   #, 'STATUS'
                 on='SK_ID_BUREAU', 
                 how='left')
msno.bar(bureau_result);
# cat_data_distribution(bureau_result) 
# bureau_result.columns[bureau_result.isna().any()].tolist()
#bcol = bureau_result.columns.values
new_col =['SK_ID_CURR']
select_col = [   'AMT_CREDIT_SUM',   'AMT_CREDIT_SUM_DEBT' , 'AMT_CREDIT_SUM_LIMIT'  ]

for col in select_col:
    #bureau_result['BUREAU_MEAN_'+ col] = bureau_result.groupby(['SK_ID_CURR','CREDIT_TYPE'])[col].transform('sum')
    bureau_result['BUREAU_MEAN_'+ col] = bureau_result.groupby(['SK_ID_CURR'])[col].transform('sum')
    new_col.append ('BUREAU_MEAN_'+ col)

# new_col
bureau_result = bureau_result[new_col]

bureau_result.drop_duplicates(keep = 'first',inplace = True)

# bureau_result.loc[(bureau_result['SK_ID_CURR'] == 120860)]
credit_df = pd.merge(credit_df, bureau_result, on='SK_ID_CURR', how='left')
# credit_df.head(10)

del bureau_result, bureau_data, bureau_balance_data
gc.collect()

credit_df.shape
previous_application_data = pd.read_csv(previous_application_file_path)
print ("Shape of Previous Application data : ",previous_application_data.shape)
# 
msno.bar(previous_application_data);
previous_application_data[['SK_ID_PREV','SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY',
       'AMT_APPLICATION', 'AMT_CREDIT',        'AMT_GOODS_PRICE', 
        'NAME_CONTRACT_STATUS', 'DAYS_DECISION',
       'NAME_PAYMENT_TYPE', 
       'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
       'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'SELLERPLACE_AREA',
       'NAME_SELLER_INDUSTRY']].head()
# cat_data_distribution(previous_application_data)

prev_col = ['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY',
       'AMT_APPLICATION', 'AMT_CREDIT',        'AMT_GOODS_PRICE', 
        'NAME_CONTRACT_STATUS', 'DAYS_DECISION',
       'NAME_PAYMENT_TYPE', 
       'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO',
       'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'SELLERPLACE_AREA',
       'NAME_SELLER_INDUSTRY']

previous_application_data = previous_application_data[prev_col] 
pos_cash_balance_data = pd.read_csv(POS_CASH_balance_file_path)
# pos_cash_balance_data.head()
msno.bar(pos_cash_balance_data);
# cat_data_distribution(pos_cash_balance_data)
# plot_pie_graph(pos_cash_balance_data, 'NAME_CONTRACT_STATUS')
# Select records with 'Active', 'Completed' and 'Signed' status which are more than 10% of total transactions
pos_cash_balance_data = pos_cash_balance_data.loc[(pos_cash_balance_data['NAME_CONTRACT_STATUS'] == 'Active') 
                                              | (pos_cash_balance_data['NAME_CONTRACT_STATUS'] == 'Completed')
                                             | (pos_cash_balance_data['NAME_CONTRACT_STATUS'] == 'Signed')
                                             ]

# select records where Days past Due is less than 365 days i.e 365.243
pos_cash_balance_data = pos_cash_balance_data[pos_cash_balance_data['SK_DPD'] <= 365243]
# msno.bar(pos_cash_balance_data);
# Drop columns 
pos_cash_balance_data = pos_cash_balance_data.drop([ 'SK_DPD',  'SK_DPD_DEF', 'CNT_INSTALMENT_FUTURE',
                                                     'NAME_CONTRACT_STATUS','MONTHS_BALANCE' ], axis = 1) 

# Drop Duplicate records
pos_cash_balance_data.drop_duplicates(keep = 'first',inplace = True)

## Merge or Join the Previous Application data with POS Cash data 

previous_result = pd.merge(previous_application_data,  pos_cash_balance_data, on= ['SK_ID_PREV','SK_ID_CURR'],  how='left')
# previous_result.head()

del previous_application_data, pos_cash_balance_data
gc.collect()
# previous_result.head()
credit_card_balance_data = pd.read_csv(credit_card_file_path)
# credit_card_balance_data.head()
msno.bar(credit_card_balance_data);
# credit_card_balance_data.shape

# cat_data_distribution(credit_card_balance_data)
credit_card_balance_data = credit_card_balance_data.loc[(credit_card_balance_data['NAME_CONTRACT_STATUS'] == 'Active') 
                                              | (credit_card_balance_data['NAME_CONTRACT_STATUS'] == 'Completed')
                                             | (credit_card_balance_data['NAME_CONTRACT_STATUS'] == 'Signed') ]

#** select records where Days past Due is less than 365 days **
credit_card_balance_data = credit_card_balance_data[credit_card_balance_data['SK_DPD'] <= 365243]
# credit_card_balance_data.head()
# 
new_col =['SK_ID_PREV','SK_ID_CURR']
select_col = [ 'AMT_BALANCE', 'AMT_PAYMENT_TOTAL_CURRENT',  'AMT_RECIVABLE' , 'AMT_TOTAL_RECEIVABLE'   ]

for col in select_col:
    credit_card_balance_data['CC_'+ col] = credit_card_balance_data.groupby(['SK_ID_PREV','SK_ID_CURR'])[col].transform('mean')
    new_col.append ('CC_'+ col)


credit_card_balance_data = credit_card_balance_data[new_col]
# Drop Duplicate records
credit_card_balance_data.drop_duplicates(keep = 'first',inplace = True)
# msno.bar(credit_card_balance_data);
previous_result = pd.merge(previous_result, credit_card_balance_data, on= ['SK_ID_PREV','SK_ID_CURR'],  how='left')  

# previous_result.shape
del credit_card_balance_data
gc.collect()
installments_payments_data = pd.read_csv(installments_payments_file_path)
print ("Shape of Installment Payment data : ",installments_payments_data.shape)
msno.bar(installments_payments_data);
#cat_data_distribution(installments_payments_data)
plot_pie_graph(installments_payments_data, 'NUM_INSTALMENT_VERSION')
#plot_bar_graph(installments_payments_data, 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_VERSION') 
# Discard records with rare or less count of 'NUM_INSTALMENT_VERSION' compared to total transactions 
installments_payments_data = installments_payments_data.loc[  (installments_payments_data['NUM_INSTALMENT_VERSION'] == 0.0) 
                                                            | (installments_payments_data['NUM_INSTALMENT_VERSION'] == 1.0)
                                                            | (installments_payments_data['NUM_INSTALMENT_VERSION'] == 2.0)
                                                            | (installments_payments_data['NUM_INSTALMENT_VERSION'] == 3.0)
                                                         #   | (installments_payments_data['NUM_INSTALMENT_VERSION'] == 4.0)
                                                         #   | (installments_payments_data['NUM_INSTALMENT_VERSION'] == 5.0)
                                              ]
# installments_payments_data.loc[(installments_payments_data['SK_ID_PREV'] == 1496271)]
installments_payments_data['DELAY_INSTALMENT_PAYMENT'] = installments_payments_data['DAYS_INSTALMENT'] - installments_payments_data['DAYS_ENTRY_PAYMENT']
installments_payments_data = installments_payments_data[installments_payments_data['DELAY_INSTALMENT_PAYMENT'] > -181]
# msno.bar(installments_payments_data);
installments_col  = ['SK_ID_PREV', 'SK_ID_CURR','AMT_INSTALMENT', 'AMT_PAYMENT','DELAY_INSTALMENT_PAYMENT']
installments_payments_data = installments_payments_data[installments_col]

# installments_payments_data.loc[(installments_payments_data['SK_ID_PREV'] == 1496271)]
installments_payments_data.drop_duplicates(keep = 'first',inplace = True)

# examine duplicated rows
installments_payments_data.loc[installments_payments_data.duplicated(), :]

# installments_payments_data.shape
# installments_payments_data.loc[(installments_payments_data['SK_ID_PREV'] == 1496271)]
previous_result = pd.merge(previous_result,  installments_payments_data,  on= ['SK_ID_PREV','SK_ID_CURR'],  how='left')
# previous_result.shape

del installments_payments_data
gc.collect()
msno.bar(previous_result);
previous_result.columns.values
select_col = ['SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_ANNUITY',
       'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
       'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE',
       'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 
         'AMT_INSTALMENT', 'AMT_PAYMENT'
      ]

previous_result = previous_result[select_col]

previous_result.head()
cat_data_distribution(previous_result)
previous_result = previous_result.loc[(previous_result['NAME_CONTRACT_TYPE'] != 'XNA')  ]
previous_result = previous_result.loc[(previous_result['NAME_CLIENT_TYPE'] != 'XNA')  ]
previous_result.drop_duplicates(keep = 'first',inplace = True)
previous_result.head()
prev_col = ['SK_ID_CURR','NAME_CONTRACT_STATUS'] 
select_col_sum = [ 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE','AMT_INSTALMENT', 'AMT_PAYMENT']
for col in select_col_sum:
    previous_result['PREV_'+ col] = previous_result.groupby(['SK_ID_CURR'])[col].transform('sum')
    prev_col.append ('PREV_'+ col)
    

previous_result = previous_result[prev_col]
# previous_result.head()

previous_result.drop_duplicates(keep = 'first',inplace = True)
credit_df = pd.merge(credit_df, previous_result, on='SK_ID_CURR', how='left') 
# credit_df.shape

del previous_result
gc.collect()
msno.bar(credit_df);
cred_col = credit_df.columns.values
credit_df =  credit_df[cred_col]
credit_df.shape
credit_df.drop_duplicates(keep = 'first',inplace = True)
credit_df.shape
tt= time()
credit_df = data_preprocessing (credit_df)
print (" Total time for Data Pre Processing :  ({0:.3f} s)\n".format(time() - tt) )
# Separate the original Train data and Test data
train_df = credit_df.loc[(credit_df['TARGET'] != -1)]
test_df = credit_df.loc[(credit_df['TARGET'] == -1)]

# Drop the temporary target column from Test data
test_df = test_df.drop(["TARGET"], axis = 1) 
# train_df.head()

del credit_df
gc.collect()
# train_df.hist(bins=10,figsize=(25,30),grid=False);
tt= time()

# Get the training features and response variables
train_Y = train_df['TARGET']                   # Responce variable column for training
train_X = train_df.drop(["TARGET","SK_ID_CURR"], axis = 1)  # Feature variable columns for training

print('Original target dataset shape {}'.format(Counter(train_Y)))

# Scale the training data 
scaler = StandardScaler()  

#print('Standard scaling of train and test data in progress ...')
train_X_scale = scaler.fit(train_X).transform(train_X)

#
# Using XGBoost's "plot_importance" method to identify the importand features 
xgr = XGBRegressor(n_estimators=15, learning_rate=1.0, objective='binary:logistic', 
                    booster='gbtree',n_jobs= -1, #gamma=0.5 , 
                    min_child_weight=30, subsample=0.5, 
                    colsample_bytree=0.9, reg_alpha=0.01, reg_lambda=0.05,
                    random_state=101)

# train model  train_X_scale
xgr.fit(train_X, train_Y)

# Feature Importance
plt.figure(figsize=(20,15))
xgb.plot_importance(xgr, max_num_features=50, height=0.8, ax=plt.gca());
print (" Total Feature impt time :  ({0:.3f} s)\n".format(time() - tt) )
#
importances_df = pd.DataFrame({'feature':train_X.columns,'importance':np.round(xgr.feature_importances_,3)})
importances_df = importances_df.sort_values('importance',ascending=False) #.set_index('feature')
importances_df[:30]

feat_df = pd.DataFrame(importances_df[importances_df['importance']>= 0.019])
feat_col = list(feat_df['feature'])
print (" Total Importance Feature selected :  ", len(feat_col) )
# feat_col, len(feat_col)

del train_X, train_Y, importances_df, train_X_scale
gc.collect()
def evaluate_result(clf, X_train, y_train, X_test, y_test, param_flg ) :
    #print( 'Fitting the training set' ) 
    clf.fit(X_train, y_train)
    if  param_flg == 'Y':     print (clf.best_params_ )
 
    # Predict on training set
    #print( 'Predicting the training set' )
    pred_train_y = clf.predict(X_train)
      
    # Predict on testing set
    pred_test_y = clf.predict(X_test)
    #pred_prob_y = clf.predict_proba(X_test)
    
    # Is our model still predicting just one class? it shouldhave all classes say [0,1]
    print( "\n PREDICTING TARGET CLASS [train data]  : " , np.unique( pred_train_y ))
    print( "\n PREDICTING TARGET CLASS [test data ]  : " , np.unique( pred_test_y ) )
    print("\n" )
    
    # How's our accuracy?
    print(" 1. ACCURACY SCORE [train data]           : ", accuracy_score(y_train, pred_train_y) )
    print(" 2. ACCURACY SCORE [test data ]           : ", accuracy_score(y_test, pred_test_y) )
    
    # How's our ROC score?
    print(" 3. ROC-AUC SCORE  [test data ]           :  ", roc_auc_score(y_test, pred_test_y) )

    # How's actual vs prediction classification?
    conf_mat = confusion_matrix(y_test, pred_test_y)
    class_rpt = classification_report(y_test, pred_test_y)
    
    del pred_train_y, X_train, y_train, X_test, y_test
    gc.collect()
    
    return  pred_test_y, clf, conf_mat, class_rpt

### ROC Curve
def roc_plot(pred_y, test_y):
    ##Computing false and true positive rates
    fpr, tpr, thresholds= roc_curve(pred_y,test_y,drop_intermediate=False)
    
    print( "\n AUC and ROC curve : " )
    plt.figure()
    
    ##Adding the ROC
    plt.plot(fpr, tpr, color='red',    lw=2, label='ROC curve')
    
    ##Random FPR and TPR
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    
    ##Title and label
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.show()
    return

### Precision-Recall Curve for imblance class
def precision_recall_plot(pred_y, test_y):
    
    precision, recall, thresholds = precision_recall_curve( test_y, pred_y)
    
    print(" precision   : ", precision )
    print(" recall      : ", recall )
    print(" thresholds  : ", thresholds )
    
    plt.step(recall, precision, color='b', alpha=0.2,  where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,  color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format( average_precision_score(test_y, pred_y)))
    return

# PLOT HEATMAP OF CONFUSION MATRIX
def conf_matrix():
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
def submission_file(subfilename):
    
    # Get 'SK_ID_CURR' for submission id column
    submission_id = pd.DataFrame({ 'SK_ID_CURR' : test_df['SK_ID_CURR']}, dtype=np.int32)
    
    # Predict the target using the trained classifier and convert into dataframe
    sub_test_data = test_df[feat_col]
    pred_test_y = trained_clf.predict(sub_test_data)
    #     
    submission_tar = pd.DataFrame({ 'TARGET': pred_test_y},  dtype=np.int32)
    print ("Shape of Application Test data : ",submission_tar['TARGET'].value_counts())
    
    # Reset indexes
    submission_id.reset_index(drop=True, inplace=True)
    submission_tar.reset_index(drop=True, inplace=True)
    
    # Concat the the id and the predicted result
    submission_df =  pd.concat([submission_id,submission_tar],axis=1)    
    
    now = datetime.datetime.now()
    
    subname = subfilename + str (now.strftime("%Y-%m-%d_%H-%M")) + ".csv"
    # save into .csv submission file
    
    submission_df.to_csv("../Kaggle_HomeCredit/Output/" + subname, index=False)
    return submission_df

X = train_df[feat_col]
y = train_df.TARGET
t = time() 
hyperparameters = { 'xgbclassifier__max_depth': [15],                 'xgbclassifier__learning_rate': [0.25],
                    'xgbclassifier__n_estimators': [10],              'xgbclassifier__nthread': [-1],
                    'xgbclassifier__reg_alpha': [ 0.7],               'xgbclassifier__reg_lambda': [ 1.0],  
                    'xgbclassifier__max_delta_step': [0],             'xgbclassifier__min_child_weight': [10.0],
                    #'xgbclassifier__subsample': [1.0],              #'xgbclassifier__colsample_bytree': [0.9],
                    'xgbclassifier__objective': ['binary:logistic' ], 'xgbclassifier__scale_pos_weight': [1],
                    'xgbclassifier__gamma': [0.05],                   'xgbclassifier__seed': [101]
                  }

## Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

print( "Test target shape: \n", y_train.value_counts() )
print( "Test target shape: \n", y_test.value_counts() )

# Train model
pipeline = make_pipeline (MinMaxScaler(), XGBClassifier())

#print (pipeline.get_params())
print( "\n Model Training: " )
GSC = GridSearchCV(pipeline, hyperparameters, n_jobs = -1, scoring = 'roc_auc', cv=4)

# Call train_evaluate to model and evaluate 
pred_test_y, trained_clf, conf_mat, class_rpt =  evaluate_result(GSC, X_train, y_train, X_test, y_test, param_flg = 'Y')

print (" Total time for Training and Evaluating the Model :  ({0:.3f} s)\n".format(time() - t) )


# PLOT HEATMAP OF CONFUSION MATRIX
conf_matrix()
## Classification Report
print(class_rpt)
## ROC and Precision-Recall Plot
roc_plot(pred_test_y, y_test)
precision_recall_plot(pred_test_y, y_test)
t = time() 

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=101)

# Get the training features and response variables
# train_Y = train_df['TARGET']      # Responce variable column for training
# train_X = train_df[feat_col]      # Feature variable columns for training
X = train_df[feat_col]
y = train_df.TARGET
# X is the feature set and y is the target
for train_index, test_index in skf.split(X, y): 
    print("Train:", train_index) 
    print( "Validation:", test_index) 
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# Train model
ETC = ExtraTreesClassifier(n_estimators=15, max_depth=15, 
                           min_samples_split=5, min_samples_leaf=5, min_weight_fraction_leaf=0.0,
                           max_leaf_nodes=15, class_weight ='balanced_subsample',n_jobs=1, random_state=101)  
# (n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
# min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
# min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, 
# warm_start=False, class_weight=None)

# Train and Evaluate Model
pred_test_y, trained_clf, conf_mat, class_rpt = evaluate_result(ETC, X_train, y_train, X_test, y_test, param_flg = 'N')
# PLOT HEATMAP OF CONFUSION MATRIX
conf_matrix()
## Classification Report
print(class_rpt)
## ROC and Precision-Recall Plot
roc_plot(pred_test_y, y_test)
precision_recall_plot(pred_test_y, y_test)
print (" Total training time :  ({0:.3f} s)\n".format(time() - t) )
# RESAMPLE THE IMBALANCED CLASS BY OVERSAMPLING THE MINORITY CLASS 
print( train_df['TARGET'].value_counts())

# Separate majority and minority classes
df_majority = train_df[train_df.TARGET==0]
df_minority = train_df[train_df.TARGET==1]

# df_majority['TARGET'].value_counts()
# df_minority['TARGET'].value_counts()[1]

# Upsample minority class
df_sampled = resample(df_minority, replace=True,                            # sample with replacement
                      n_samples=df_majority['TARGET'].value_counts()[0],    # to match majority class
                      random_state=123)                                     # reproducible results
 
# Combine majority class with upsampled minority class
# resampled_df = pd.concat([df_minority, df_sampled], axis = 0)
resampled_df = pd.concat([df_majority, df_sampled]) 

#df_upsampled = df_upsampled.reset_index()
#df_upsampled = df_upsampled.drop(['index'], axis = 1) 

# Display new class counts
y = resampled_df['TARGET']
X = resampled_df.drop('TARGET', axis = 1)

print( X.shape, y.shape )
print( y.value_counts())
# Resampled data with selected importand Features
X = X[feat_col]
print( X.shape)

# TRAIN MODEL
t = time()  
# SPLIT TRAINING DATA 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
print( "Train target shape : \n", y_train.value_counts() )
print( "Test target shape  : \n", y_test.value_counts() )


print( "\n TRAINING THE MODEL " )
hyperparameters = { 'xgbclassifier__max_depth': [15],                 'xgbclassifier__learning_rate': [0.25],
                    'xgbclassifier__n_estimators': [10],              'xgbclassifier__nthread': [-1],
                    'xgbclassifier__reg_alpha': [ 0.7],               'xgbclassifier__reg_lambda': [ 1.0],  
                    'xgbclassifier__max_delta_step': [0],             'xgbclassifier__min_child_weight': [10.0],
                    #'xgbclassifier__subsample': [1.0],              #'xgbclassifier__colsample_bytree': [0.9],
                    'xgbclassifier__objective': ['binary:logistic' ], 'xgbclassifier__scale_pos_weight': [1],
                    'xgbclassifier__gamma': [0.05],                   'xgbclassifier__seed': [101]
                  }

#pipeline = make_pipeline (preprocessing.StandardScaler(), XGBClassifier())
pipeline = make_pipeline (MinMaxScaler(), XGBClassifier())
#print (pipeline.get_params())

GSC = GridSearchCV(pipeline, hyperparameters, n_jobs = -1, scoring = 'roc_auc', cv=4)

# Call train_evaluate to model and evaluate 
pred_test_y, trained_clf, conf_mat, class_rpt  =  evaluate_result(GSC, X_train, y_train, X_test, y_test, param_flg = 'Y')

print (" Total time for Training and Evaluating the Model :  ({0:.3f} s)\n".format(time() - t) )
# PLOT HEATMAP OF CONFUSION MATRIX
conf_matrix()
## Classification Report
print(class_rpt)
## ROC and Precision-Recall Plot
roc_plot(pred_test_y, y_test)
precision_recall_plot(pred_test_y, y_test)
from imblearn.over_sampling import     ADASYN 

X = train_df[feat_col]
y = train_df.TARGET
print('Original dataset shape {}'.format(Counter(y)))

ada = ADASYN(random_state=42)
# Train using original preprocessed training data
X_res, y_res = ada.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))

# Split Training data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.30, random_state=101)
t = time() 
# Define Classifier
ETC = ExtraTreesClassifier( n_estimators=15, max_depth= 7, min_samples_split=5,min_samples_leaf=5)
# Train and Evaluate Model
pred_test_y, trained_clf, conf_mat, class_rpt = evaluate_result(ETC, X_train, y_train, X_test, y_test, param_flg = 'N')

print (" Total training time :  ({0:.3f} s)\n".format(time() - t) )
# PLOT HEATMAP OF CONFUSION MATRIX
conf_matrix()
## Classification Report
print(class_rpt)
## ROC and Precision-Recall Plot
roc_plot(pred_test_y, y_test)
precision_recall_plot(pred_test_y, y_test)
print (" Total training time :  ({0:.3f} s)\n".format(time() - t) )
from imblearn.over_sampling import SMOTE
t = time()
X = train_df[feat_col]
y = train_df.TARGET

print('Original dataset shape {}'.format(Counter(y)))

smt = SMOTE(random_state=42)
X_res, y_res = smt.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))

# Split Training data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.30, random_state=101)

# Call train_evaluate to model and evaluate 
pred_test_y, trained_clf, conf_mat, class_rpt  =  evaluate_result(GSC, X_train, y_train, X_test, y_test, param_flg = 'N')

print (" Total time for Training and Evaluating the Model :  ({0:.3f} s)\n".format(time() - t) )
# PLOT HEATMAP OF CONFUSION MATRIX
conf_matrix()
## Classification Report
print(class_rpt)
## ROC and Precision-Recall Plot
roc_plot(pred_test_y, y_test)
precision_recall_plot(pred_test_y, y_test)
from imblearn.under_sampling import AllKNN

t = time()
X = train_df[feat_col]
y = train_df.TARGET

print('Original dataset shape {}'.format(Counter(y)))

akn = AllKNN(random_state=42)
X_res, y_res = akn.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))


# Split Training data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.30, random_state=101)

# Call train_evaluate to model and evaluate 

pred_test_y, trained_clf, conf_mat, class_rpt  =  evaluate_result(GSC, X_train, y_train, X_test, y_test, param_flg = 'N')

print (" Total time for Training and Evaluating the Model :  ({0:.3f} s)\n".format(time() - t) )
# PLOT HEATMAP OF CONFUSION MATRIX
conf_matrix()
## Classification Report
print(class_rpt)
## ROC and Precision-Recall Plot
roc_plot(pred_test_y, y_test)
precision_recall_plot(pred_test_y, y_test)
print( "Time taken to train the training set in {} secs" .format(time() - tstart))
