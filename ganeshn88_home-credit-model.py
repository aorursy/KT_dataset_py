#numpy and pandas for data manipulation

import numpy as np

import pandas as pd 



# sklearn preprocessing for dealing with categorical variables

from sklearn.preprocessing import LabelEncoder



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns



#Data Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler



#Model building

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import lightgbm as lgb



#Model Evaluation



from sklearn.metrics import roc_auc_score



#Saving Models

from sklearn.externals import joblib
# List files available

print(os.listdir("../input/home-credit-default-risk/"))
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





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
def importdata():    

    print('-' * 80)

    print('train')

    train = import_data('../input/home-credit-default-risk/application_train.csv')



    print('-' * 80)

    print('test')

    test = import_data('../input/home-credit-default-risk/application_test.csv')



    print('-' * 80)

    print('bureau_balance')

    bureau_balance = import_data('../input/home-credit-default-risk/bureau_balance.csv')



    print('-' * 80)

    print('bureau')

    bureau = import_data('../input/home-credit-default-risk/bureau.csv')



    print('-' * 80)

    print('credit_card_balance')

    credit_card = import_data('../input/home-credit-default-risk/credit_card_balance.csv')



    print('-' * 80)

    print('installments_payments')

    installments = import_data('../input/home-credit-default-risk/installments_payments.csv')



    print('-' * 80)

    print('pos_cash_balance')

    pos_cash = import_data('../input/home-credit-default-risk/POS_CASH_balance.csv')



    print('-' * 80)

    print('previous_application')

    previous_app = import_data('../input/home-credit-default-risk/previous_application.csv')

    return train,test,bureau_balance,bureau,credit_card,installments,pos_cash,previous_app


def one_hot_code(app_train,app_test):

    

    app_train = pd.get_dummies(app_train)

    app_test = pd.get_dummies(app_test)

    

    train_labels = app_train['TARGET']

    print('Aligning Train and Test Data')

    app_train, app_test = app_train.align(app_test, join='inner', axis=1)

    



    print('Training Features shape: ', app_train.shape)

    print('Testing Features shape: ', app_test.shape)

    

    return app_train,app_test,train_labels

    
def scale_data(traindata,testdata):

   

    imputer = SimpleImputer(strategy = 'median')



    # Scale each feature to 0-1

    scaler = MinMaxScaler(feature_range = (0, 1))



    train = traindata.copy()

    test = testdata.copy()



    imputer.fit(train)

    train = imputer.transform(train)

    test = imputer.transform(test)



    scaler.fit(train)

    train = scaler.transform(train)

    test = scaler.transform(test)



    print('Training data shape: ', train.shape)

    print('Testing data shape: ', test.shape)



    return train,test

def featureengineering(df):

    

    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    

    return df

def preprocess_buildmodel(train,test,cvtype):

    train = featureengineering(train)

    test = featureengineering(test)

    print('--------Encoding Categorical Data----------')

    

    app_train_encoded,app_test_encoded,target = one_hot_code(train,test)

    print('-----------Scaling Data------------------------')

    train_scaled,test_scaled = scale_data(app_train_encoded,app_test_encoded)

    

    print('---------Splitting Data-----------------')

    X_train,X_test,Y_train,Y_test = train_test_split(train_scaled,target,test_size=0.2,random_state = 1)

    params = {

    'xgb': { 'learning_rate' : [0.05,0.10] ,'max_depth': [ 3,4,5,6,7] ,'n_estimators' : [500,1000] },

    'lgb': {  'learning_rate' : [0.05,0.10] ,'max_depth': [ 3,4,5,6,7] ,'n_estimators' : [500,1000]  }

    }

    models = {

    'xgb': XGBClassifier(), 

    'lgb':lgb.LGBMClassifier()

    }

    

    from collections import defaultdict

    model_result = defaultdict(list)

    

    for key in models.keys():

        

        model = models[key]

        param = params[key]

        

        if cvtype == 'grid':

            clf=GridSearchCV(model,param, cv=5, verbose=0)

        elif cvtype == 'rand':

            clf = RandomizedSearchCV(model, param, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

    

        clf.fit(X_train,Y_train,early_stopping_rounds=10,eval_metric = 'auc',eval_set=[(X_train, Y_train), (X_test, Y_test)])

    

        aucscoretr = roc_auc_score(Y_train, clf.predict_proba(X_train)[:,1])

        aucscore = roc_auc_score(Y_test, clf.predict_proba(X_test)[:,1])

        joblib_file = key + '_' + cvtype + '_model.pkl'

        joblib.dump(clf, joblib_file)

        model_result[key] = clf,clf.best_estimator_.get_params(),aucscoretr,aucscore,joblib_file

        

    return model_result,X_train,X_test,Y_train,Y_test,test_scaled

    
train,test,bureau_balance,bureau,credit_card,installments,pos_cash,previous_app = importdata()
data_size = 1000
#app_train = train.sample(data_size)

#app_test = test

#print('Appl data columns',train.columns)

#print('Shape',app_train.shape)

#model_result,X_train,X_test,Y_train,Y_test,test_scaled =preprocess_buildmodel(app_train,app_test,'rand')
#print('XGB Train AUC with App data only',model_result['xgb'][2])

#print('XGB Validation AUC with App data only',model_result['xgb'][3])

#print('LGB Train AUC with App data only',model_result['lgb'][2])

#print('LGB Validation AUC with App data only',model_result['lgb'][3])
#lgb_model = joblib.load(model_result['lgb'][4])

#xgb_model = joblib.load(model_result['xgb'][4])

#lgb_pred = lgb_model.predict_proba(test_scaled)[:, 1]

#xgb_pred = xgb_model.predict_proba(test_scaled)[:, 1]

#submit = app_test[['SK_ID_CURR']]

#submit['TARGET'] = (lgb_pred + xgb_pred)/2
#submit.to_csv('homecredit_withappdata_blend.csv', index = False)
prev_apps_avg = previous_app.groupby('SK_ID_CURR').mean()

prev_apps_avg.columns = ['p_' + col for col in prev_apps_avg.columns]

train_prev_app = train.merge(right=prev_apps_avg.reset_index(), how='left', on='SK_ID_CURR')



avg_inst = installments.groupby('SK_ID_CURR').mean()

avg_inst.columns = ['i_' + f_ for f_ in avg_inst.columns]

train_prev_app_inst = train_prev_app.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')



pos_cash = pos_cash.groupby('SK_ID_CURR').mean()

train_prev_app_inst_pos = train_prev_app_inst.merge(right=pos_cash.reset_index(), how='left', on='SK_ID_CURR')



avg_cc_bal = credit_card.groupby('SK_ID_CURR').mean()

avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

train_prev_app_inst_pos_credit = train_prev_app_inst_pos.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')



bureau_avg = bureau.groupby('SK_ID_CURR').mean()

bureau_avg.columns = ['B_' + f_ for f_ in bureau_avg.columns]

train_prev_app_inst_pos_credit_bureau = train_prev_app_inst_pos_credit.merge(right=bureau_avg.reset_index(), how='left', on='SK_ID_CURR')
app_train = train_prev_app_inst_pos_credit_bureau

app_test = test
app_train.columns
#app_train = train_prev_app_inst_pos_credit_bureau.sample(data_size)

app_train = train_prev_app_inst_pos_credit_bureau

app_test = test

print('Shape',app_train.shape)

model_result,X_train,X_test,Y_train,Y_test,test_scaled =preprocess_buildmodel(app_train,app_test,'rand')
lgb_model = joblib.load(model_result['lgb'][4])

xgb_model = joblib.load(model_result['xgb'][4])

lgb_pred = lgb_model.predict_proba(test_scaled)[:, 1]

xgb_pred = xgb_model.predict_proba(test_scaled)[:, 1]

submit = app_test[['SK_ID_CURR']]

submit['TARGET'] = (lgb_pred + xgb_pred)/2
print('XGB Train AUC with Full data only',model_result['xgb'][2])

print('XGB Validation AUC with Full data only',model_result['xgb'][3])

print('LGB Train AUC with Full data only',model_result['lgb'][2])

print('LGB Validation AUC with Full data only',model_result['lgb'][3])
submit.to_csv('homecredit_withprevapp_blend.csv', index = False)