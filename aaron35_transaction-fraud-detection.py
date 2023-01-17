#Load data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings('ignore')



#get training data and testing data

train_identity= pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv", index_col='TransactionID')

train_transaction= pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv", index_col='TransactionID')

test_identity= pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv", index_col='TransactionID')

test_transaction= pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv", index_col='TransactionID')

submission= pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv", index_col='TransactionID')
#reduce memory usage by setting the proper data structure

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

            df[col] = df[col].astype('object')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

#    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df



#apply the function on the datasets

train_transaction=reduce_mem_usage(train_transaction)

train_identity=reduce_mem_usage(train_identity)

test_transaction=reduce_mem_usage(test_transaction)

test_identity=reduce_mem_usage(test_identity)

submission=reduce_mem_usage(submission)
# join the identity table and transaction table together

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

x_test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')



#make the column name of train and test consistent(modify column name)

train = train.rename(columns=lambda x: x.replace('_','').replace("-",""))

x_test = x_test.rename(columns=lambda x: x.replace('_','').replace("-",""))



y_train=train['isFraud']

x_train=train.copy()

x_train.drop('isFraud', axis=1, inplace=True)



for col in x_train.columns:

    x_test[col]=x_test[col].astype(x_train[col].dtype)



#delete the unnecessary datasets

del train_identity,train_transaction,test_identity,test_transaction, train



#show data size

print("\nThe x_train data size  is : {} ".format(x_train.shape)) 

print("The test data size  is : {} ".format(x_test.shape))

import gc



def id_split(dataframe):

    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]

    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1].astype('object')



    dataframe['OS_id_30'] = dataframe['id30'].str.split(' ', expand=True)[0]

    dataframe['version_id_30'] = dataframe['id30'].str.split(' ', expand=True)[1].astype('object')



    dataframe['browser_id_31'] = dataframe['id31'].str.split(' ', expand=True)[0]

    dataframe['version_id_31'] = dataframe['id31'].str.split(' ', expand=True)[1].astype('object')



    dataframe['screen_width'] = dataframe['id33'].str.split('x', expand=True)[0]

    dataframe['screen_height'] = dataframe['id33'].str.split('x', expand=True)[1]



    dataframe['id34'] = dataframe['id34'].str.split(':', expand=True)[1]

    dataframe['id23'] = dataframe['id23'].str.split(':', expand=True)[1]

    

    

    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    dataframe['had_id'] = 1

    

    gc.collect()#run a Garbage Collector to release memory

    

    return dataframe



#apply the function on the datasets

x_train = id_split(x_train)

x_test = id_split(x_test)



#show data size

print("\nThe x_train data size  is : {} ".format(x_train.shape)) 

print("The test data size  is : {} ".format(x_test.shape))
#show missing values

miss_number=x_train.isnull().sum()

miss_ratio=x_train.isnull().sum()/len(x_train)

miss_info=pd.DataFrame({'Number of miss':miss_number,'Proportion of miss':miss_ratio},)

miss_info=miss_info.loc[miss_info['Number of miss']>0]

miss_info=miss_info.sort_values(by='Number of miss',ascending=0)

print(miss_info[miss_info['Proportion of miss']>0.99])



#drop the feature with overwhelm missing values

x_train.drop(list(miss_info[miss_info['Proportion of miss']>0.99].index), axis=1, inplace=True)

x_test.drop(list(miss_info[miss_info['Proportion of miss']>0.99].index), axis=1, inplace=True)



#fill the rest missing value with mode

for col in x_train.columns:

    x_train[col] = x_train[col].fillna(x_train[col].mode()[0])

    x_test[col] = x_test[col].fillna(x_train[col].mode()[0])
#Label Encoding the Categorical Variables

from sklearn.preprocessing import LabelEncoder

for col in x_train.columns:

    if x_train[col].dtype=='object' or x_test[col].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(x_train[col].values.astype('str')) + list(x_test[col].values.astype('str')))

        x_train[col] = lbl.transform(list(x_train[col].values.astype('str')))

        x_test[col] = lbl.transform(list(x_test[col].values.astype('str')))
# Load packages

import pandas as pd

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt #using for plot

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV#GridSearchCV
#logistic regression

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
#lightGBM

from sklearn.model_selection import KFold

import lightgbm as lgb

import datetime



print('####################################################\n{}\start_time'.format(datetime.datetime.now().strftime('%H:%M')))



params = {'num_leaves': [491],

          'min_child_weight': [0.03454472573214212],

          'feature_fraction': [0.3797454081646243],

          'bagging_fraction': [0.4181193142567742],

          'min_data_in_leaf': [106],

          'objective': ['binary'],

          'max_depth': [-1],

          'learning_rate': [0.006883242363721497],

          "boosting_type": ["gbdt"],

          "bagging_seed": [11],

          "metric": ['auc'],

          "verbosity": [-1],

          'reg_alpha': [0.3899927210061127],

          'reg_lambda': [0.6485237330340494],

          'random_state': [47],

          'device_type':['gpu']

         }



lgb_temp = lgb.LGBMClassifier()

model_lgb_tuned = GridSearchCV(lgb_temp, params,scoring='roc_auc')

model_lgb_tuned.fit(x_train,y_train)

model_lgb = lgb.LGBMClassifier(**model_lgb_tuned.best_params_)



print(model_lgb)

print('{}\tEnd_time\n####################################################'.format(datetime.datetime.now().strftime('%H:%M')))
#xgboost

import xgboost as xgb



print('####################################################\n{}\start_time'.format(datetime.datetime.now().strftime('%H:%M')))



params={

    'n_estimators':[500],

    'max_depth':[3],

    'learning_rate':[0.05],

    'subsample':[0.5],

    'tree_method':['gpu_hist']  # THE MAGICAL PARAMETER, use gpu

        }





xgb_temp = xgb.XGBClassifier()

model_xgb_tuned = GridSearchCV(xgb_temp, params,scoring='roc_auc')

model_xgb_tuned.fit(x_train,y_train)

model_xgb = xgb.XGBClassifier(**model_xgb_tuned.best_params_)



print(model_xgb)

print('{}\tEnd_time\n####################################################'.format(datetime.datetime.now().strftime('%H:%M')))
%%capture

#random forest (random forest is time consuming and space consuming. Can not run it with other models in one Kaggle notebook)

'''

from sklearn.ensemble import RandomForestClassifier

print('####################################################\n{}\start_time'.format(datetime.datetime.now().strftime('%H:%M')))



params={

    'n_estimators':[200],

    'max_features':[0.3],

    'min_samples_leaf':[20],

    #'verbose':[1],#show steps

    'n_jobs':[-1]

}



rf_temp = RandomForestClassifier();

model_rf_tuned = GridSearchCV(rf_temp, params,scoring='roc_auc');

model_rf_tuned.fit(x_train,y_train);

model_rf = RandomForestClassifier(**model_rf_tuned.best_params_);



model_rf.fit(x_train, y_train)

sub_rf = pd.DataFrame()

sub_rf['TransactionID'] = x_test.index.tolist()

sub_rf['isFraud'] = model_rf.predict_proba(x_test)[:,1]

sub_rf.to_csv('submission_rf.csv',index=False)



print(model_rf)

print('{}\tEnd_time\n####################################################'.format(datetime.datetime.now().strftime('%H:%M')))

'''
%%time

#cross validation to Compare the Performance and Stacking the Models

from sklearn.model_selection import KFold,cross_val_score

from mlxtend.classifier import StackingClassifier



#Validation function

n_folds = 3



models = {

    'Logistic':model_lr,

    'Lightgbm':model_lgb,

    'XGBoost':model_xgb,

    #'Random forest':model_rf  #random forest is time consuming

    }





# kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)

kf=n_folds



for model_ind, model_fn in models.items():

    print('Fitting:\t{}'.format(model_ind))

    #model_fn.fit(x_train, y_train)

    

    #cross validation

    auc= cross_val_score(model_fn, x_train, y_train, scoring='roc_auc', cv = kf)    

    

    print('Done! Error:\t{}\n'.format(auc.mean()))





#combine the model together(stacking)

lr = LogisticRegression()

averaged_models=StackingClassifier(classifiers=[model_lgb, model_xgb], 

                                   use_probas=True,average_probas=True,meta_classifier=lr)



# kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)

auc= cross_val_score(averaged_models, x_train, y_train, scoring='roc_auc', cv = kf)

score =auc.mean()

print(" Averaged base models score: \t{}\n".format(score))

#We use the stacked model for our final predictions.

averaged_models.fit(x_train, y_train)

sub = pd.DataFrame()

sub['TransactionID'] = x_test.index.tolist()

sub['isFraud'] = averaged_models.predict_proba(x_test)[:,1]

sub.to_csv('submission.csv',index=False)