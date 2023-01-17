import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from pandas.plotting import scatter_matrix

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

from sklearn.preprocessing import LabelEncoder

import datetime

import warnings

warnings.filterwarnings('ignore')

data_raw= pd.read_csv("/kaggle/input/loan-default/train_data.csv")

data_test=pd.read_csv("/kaggle/input/loan-default/updated_test_data_20200728.csv")
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

data_train=reduce_mem_usage(data_raw)

data_test=reduce_mem_usage(data_test)

data_raw=None
#remove time travel feature

time_travel_feature=['recoveries','collection_recovery_fee','settlement_term',

                     'settlement_percentage','settlement_amount','debt_settlement_flag','out_prncp']#'last_pymnt_d','total_rec_prncp','last_pymnt_amnt','last_credit_pull_d','total_pymnt'

data_train.drop(time_travel_feature, axis=1, inplace=True)

data_test.drop(time_travel_feature, axis=1, inplace=True)

index_test=data_test['index']

data_test.drop('index', axis=1, inplace=True)
# #correlation

corr_mat=data_train.corr()



def get_abs_correlations_pair(corr_mat):

    au_corr = corr_mat.abs().unstack()   

    labels_to_drop = set()

    cols = corr_mat.columns

    for i in range(0, corr_mat.shape[1]):

        for j in range(0, i+1):

            labels_to_drop.add((cols[i], cols[j])) 

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[au_corr.notna()]



cor_pair_list=get_abs_correlations_pair(corr_mat)

cor_pair_list=cor_pair_list[cor_pair_list>0.9]

top_corr_features=cor_pair_list.index[cor_pair_list>0.9]



fig, axs = plt.subplots(3,5,figsize=(30,20))

for i in range(len(top_corr_features)):

    axs[i//5,i%5].scatter(data_train[top_corr_features[i][0]],data_train[top_corr_features[i][1]])

    axs[i//5,i%5].set_xlabel(top_corr_features[i][0])

    axs[i//5,i%5].set_ylabel(top_corr_features[i][1])
not_useful=['next_pymnt_d','funded_amnt','out_prncp_inv',

            'total_pymnt_inv','funded_amnt_inv','open_acc','num_rev_tl_bal_gt_0','tot_hi_cred_lim',

           'total_il_high_credit_limit','mths_since_last_record']#because of multicolinearity

data_train.drop(not_useful, axis=1, inplace=True)

data_test.drop(not_useful, axis=1, inplace=True)

data_train['ins_inc_ratio']=data_train['installment']/(data_train['annual_inc']+data_train['annual_inc_joint'])

data_test['ins_inc_ratio']=data_test['installment']/(data_test['annual_inc']+data_test['annual_inc_joint'])
#show missing values



miss_number=data_train.isnull().sum()

miss_ratio=data_train.isnull().sum()/len(data_train)

miss_info=pd.DataFrame({'Number of miss':miss_number,'Proportion of miss':miss_ratio},)

miss_info=miss_info.loc[miss_info['Number of miss']>0]

miss_info=miss_info.sort_values(by='Number of miss',ascending=0)

print(miss_info[miss_info['Proportion of miss']>0.9])



#drop the feature with overwhelm missing values

data_train.drop(list(miss_info[miss_info['Proportion of miss']>0.9].index), axis=1, inplace=True)

data_test.drop(list(miss_info[miss_info['Proportion of miss']>0.9].index), axis=1, inplace=True)



#fill missing value for specific features

for col in ['mths_since_last_delinq','mths_since_recent_bc_dlq','mths_since_last_major_derog',

            'mths_since_recent_revol_delinq','il_util']:

    data_train[col] = data_train[col].fillna(data_train[col].max())

    data_test[col] = data_test[col].fillna(data_train[col].max())



for col in ['inq_last_12m','total_cu_tl','open_acc_6m','open_il_12m','open_il_24m', 'open_act_il',

            'open_rv_12m','open_rv_24m','max_bal_bc','inq_fi','mths_since_recent_inq']:

    data_train[col] = data_train[col].fillna(0)

    data_test[col] = data_test[col].fillna(0)

    

for col in ['mths_since_rcnt_il']:

    data_train[col] = data_train[col].fillna(data_train[col].min())

    data_test[col] = data_test[col].fillna(data_train[col].min())

    

#fill the rest missing value with mode

for col in data_train.columns:

    data_train[col] = data_train[col].fillna(data_train[col].mode()[0])

    data_test[col] = data_test[col].fillna(data_train[col].mode()[0])
# Number of unique classes in each object column

data_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
not_useful=['emp_title','title','earliest_cr_line','zip_code','issue_d']

data_train.drop(not_useful, axis=1, inplace=True)

data_test.drop(not_useful, axis=1, inplace=True)
#Label Encoding the Categorical Variables



change_dict_grade = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}

data_train['grade'] = data_train['grade'].map(change_dict_grade).fillna(data_train['grade'])

data_test['grade'] = data_test['grade'].map(change_dict_grade).fillna(data_test['grade'])



data_train['sub_grade']=data_train['sub_grade'].str[0].map(change_dict_grade)*7+data_train['sub_grade'].str[1].astype(int)

data_test['sub_grade']=data_test['sub_grade'].str[0].map(change_dict_grade)*7+data_test['sub_grade'].str[1].astype(int)



data_train['emp_length']=data_train['emp_length'].str.extract('(\d+)').astype(int)

data_test['emp_length']=data_test['emp_length'].str.extract('(\d+)').astype(int)



change_dict_verif={'Verified':1,'Source Verified':2,'Not Verified':3}

data_train['verification_status'] = data_train['verification_status'].map(change_dict_verif).fillna(data_train['verification_status'])

data_test['verification_status'] = data_test['verification_status'].map(change_dict_verif).fillna(data_test['verification_status'])
print(data_train['loan_status'].value_counts())

change_dict_respond={'Fully Paid':0,'Current':0,'Charged Off':1}

data_train['loan_status'] = data_train['loan_status'].map(change_dict_respond).fillna(data_train['loan_status'])
#now let's have a final look of the sumary of the dataset after wragling

data_train.describe(include='all')
#down sampling

data_train_1=data_train.loc[data_train['loan_status']==1]

data_train_2=data_train.loc[data_train['loan_status']==0].sample(n = len(data_train_1)) 

data_train=pd.concat([data_train_1, data_train_2])
#LabelEncoder

def labelencoder_dataframe(df):

    for col in df.columns:

        if len(set(df[col]))==2:

            lbl = LabelEncoder()

            lbl.fit(list(df[col].values.astype('str')))

            df[col] = lbl.transform(list(df[col].values.astype('str')))

    return df



#separate feature and responds

x_train=data_train.copy()

x_train.drop('loan_status', axis=1, inplace=True)

y_train=data_train['loan_status']



data_test.drop('loan_status', axis=1, inplace=True)

data_train=None



x_train=labelencoder_dataframe(x_train)

data_test=labelencoder_dataframe(data_test)



#Get dummies for catigory variables.

n_train=x_train.shape[0]

data_temp=pd.concat((x_train, data_test),sort=False)

data_temp=pd.get_dummies(data_temp)



x_train=data_temp[:n_train]

data_test=data_temp[n_train:]

data_temp=None
import re



regex = re.compile(r"\[|\]|<", re.IGNORECASE)



x_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_train.columns.values]

data_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data_test.columns.values]

#logistic regression

from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(penalty='l2')
#xgboost

import xgboost as xgb



print('####################################################\n{}\start_time'.format(datetime.datetime.now().strftime('%H:%M')))



params={

    'n_estimators':[100,300],#100,300

    'max_depth':[5],#3,5,10

    'learning_rate':[0.01],#0.05,0.1,0.01

    'subsample':[0.8],#0.5,0.7,0.8,0.9,1

    'tree_method':['gpu_hist']  # THE MAGICAL PARAMETER, use gpu

        }





xgb_temp = xgb.XGBClassifier()

model_xgb_tuned = GridSearchCV(xgb_temp, params,scoring='roc_auc',cv=5)

model_xgb_tuned.fit(x_train,y_train)

model_xgb = xgb.XGBClassifier(**model_xgb_tuned.best_params_)



print(model_xgb)

print('{}\tEnd_time\n####################################################'.format(datetime.datetime.now().strftime('%H:%M')))
#lightGBM

from sklearn.model_selection import KFold

import lightgbm as lgb

import datetime



print('####################################################\n{}\start_time'.format(datetime.datetime.now().strftime('%H:%M')))







params = {'objective': ['binary'],

          'boosting_type': ["gbdt"],

          'metric': ['auc'],

          #'verbosity': [-1],

          'n_estimators':[100,300],#100,300

          'feature_fraction': [0.8],#0.5,0.7,0.9,0.6,0.8

          'bagging_fraction': [0.7],#0.5,0.7,0.9,0.6,0.8

          'bagging_freq':[1],#1,5,10

          'max_depth': [5],#3,4,5,8,10,6,7

          'learning_rate': [0.1],#0.1,0.05

          'device_type':['gpu']

         }





lgb_temp = lgb.LGBMClassifier()

model_lgb_tuned = GridSearchCV(lgb_temp, params,scoring='roc_auc',n_jobs=-1,cv=5)

model_lgb_tuned.fit(x_train,y_train)

model_lgb = lgb.LGBMClassifier(**model_lgb_tuned.best_params_)



#print(model_lgb)

print(model_lgb_tuned.best_params_)



print('{}\tEnd_time\n####################################################'.format(datetime.datetime.now().strftime('%H:%M')))
%%time

#cross validation to Compare the Performance and Stacking the Models

from sklearn.model_selection import KFold,cross_val_score

from mlxtend.classifier import StackingClassifier



#Validation function

n_folds = 5



models = {

    'Logistic':model_lr,

    'Lightgbm':model_lgb,

    'XGBoost':model_xgb,

    }



for model_ind, model_fn in models.items():

    print('Fitting:\t{}'.format(model_ind))

    #model_fn.fit(x_train, y_train)

    

    #cross validation

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)

    auc= cross_val_score(model_fn, x_train, y_train, scoring='roc_auc', cv = kf)    

    

    print('Done! Error:\t{}\n'.format(auc.mean()))



#combine the model together(stacking)

lr = LogisticRegression()

averaged_models=StackingClassifier(classifiers=[model_lgb, model_xgb], 

                                   use_probas=True,average_probas=True,meta_classifier=lr)



kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)

auc= cross_val_score(averaged_models, x_train, y_train, scoring='roc_auc', cv = kf)

score =auc.mean()

print(" Averaged base models score: \t{}\n".format(score))
#feature importance

lgb_reg = model_lgb.fit(x_train,y_train)

ax = lgb.plot_importance(lgb_reg,  max_num_features=10, height = 0.5,importance_type='gain')

plt.show()
#We use the stacked model for our final predictions.

averaged_models.fit(x_train, y_train)

sub = pd.DataFrame()

sub['index'] = index_test

sub['prediction_score'] = averaged_models.predict_proba(data_test)[:,1]

sub.to_csv('predictions_Aaron.csv',index=False)