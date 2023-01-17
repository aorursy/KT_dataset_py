# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/data-dictionary-xlsx/Data Dictionary.xlsx"))



# Any results you write to the current directory are saved as output.

# ['databikeloanprediction', 'data-dictionary-xlsx', 'bike-loan-defaulter-prediction']
train_data = pd.read_csv('../input/databikeloanprediction/train.csv')

test_data = pd.read_csv('../input/bike-loan-defaulter-prediction/test_bqCt9Pv.csv')

sample_submission = pd.read_csv('../input/bike-loan-defaulter-prediction/sample_submission_24jSKY6.csv')

data_dict = pd.read_excel('../input/data-dictionary-xlsx/Data Dictionary.xlsx')
sample_submission.head()
train_data.head(10)
train_data.info()
test_data.info()
# Renamed the column names as dots(.) available in column names

train_data.columns = ['UniqueID', 'disbursed_amount', 'asset_cost', 'ltv', 'branch_id',

'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Date_of_Birth',

'Employment_Type', 'DisbursalDate', 'State_ID', 'Employee_code_ID',

'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag',

'Driving_flag', 'Passport_flag', 'PERFORM_CNS_SCORE',

'PERFORM_CNS_SCORE_DESCRIPTION', 'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS',

'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT',

'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS',

'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT',

'SEC_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT',

'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',

'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'NO_OF_INQUIRIES',

'loan_default']
test_data.columns = ['UniqueID', 'disbursed_amount', 'asset_cost', 'ltv', 'branch_id',

'supplier_id', 'manufacturer_id', 'Current_pincode_ID', 'Date_of_Birth',

'Employment_Type', 'DisbursalDate', 'State_ID', 'Employee_code_ID',

'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag',

'Driving_flag', 'Passport_flag', 'PERFORM_CNS_SCORE',

'PERFORM_CNS_SCORE_DESCRIPTION', 'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS',

'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT',

'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS',

'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT',

'SEC_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT',

'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS',

'AVERAGE_ACCT_AGE', 'CREDIT_HISTORY_LENGTH', 'NO_OF_INQUIRIES']
# Dropping some columns

train_data = train_data.drop(['UniqueID'],axis=1)

__train_data = train_data.drop(['loan_default'],axis=1)



test_data = test_data.drop(['UniqueID'],axis=1)
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



## Testing Pupose Feature Selection ##

y = train_data['loan_default'] # Convert from string "Yes"/"No" to binary

feature_names = [i for i in __train_data.columns if __train_data[i].dtype in [np.int64]]

X = train_data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)





import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
train_data.groupby('PERFORM_CNS_SCORE_DESCRIPTION').loan_default.sum()
import seaborn as sns

sns.countplot(train_data['loan_default'])
#.groupby('Employment_Type').loan_default.sum

sns.countplot(train_data['Employment_Type'])
#sns.catplot(x="Employment_Type",hue="loan_default", kind="box", data=train_data)
sns.catplot(y="PERFORM_CNS_SCORE_DESCRIPTION", x="loan_default",kind="bar", data=train_data)
train_data.groupby('PERFORM_CNS_SCORE_DESCRIPTION').loan_default.count()
predictor_with_categorical = train_data.select_dtypes(exclude=['int64','float64','<M8[ns]'])

predictor_with_categorical['Employment_Type'] = predictor_with_categorical['Employment_Type'].fillna(predictor_with_categorical['Employment_Type'].mode()[0])

#predictor_with_categorical.columns



columns_names_categorical = train_data.loc[:,['Employment_Type','MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 

                                              'VoterID_flag','Driving_flag', 'Passport_flag']]

one_hot_encoded_training_predictors_with_categorical=pd.get_dummies(columns_names_categorical)



predictor_with_categorical_test = test_data.select_dtypes(exclude=['int64','float64','<M8[ns]'])

predictor_with_categorical_test['Employment_Type'] = predictor_with_categorical_test['Employment_Type'].fillna(predictor_with_categorical_test['Employment_Type'].mode()[0])

#predictor_with_categorical.columns



columns_names_categorical_test = test_data.loc[:,['Employment_Type','PERFORM_CNS_SCORE_DESCRIPTION','MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 

                                              'VoterID_flag','Driving_flag', 'Passport_flag']]

one_hot_encoded_training_predictors_with_categorical_test=pd.get_dummies(columns_names_categorical_test)

one_hot_encoded_training_predictors_with_categorical.head()
from datetime import date



# --------Train data--------------#

predictor_with_categorical['Date_of_Birth'] = pd.to_datetime(predictor_with_categorical['Date_of_Birth'],format="%d-%m-%y")

predictor_with_categorical['DisbursalDate'] = pd.to_datetime(predictor_with_categorical['DisbursalDate'],format="%d-%m-%y")





# --------Test data--------------#

predictor_with_categorical_test['Date_of_Birth'] = pd.to_datetime(predictor_with_categorical_test['Date_of_Birth'],format="%d-%m-%y")

predictor_with_categorical_test['DisbursalDate'] = pd.to_datetime(predictor_with_categorical_test['DisbursalDate'],format="%d-%m-%y")

from dateutil.relativedelta import relativedelta



##-----------Some of helper function-----------##

def f(end):

    r = relativedelta(pd.to_datetime('now'), end) 

    return r.years



def fmonth(end):

    r = relativedelta(pd.to_datetime('now'), end) 

    return r.years * 12 + r.months



def extract_month_str(_input):

    '''This func extract month from a string'''

    import re

    list  = re.findall('\d',_input)

    return int(list[0])*12 + int(list[1])





## ----------Applying function on Train Data--------- ##

predictor_with_categorical['Age']= predictor_with_categorical['Date_of_Birth'].apply(f)

predictor_with_categorical['DisbursalPeriod'] = predictor_with_categorical['DisbursalDate'].apply(fmonth)

predictor_with_categorical['AVERAGE_ACCT_AGE_months'] = predictor_with_categorical['AVERAGE_ACCT_AGE'].apply(extract_month_str)

predictor_with_categorical['CREDIT_HISTORY_LENGTH_months'] = predictor_with_categorical['CREDIT_HISTORY_LENGTH'].apply(extract_month_str)





## ----------Applying function on Test Data--------- ##

predictor_with_categorical_test['Age']= predictor_with_categorical_test['Date_of_Birth'].apply(f)

predictor_with_categorical_test['DisbursalPeriod'] = predictor_with_categorical_test['DisbursalDate'].apply(fmonth)

predictor_with_categorical_test['AVERAGE_ACCT_AGE_months'] = predictor_with_categorical_test['AVERAGE_ACCT_AGE'].apply(extract_month_str)

predictor_with_categorical_test['CREDIT_HISTORY_LENGTH_months'] = predictor_with_categorical_test['CREDIT_HISTORY_LENGTH'].apply(extract_month_str)



predictor_with_categorical.head()
predictor_with_categorical.columns
predictor_with_categorical.Employment_Type.unique()
predictor_with_categorical.columns[predictor_with_categorical.isnull().any()]



train_data.columns
train_data.head()
type(list(train_data.columns))
# for i in list(train_data.columns):

#     count = len(train_data[i].unique())

#     print('The column {} is having {} unique values'.format(i,count))

    
_train_data = train_data.loc[:,['disbursed_amount', 'asset_cost', 'ltv', 'PERFORM_CNS_SCORE',

        'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS',

       'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT',

       'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS',

       'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT',

       'SEC_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT',

       'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES',

       'State_ID', 'Employee_code_ID','branch_id','supplier_id','manufacturer_id','Current_pincode_ID']]



# 'PERFORM_CNS_SCORE_DESCRIPTION','Aadhar_flag', 'PAN_flag', 'VoterID_flag',





##------Test data selecting-------##



_test_data = test_data.loc[:,['disbursed_amount', 'asset_cost', 'ltv', 'PERFORM_CNS_SCORE',

        'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS',

       'PRI_OVERDUE_ACCTS', 'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT',

       'PRI_DISBURSED_AMOUNT', 'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS',

       'SEC_OVERDUE_ACCTS', 'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT',

       'SEC_DISBURSED_AMOUNT', 'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT',

       'NEW_ACCTS_IN_LAST_SIX_MONTHS', 'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES', 

        'State_ID', 'Employee_code_ID','branch_id','supplier_id','manufacturer_id','Current_pincode_ID']]
_train_data.columns
predictor_with_categorical[predictor_with_categorical.Age < 0].head()
test_predictor = predictor_with_categorical.loc[:,['Employment_Type','AVERAGE_ACCT_AGE','CREDIT_HISTORY_LENGTH','DisbursalPeriod'

                                                                         ,'AVERAGE_ACCT_AGE_months', 

                                                   'CREDIT_HISTORY_LENGTH_months','branch_id','supplier_id','manufacturer_id','Current_pincode_ID'

                                                                        ]]

# predictor_with_categorical[predictor_with_categorical.Date_of_Birth.dt.year > 2018]

# import datetime

# dd = datetime.datetime.strptime(d,'%y%m%d')

# if dd.year > 2005:

#    dd = dd.replace(year=dd.year-100)

merged_df_predictors_train=pd.concat([_train_data,one_hot_encoded_training_predictors_with_categorical],axis=1)

merged_df_predictors_test=pd.concat([_test_data,one_hot_encoded_training_predictors_with_categorical_test],axis=1)

#merged_df_predictors_test['PERFORM_CNS_SCORE_DESCRIPTION_Not Scored: More than 50 active Accounts found'] = 0

merged_df_predictors_train.columns
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

feature_names = ['disbursed_amount', 'asset_cost', 'ltv', 'PERFORM_CNS_SCORE',

       'PRI_NO_OF_ACCTS', 'PRI_ACTIVE_ACCTS', 'PRI_OVERDUE_ACCTS',

       'PRI_CURRENT_BALANCE', 'PRI_SANCTIONED_AMOUNT', 'PRI_DISBURSED_AMOUNT',

       'SEC_NO_OF_ACCTS', 'SEC_ACTIVE_ACCTS', 'SEC_OVERDUE_ACCTS',

       'SEC_CURRENT_BALANCE', 'SEC_SANCTIONED_AMOUNT', 'SEC_DISBURSED_AMOUNT',

       'PRIMARY_INSTAL_AMT', 'SEC_INSTAL_AMT', 'NEW_ACCTS_IN_LAST_SIX_MONTHS',

       'DELINQUENT_ACCTS_IN_LAST_SIX_MONTHS', 'NO_OF_INQUIRIES',

       'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag',

       'Driving_flag', 'Passport_flag', 'Employment_Type_Salaried',

       'Employment_Type_Self employed',

        'State_ID', 'Employee_code_ID',

                'branch_id','supplier_id','manufacturer_id','Current_pincode_ID']



# 'PERFORM_CNS_SCORE_DESCRIPTION','Aadhar_flag', 'PAN_flag', 'VoterID_flag','Employment_Type',



y = train_data['loan_default']

    

X = merged_df_predictors_train[feature_names]

X_test = merged_df_predictors_test[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=None,

                                  max_features='auto',

                                  max_leaf_nodes=None,

                                  min_samples_leaf=1,

                                  min_samples_split=2, min_weight_fraction_leaf=0.0,

                                  n_estimators=10, n_jobs=1, oob_score=False, random_state=None,

                                  verbose=0, warm_start=False).fit(train_X, train_y)

y_pred = my_model.predict(val_X)
merged_df_predictors_train.columns
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
# import matplotlib.pyplot as plt

# n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

# train_results = []

# test_results = []

# for estimator in n_estimators:

#    my_model = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)

#    my_model.fit(train_X, train_y)

#    train_pred = my_model.predict(train_X)

#    false_positive_rate, true_positive_rate, thresholds = roc_curve(train_y, train_pred)

#    roc_auc = auc(false_positive_rate, true_positive_rate)

#    train_results.append(roc_auc)

#    y_pred = my_model.predict(val_X)

#    false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, y_pred)

#    roc_auc = auc(false_positive_rate, true_positive_rate)

#    test_results.append(roc_auc)

# from matplotlib.legend_handler import HandlerLine2D

# line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")

# line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")

# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

# plt.ylabel('AUC score')

# plt.xlabel('n_estimators')

# plt.show()

from sklearn.metrics import confusion_matrix

print('The accuracy of the Random forest classifier is {:.2f} out of 1 on training data'.format(my_model.score(train_X, train_y)))



from sklearn.metrics import confusion_matrix

print('The accuracy of the Random forest classifier is {:.2f} out of 1 on test data'.format(my_model.score(val_X, val_y)))

import xgboost as xgb

Xgb_model = xgb.XGBClassifier().fit(train_X, train_y)

y_pred_xgb = my_model.predict(val_X)

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, y_pred_xgb)

roc_auc_xgb = auc(false_positive_rate, true_positive_rate)

#roc_auc_xgb
# some testing

train_data.groupby('Employment_Type').loan_default.count()
train_data[['PERFORM_CNS_SCORE','PERFORM_CNS_SCORE_DESCRIPTION']]
# # Necessary imports: 

#  #cross_val_predict

# from sklearn import metrics

# from sklearn.model_selection import cross_val_score,cross_val_predict



# # Perform 6-fold cross validation

# scores = cross_val_score(my_model, train_X, train_y, cv=6)

# print('Cross-validated scores:', scores)

# # Necessary imports: 

#  #cross_val_predict

# from sklearn import metrics

# from sklearn.model_selection import cross_val_score,cross_val_predict



# # Perform 6-fold cross validation

# scores = cross_val_score(Xgb_model, train_X, train_y, cv=6)

# print('Cross-validated scores:', scores)

prediction_xgb=Xgb_model.predict(X_test) 
results_df_xgb=pd.DataFrame(prediction_xgb,columns=['predicted'])

get_final_prediction_xgb=pd.concat([sample_submission,results_df_xgb], axis=1)

get_final_prediction_xgb.drop(labels=['loan_default'], inplace=True, axis=1)

get_final_prediction_xgb.rename(columns={'predicted':'loan_default'}, inplace=True)

get_final_prediction_xgb.to_csv('Xgboost_Scaled.csv', index=False)
# #final_test_predictor = sc_X.transform(final_test_predictor)

# prediction_xgb=xgb_clf.predict(final_test_predictor) #now we pass the testing data to the trained algorithm



# from sklearn.metrics import confusion_matrix

# results_df_xgb=pd.DataFrame(prediction_xgb,columns=['predicted'])

# get_final_prediction_xgb=pd.concat([submission_df,results_df_xgb], axis=1)

# get_final_prediction_xgb.drop(labels=['is_promoted'], inplace=True, axis=1)

# get_final_prediction_xgb.rename(columns={'predicted':'is_promoted'}, inplace=True)

# # you could use any filename. We choose submission here

# get_final_prediction_xgb.to_csv('Xgboost_Scaled.csv', index=False)
# import xgboost as xgb

# from sklearn.model_selection import GridSearchCV



# xgb_model = xgb.XGBClassifier()

# optimization_dict = {'max_depth': [2,4,6],

#                      'n_estimators': [50,100,200]}



# model = GridSearchCV(xgb_model, optimization_dict, 

#                      scoring='accuracy', verbose=1)



# model.fit(train_X, train_y)

# print(model.best_score_)