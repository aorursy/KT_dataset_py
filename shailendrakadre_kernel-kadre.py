import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# for the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# for feature engineering
from feature_engine import missing_data_imputers as mdi
from feature_engine import discretisers as dsc
from feature_engine import categorical_encoders as ce
from imblearn.over_sampling import SMOTE
# Data has 2260668 observations so choosing a sample. Common home computers may give memory error with 2260668 rows
# Taking a small representative random sample. If you have better machines, you can increase the sample size



import random

# Count the lines in origenal data 'loan.csv'
num_lines = 2260668

# Sample size 
size = int(num_lines /110)

# The row indices to skip - make sure 0 is not included to keep the header!
skip_idx = random.sample(range(1, num_lines), num_lines - size)

# Read the data
data = pd.read_csv("loan.csv", skiprows=skip_idx)

df = data.copy()
print(df.shape)
print(df.columns)
df.dtypes
# We will print the first few of df little later to get a feel of the dataset
# Descrete variables
discrete = [var for var in df.columns if df[var].dtype!='O' and var!='loan_status' and df[var].nunique()<10]
# Continuous
continuous = [var for var in df.columns if df[var].dtype!='O' and var!='loan_status' and var not in discrete]

discrete
continuous
# Categorical variables. Date included 
categorical = [var for var in df.columns if df[var].dtype=='O' ]
categorical
# Print first few rows
df.head()
# Shape of the dataset
df.shape

# Date variables are seperated 
dates = []
for i in categorical:
    if (re.search("_d$", i) != None) or (re.search("_date$", i)):
        dates.append(i)
print (dates)
   
dates
# Pure categorical variables without date variables  
pure_categorical = [var for var in categorical if var not in dates]
pure_categorical
# Number of each variable types in the origenal loan.csv
print('There are {} discrete variables'.format(len(discrete)))
print('There are {} continuous variables'.format(len(continuous)))
print('There are {} categorical variables'.format(len(pure_categorical)))
print('There are {} date variables'.format(len(dates)))
# Missing Variables
# Number of missing variables

df.isnull().sum()
# % of missing variables
Null_Percent = df.isnull().mean()
df_null_percent = pd.DataFrame(Null_Percent, columns=['percent_null'])
df_null_percent['percent_null'][df_null_percent['percent_null']>0]
# Percentage of null in pure_categorical columns. Columns above 90% null values 
pure_cat_null_majo = pd.DataFrame(df[pure_categorical].isnull().mean()*100, columns=['cat_null_90'])
pure_cat_null_majo_clms = pure_cat_null_majo['cat_null_90'][pure_cat_null_majo['cat_null_90']>90]
lst_cat_90 = pure_cat_null_majo_clms.index
print(pure_cat_null_majo_clms)
print(lst_cat_90)
# Percentage of null in discrete columns. Columns above 90% null values 
pure_dis_null_majo = pd.DataFrame(df[discrete].isnull().mean()*100, columns=['dis_null_90'])
pure_dis_null_majo_clms = pure_dis_null_majo['dis_null_90'][pure_dis_null_majo['dis_null_90']>90]
lst_dis_90 = pure_dis_null_majo_clms.index
print(pure_dis_null_majo_clms)
print(lst_dis_90)
# Percentage of null in continuous columns. Columns above 90% null values 
pure_con_null_majo = pd.DataFrame(df[continuous].isnull().mean()*100, columns=['con_null_90'])
pure_con_null_majo_clms = pure_con_null_majo['con_null_90'][pure_con_null_majo['con_null_90']>90]
pure_con_null_majo_clms
lst_con_90 = pure_con_null_majo_clms.index
print(pure_con_null_majo_clms)
print(lst_con_90)
# Percentage of null in continuous columns. Columns above 90% null values 
pure_date_null_majo = pd.DataFrame(df[dates].isnull().mean()*100, columns=['date_null_90'])
pure_date_null_majo_clms = pure_date_null_majo['date_null_90'][pure_date_null_majo['date_null_90']>90]
pure_date_null_majo_clms
lst_date_90 = pure_date_null_majo_clms.index
print(pure_con_null_majo_clms)
print(lst_date_90)
# Drop all the columns from the origenal dataframe, which are null fo > 90% of the rows
df.drop(list(lst_dis_90), axis=1, inplace=True)
df.drop(list(lst_date_90), axis=1, inplace=True)
df.drop(list(lst_cat_90), axis=1, inplace=True)
df.drop(list(lst_con_90), axis=1, inplace=True)
# Shape of df after dropping all columns with above 90% null values 
df.shape
# After dropping all columns > 90% null, we are left with 107 columns. Origenally we had 145 columns
# Category types and their % in y. In this attempt we are converting the problem into a simple 0 ot 1 type
# cllassification problem
df['loan_status'].value_counts()/len(df['loan_status'])
# Preparing y for categorical predictions
loan_status_mapping_dict = {'Fully Paid' : 0, 
                            'Current' : 0,
                            'Charged Off': 1,
                            'Late (31-120 days)': 0,
                            'In Grace Period'  : 0,
                            'Late (16-30 days)': 0,
                            'Does not meet the credit policy. Status:Fully Paid':0,
                            'Does not meet the credit policy. Status:Charged Off':0,
                            'Default':1
                            }

loan_status_mapping_dict
# Map load_status
df['loan_status_mapped'] = df['loan_status'].map(loan_status_mapping_dict)
# Percentage of default and no default
df['loan_status_mapped'].value_counts()/len(df['loan_status_mapped'])
# Length of y
print(len(df['loan_status_mapped']))
# 88% are 0s, so it's a highly unblalanced y. We have to apply special techniques to balance these two classes in y
# We will do sampling only after treating missing values and categorical encoding


# Dividing the dataset into train and test with 30% as test
X_train, X_test, y_train, y_test = train_test_split(df.drop(['loan_status','loan_status_mapped'], axis=1), df['loan_status_mapped'], test_size = 0.3, random_state = 101) 
  
# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape) 
print('\n')
print("Before OverSampling, counts of label '1' in train: {}".format(sum(y_train == 1))) 
print("Before OverSampling, counts of label '0' in train: {} \n".format(sum(y_train == 0))) 
  
# Seperate variable types in X_train
dis_train = [var for var in discrete if var not in lst_dis_90]
con_train = [var for var in continuous if var not in lst_con_90]
pure_categorical_train = [var for var in pure_categorical if var not in lst_cat_90]
dates_train = [var for var in dates if var not in lst_date_90]

print('Discrete variables in train dataset', len(dis_train),'   ' , dis_train)
print('\n')
print('Continous variables in train dataset', len(con_train),'   ' ,con_train)
print('\n')
print('Categorical variables in train dataset', len(pure_categorical_train), '   ' ,pure_categorical_train)
print('\n')
print('Date variables in train dataset', len(dates_train),'   ' ,dates_train)
# Finding rare categories in pure categorical variables. Continued in the following cells
X_train.shape
X_train.columns
X_train[['term', 'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership', 'verification_status', 'pymnt_plan', 'purpose', 'title', 'zip_code', 'addr_state', 'earliest_cr_line', 'initial_list_status', 'application_type', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag']].head()

X_train[['term', 'grade', 'sub_grade', 'emp_title', 'emp_length', 
         'home_ownership', 'verification_status', 'pymnt_plan', 
         'purpose', 'title', 'zip_code', 'addr_state', 'earliest_cr_line', 
         'initial_list_status', 'application_type', 'hardship_flag', 
         'disbursement_method', 'debt_settlement_flag']].nunique()

# There are very high number of ctegories in some of the variables 
#sub_grade, emp_title, purpose, title, zip_code, addr_state, earliest_cr_line

# later we will classify strings occuring <4% as rare category

# Finding category percentages of each category in each of these features
def FindCategoryFractions(var):
     var_fraction = ((X_train[var].value_counts() / len(X_train)).sort_values())*100

     return var_fraction
     

FindCategoryFractions('sub_grade')
FindCategoryFractions('emp_title')
FindCategoryFractions('purpose')
FindCategoryFractions('title')
# zip_code, addr_state, earliest_cr_line are the remaining tiles
FindCategoryFractions('zip_code')
FindCategoryFractions('addr_state')
FindCategoryFractions('earliest_cr_line')
# Dropping the columns with very high cardinality from train and test. Total three columns are being dropped 
X_train.drop(['earliest_cr_line','zip_code','emp_title'], axis=1, inplace=True)
X_test.drop(['earliest_cr_line','zip_code','emp_title'], axis=1, inplace=True)
# Reminding train and test shape
print('Train Shape', X_train.shape)
print('Test Shape', X_test.shape)
# Cardinality is still high in the features sub_grade, purpose, title, addr_state but still managiable
# We will engineer these features by labelling <4% occuring categories as rare
#Before that we do some more necessary steps
# Let's have a look at the descrete variables 
# Discrete variables in train dataset 4     ['policy_code', 'acc_now_delinq', 'num_tl_120dpd_2m', 'num_tl_30dpd']
X_train[ ['policy_code', 'acc_now_delinq', 'num_tl_120dpd_2m', 'num_tl_30dpd']].head()
X_train[ ['policy_code', 'acc_now_delinq', 'num_tl_120dpd_2m', 'num_tl_30dpd']].tail()
# Looks like all the descrete variables contain numerics and some null values.
# After imputing the null values, they should be ready for ML models
# Let's look at the null value treatment now for X_Train
X_Train_null_features = pd.DataFrame(X_train.isnull().mean(), columns = ['X_train_null'])
X_Train_null_features
#X_Train_null_features['X_train_null'][X_Train_null_features['X_train_null']>0]
X_Train_null_features_lst = list(X_Train_null_features['X_train_null'][X_Train_null_features['X_train_null']>0].index)
X_Train_null_features_lst
# The following columns have some null values in it
# Seperating features with null values as per their data type
X_Train_null_cat = [var for var in X_Train_null_features_lst if var in pure_categorical_train]
X_Train_null_dis = [var for var in X_Train_null_features_lst if var in dis_train]
X_Train_null_con = [var for var in X_Train_null_features_lst if var in con_train]
X_Train_null_dates = [var for var in X_Train_null_features_lst if var in dates_train]
# Category columns we are left with that have some null values
X_Train_null_cat
# Treating date variables. Continued in the following cells
X_train[dates_train].info()
# Converting all date objects into date-time nvariables 
X_train['issue_d'] = pd.to_datetime(X_train['issue_d'])
X_train['last_pymnt_d'] = pd.to_datetime(X_train['last_pymnt_d'])
X_train['next_pymnt_d'] = pd.to_datetime(X_train['next_pymnt_d'])
X_train['last_credit_pull_d'] = pd.to_datetime(X_train['last_credit_pull_d'])

X_test['issue_d'] = pd.to_datetime(X_test['issue_d'])
X_test['last_pymnt_d'] = pd.to_datetime(X_test['last_pymnt_d'])
X_test['next_pymnt_d'] = pd.to_datetime(X_test['next_pymnt_d'])
X_test['last_credit_pull_d'] = pd.to_datetime(X_test['last_credit_pull_d'])
X_test[dates_train].info()
X_train[dates_train].head()
X_train['last_credit_pull_d'].value_counts()
# Create new features by extracting quarter and year from the date time variables. 

X_train['issue_d_qtr']= X_train['issue_d'].dt.quarter
X_train['issue_d_year']= X_train['issue_d'].dt.year

X_train['last_pymnt_d_qtr']= X_train['last_pymnt_d'].dt.quarter
X_train['last_pymnt_d_year']= X_train['last_pymnt_d'].dt.year

X_train['next_pymnt_d_qtr']= X_train['next_pymnt_d'].dt.quarter
X_train['next_pymnt_d_year']= X_train['next_pymnt_d'].dt.year

X_train['last_credit_pull_d_qtr']= X_train['last_credit_pull_d'].dt.quarter
X_train['last_credit_pull_d_year']= X_train['last_credit_pull_d'].dt.year

# Date variables in train and test ['issue_d','last_pymnt_d','next_pymnt_d','last_credit_pull_d']
X_test['issue_d_qtr']= X_test['issue_d'].dt.quarter
X_test['issue_d_year']= X_test['issue_d'].dt.year

X_test['last_pymnt_d_qtr']= X_test['last_pymnt_d'].dt.quarter
X_test['last_pymnt_d_year']= X_test['last_pymnt_d'].dt.year

X_test['next_pymnt_d_qtr']= X_test['next_pymnt_d'].dt.quarter
X_test['next_pymnt_d_year']= X_train['next_pymnt_d'].dt.year

X_test['last_credit_pull_d_qtr']= X_test['last_credit_pull_d'].dt.quarter
X_test['last_credit_pull_d_year']= X_test['last_credit_pull_d'].dt.year

# Drop the origenal date variables from X_Train and X_test
date_var_drop =  ['issue_d','last_pymnt_d','next_pymnt_d','last_credit_pull_d']
X_train.drop(date_var_drop, axis=1, inplace=True)
X_test.drop(date_var_drop, axis=1, inplace=True)
# New date deriaved variables - ['issue_d_qtr','issue_d_year','last_pymnt_d_qtr','last_pymnt_d_year','next_pymnt_d_qtr','next_pymnt_d_year','last_credit_pull_d_qtr','last_credit_pull_d_year']
# Random check
X_test['last_credit_pull_d_qtr'].head()

X_test['last_credit_pull_d_year'].tail()
X_train.head()
# Newly derieved variables from original train and test date objects
date_derived_car = ['issue_d_qtr','issue_d_year','last_pymnt_d_qtr','last_pymnt_d_year','next_pymnt_d_qtr','next_pymnt_d_year','last_credit_pull_d_qtr','last_credit_pull_d_year']
X_train[date_derived_car].head()
# Imputing null places in date_derived_car 
X_train[date_derived_car].isnull().sum()
date_derived_car_with_null =  ['last_pymnt_d_qtr','last_pymnt_d_year', 'next_pymnt_d_qtr','next_pymnt_d_year']
# These are the variables with some null values in it
# Transforming imputations for missing data. Let's have a relook at the shates of train and test
print(X_train.shape)
print(X_test.shape)
# Preparing to inpute with Feature-Engine library.
# If you have not done already, you need to install it  
pipe = Pipeline([('imuter_discrete', mdi.ArbitraryNumberImputer(arbitrary_number=-1, variables=X_Train_null_dis)),
                 ('imuter_continuous', mdi.ArbitraryNumberImputer(arbitrary_number=-1, variables=X_Train_null_con)),
                 ('imuter_categorical',mdi.CategoricalVariableImputer(variables=X_Train_null_cat)),
                 ('imputer_dates', mdi.ArbitraryNumberImputer(arbitrary_number=-1, variables=date_derived_car_with_null))
                ])
#Fit X_train to this newly created imputing pipe
pipe.fit(X_train)
#Now we can thansform X_train using this pipe object. It will fill up the missing values for us
temp = pipe.transform(X_train)
temp.shape
# Check for null values in tnanformed train dataset
temp.isnull().sum()
########## CHEERS IT LOOKS LIKE WE HAVE NO NULL VALUES LEFT IN ANY OF COLUMNS. EVERYTHING WENT AS PLANNED ############
# WE CAN PERFORM ADDITIONAL STEPS TO CHECK IF THERE ARE STILL ANY COLUMNS WITH NULL VALUES
# BUT WE WILL STOP HERE AS ABOVE OUTPUT SHOWS NO APPARENT NULL VALUES
# Imputing for missing variables in X_test
temp_test = pipe.transform(X_test)
temp_test.shape
# Check for null values in tnanformed test dataset
temp_test_null_df = pd.DataFrame(temp_test.isnull().sum(), columns=['testnull'])
temp_test_null_df['testnull'][temp_test_null_df['testnull']>0]
# It will show columns with stiil having null values. There is one but with null vaules still
# Looks like strange. Only two columns are left with one null value each
# Let's investigate
X_test[['last_credit_pull_d_qtr','last_credit_pull_d_year']][X_test['last_credit_pull_d_qtr'].isnull()]
# the output is not shown below as I had rerum after deliting index with nulll row
# So when you will run from the beginnning it will definitely show up
# The single observation number 18553 has both the null values.
# It might be because of some case that is there in test and not come in train. It's just a speculation however
# For now I am going to delete the observation number 18553 from the test
X_test.drop(18553, inplace = True)
# Now looks like there are no null values left. Now let's have a relook at train 

temp_train_null_df = pd.DataFrame(temp.isnull().sum(), columns=['testnull'])
temp_train_null_df['testnull'][temp_train_null_df['testnull']>0]
# Now there are no null values left in test. Now let's have a relook at test
#temp.isnull().sum()
temp_test_null_df = pd.DataFrame(temp_test.isnull().sum(), columns=['testnull'])
temp_test_null_df['testnull'][temp_test_null_df['testnull']>0]
# X_train after transformation is represented by temp
# X_test after transformation is represented by temp_test
# In the above two cells, we have varified both do not have any null values left.
####### Both train and test imputing transformations are successfull ###### 
# Recheck shape after transformation
print(temp.shape)
print(temp_test.shape)
# Giving meaningful manes
X_train_imputed = temp
X_test_imputed = temp_test
# Recheck shape after transformation
print(X_train_imputed.shape)
print(X_test_imputed.shape)
#Now let's trun towards categorical variables for rare labels and encoding 
# [sub_grade, emp_length, purpose, addr_state] stiil have high cardinality 
# Let's have a relook at their cardinality again
X_train_imputed[['sub_grade', 'emp_length', 'purpose', 'addr_state']].nunique()
X_train_dropped = ['loan_status', 'earliest_cr_line', 'zip_code', 'emp_title']
X_train_remaining_categorical = [var for var in pure_categorical_train if var not in X_train_dropped]

# we will use X_train_remaining_categorical for categorical encoding. 
# These are the categorical variables remaining in X_train after dropping several columns in earliee spets
# let's create a imputer pipe using feature engine library
encoding_pipe = Pipeline([('encoder_rare_label',
                                               ce.RareLabelCategoricalEncoder(tol=0.004,
                                               n_categories=6,
                                               variables=['sub_grade', 'emp_length', 'purpose', 'addr_state'])),
                           ('categorical_encoder',
                                               ce.OrdinalCategoricalEncoder(encoding_method='ordered',
                                               variables=X_train_remaining_categorical))
                        ])

encoding_pipe.fit(X_train_imputed, y_train)
# print(X_train_imputed.shape)
# print(X_test_imputed.shape)
# Transfor with this encoding pipe
# X_train = ordinal_enc.transform(X_train)
#pipe.fit(X_test_imputed)
X_train_imputed_encoded = encoding_pipe.transform(X_train_imputed)
#temp_test = pipe.transform(X_test)
# transform test
X_test_imputed_encoded = encoding_pipe.transform(X_test_imputed)
# Now date engineering and tranformation steps are complete
# Time to do some random sanity checks
print('Train final shape',X_train_imputed_encoded.shape)

print('Test final shape',X_test_imputed_encoded.shape)
X_train_imputed_encoded.head()
X_test_imputed_encoded.head()
#### NEXT STEPS ####

# Apparantly, on the surface, everything is numeric and there are no null values. Let's proceed with ML algorithms
# We will mainly use here Ramdom Forest & XGBoost, which are are not sensitive to outliers and do not need a normal -
# - distribution for every variable. XGBoost had given us the best results many times in the past when compared to some linear models

(y_train.value_counts())/len(y_train)
# in y_train 0 is coming 88% and 1 is coming only 11%
# Do the oversample to balance 1 in the train datasets
#from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 2) 
X_train_smote, y_train_smote= sm.fit_sample(X_train_imputed_encoded, y_train.ravel())


print('\n')
print('After OverSampling, the shape of train_X: {}'.format(X_train_smote.shape)) 
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_smote.shape)) 
print('\n')
print("After OverSampling, counts of label '1': {}".format(sum(y_train_smote == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_smote == 0))) 

# Origenal X_train and y_train side
print('Train size before oversampling,',X_train_imputed_encoded.shape)
# Origenal X_train and y_train side
print('Test size before oversampling,',X_test_imputed_encoded.shape)

import matplotlib.pyplot as plt
from sklearn import  metrics, model_selection
from xgboost.sklearn import XGBClassifier
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
# XGBOost parameters for tuning the algorithm to get optimum performance
# Let's work with some ramdomly chosen hyper parameters 
XGBmodel = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample = 0.1)
#XGBmodel = XGBClassifier()
# grid_search.fit(X, label_encoded_y)
# y_pred = model.predict(X_test)
# lr1.fit(X_train_res, y_train_res.ravel())
#predictions = lr1.predict(X_test) 
  
# print classification report 
#print(classification_report(y_test, predictions)) 
# Fit the model 
XGBmodel.fit(X_train_smote, y_train_smote.ravel())
# Make the predictions 
y_pred = XGBmodel.predict(X_test_imputed_encoded)
# Model performance
print(classification_report(y_test, y_pred)) 
print('\n')
print(confusion_matrix(y_test, y_pred )) 
# Test the accuracy on train data
y_pred_train = XGBmodel.predict(X_train_imputed_encoded)
print(classification_report(y_train, y_pred_train )) 
print('\n')
# onfusion_matrix(y_true, y_pred, labels=['Cat', 'Dog', 'Rabbit']) # confusion_matrix
print(confusion_matrix(y_train, y_pred_train )) 
# Link to learn one more about SMOTE based sampling
## https://www.geeksforgeeks.org/ml-handling-imbalanced-data-with-smote-and-near-miss-algorithm-in-python/
# Import necessary packages 
from numpy import loadtxt
from xgboost import plot_importance
import matplotlib.pyplot as plt
%matplotlib inline
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# Plot feature importance
# XGBClassifier.get_booster().get_score(importance_type= f)
f = 'gain'
#XGBClassifier.get_booster().get_score(importance_type= 'gain')

importance_dict = XGBmodel.get_booster().get_score(importance_type='gain')
from collections import OrderedDict 

importance_dict_sorted_values_rev = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
#importance_dict_sorted_values_acen = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1])}
importance_dict_sorted_values_rev

# Initialize limit  
N = 30
    
# Using items() + list slicing  
# Get first K items in dictionary  
sorted_features_by_gain = dict(list(importance_dict_sorted_values_rev.items())[0: N])  
sorted_features_by_gain
import matplotlib.pyplot as plt
fig= plt.figure(figsize=(40,40))
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 35}

plt.rc('font', **font)
#data = {'apple': 67, 'mango': 60, 'lichi': 58}
names = list(sorted_features_by_gain.keys())
values = list(sorted_features_by_gain.values())

#tick_label does the some work as plt.xticks()
plt.barh(range(len(sorted_features_by_gain)),values,tick_label=names)
plt.gca().invert_yaxis()
#plt.savefig('bar.png')

plt.show()


