# Libraries
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
import string
#import itertools
#from itertools import product
#setting working directory
#os.chdir("/home/srishti/Srishti Saha- backup/misc/personal/kickstarter_projects")
# read in data
kickstarters_2017 = pd.read_csv("../input/ks-projects-201801.csv")
kickstarters_2017.head()
#printing all summary of the kickstarter data
#this will give the dimensions of data set : (rows, columns)
print(kickstarters_2017.shape)
#columns and data types
print(kickstarters_2017.info())
#basic stats of columns
print(kickstarters_2017.describe())
#number of unique values in all columns
print(kickstarters_2017.nunique())
#Distribution of data across state
percent_success = round(kickstarters_2017["state"].value_counts() / len(kickstarters_2017["state"]) * 100,2)

print("State Percent: ")
print(percent_success)
#renaming column usd_pledged as there is no '_' in the actual dataset variable name
col_names_prev=list(kickstarters_2017)
col_names_new= ['ID',
 'name',
 'category',
 'main_category',
 'currency',
 'deadline',
 'goal',
 'launched',
 'pledged',
 'state',
 'backers',
 'country',
 'usd_pledged',
 'usd_pledged_real',
 'usd_goal_real']
kickstarters_2017.columns= col_names_new
#segregating the variables as categorical and constinuous
cat_vars=[ 'category', 'main_category', 'currency','country']
cont_vars=['goal', 'pledged', 'backers','usd_pledged','usd_pledged_real','usd_goal_real']
#correlation of continuous variables
kickstarters_2017[cont_vars].corr()
#setting unique ID as index of the table
#this is because the ID column will not be used in the algorithm. yet it is needed to identify the project
df_kick= kickstarters_2017.set_index('ID')
# Filtering only for successful and failed projects
kick_projects = df_kick[(df_kick['state'] == 'failed') | (df_kick['state'] == 'successful')]
#converting 'successful' state to 1 and failed to 0
kick_projects['state'] = (kick_projects['state'] =='successful').astype(int)
print(kick_projects.shape)
#checking distribution of projects across various main categories
kick_projects.groupby(['main_category','state']).size()
#kick_projects.groupby(['category','state']).size()
#correlation of continuous variables with the dependent variable
kick_projects[['goal', 'pledged', 'backers','usd_pledged','usd_pledged_real','usd_goal_real','state']].corr()
#creating derived metrics/ features

#converting the date columns from string to date format
#will use it to derive the duration of the project
kick_projects['launched_date'] = pd.to_datetime(kick_projects['launched'], format='%Y-%m-%d %H:%M:%S')
kick_projects['deadline_date'] = pd.to_datetime(kick_projects['deadline'], format='%Y-%m-%d %H:%M:%S')
kick_projects= kick_projects.sort_values('launched_date',ascending=True)
kick_projects.head()
#creating features from the project name

#length of name
kick_projects['name_len'] = kick_projects.name.str.len()

# presence of !
kick_projects['name_exclaim'] = (kick_projects.name.str[-1] == '!').astype(int)

# presence of !
kick_projects['name_question'] = (kick_projects.name.str[-1] == '?').astype(int)

# number of words in the name
kick_projects['name_words'] = kick_projects.name.apply(lambda x: len(str(x).split(' ')))

# if name is uppercase
kick_projects['name_is_upper'] = kick_projects.name.str.isupper().astype(float)
# normalizing goal by applying log
kick_projects['goal_log'] = np.log1p(kick_projects.goal)
#creating goal features to check what range goal lies in
kick_projects['Goal_10'] = kick_projects.goal.apply(lambda x: x // 10)
kick_projects['Goal_1000'] = kick_projects.goal.apply(lambda x: x // 1000)
kick_projects['Goal_100'] = kick_projects.goal.apply(lambda x: x // 100)
kick_projects['Goal_500'] = kick_projects.goal.apply(lambda x: x // 500)
#features from date column
kick_projects['duration']=(kick_projects['deadline_date']-kick_projects['launched_date']).dt.days
#the idea for deriving launched quarter month year is that perhaps projects launched in a particular year/ quarter/ month might have a low success rate
kick_projects['launched_quarter']= kick_projects['launched_date'].dt.quarter
kick_projects['launched_month']= kick_projects['launched_date'].dt.month
kick_projects['launched_year']= kick_projects['launched_date'].dt.year
kick_projects['launched_week']= kick_projects['launched_date'].dt.week
#additional features from goal, pledge and backers columns
kick_projects.loc[:,'goal_reached'] = kick_projects['pledged'] / kick_projects['goal'] # Pledged amount as a percentage of goal.
#The above field will be used to compute another metric
# In backers column, impute 0 with 1 to prevent undefined division.
kick_projects.loc[kick_projects['backers'] == 0, 'backers'] = 1 
kick_projects.loc[:,'pledge_per_backer'] = kick_projects['pledged'] / kick_projects['backers'] # Pledged amount per backer.
#will create percentile buckets for the goal amount in a category
kick_projects['goal_cat_perc'] =  kick_projects.groupby(['category'])['goal'].transform(
                     lambda x: pd.qcut(x, [0, .35, .70, 1.0], labels =[1,2,3]))

#will create percentile buckets for the duration in a category
kick_projects['duration_cat_perc'] =  kick_projects.groupby(['category'])['duration'].transform(
                     lambda x: pd.qcut(x, [0, .35, .70, 1.0], labels =False, duplicates='drop'))
#creating a metric to see number of competitors for a given project in a given quarter
#number of participants in a given category, that launched in the same year and quarter and in the same goal bucket
ks_particpants_qtr=kick_projects.groupby(['category','launched_year','launched_quarter','goal_cat_perc']).count()
ks_particpants_qtr=ks_particpants_qtr[['name']]
#since the above table has all group by columns created as index, converting them into columns
ks_particpants_qtr.reset_index(inplace=True)

#creating a metric to see number of competitors for a given project in a given month
#number of participants in a given category, that launched in the same year and month and in the same goal bucket
ks_particpants_mth=kick_projects.groupby(['category','launched_year','launched_month','goal_cat_perc']).count()
ks_particpants_mth=ks_particpants_mth[['name']]
#since the above table has all group by columns created as index, converting them into columns
ks_particpants_mth.reset_index(inplace=True)

#creating a metric to see number of competitors for a given project in a given week
#number of participants in a given category, that launched in the same year and week and in the same goal bucket
ks_particpants_wk=kick_projects.groupby(['category','launched_year','launched_week','goal_cat_perc']).count()
ks_particpants_wk=ks_particpants_wk[['name']]
#since the above table has all group by columns created as index, converting them into columns
ks_particpants_wk.reset_index(inplace=True)
#renaming columns of the derived table
colmns_qtr=['category', 'launched_year', 'launched_quarter', 'goal_cat_perc', 'participants_qtr']
ks_particpants_qtr.columns=colmns_qtr

colmns_mth=['category', 'launched_year', 'launched_month', 'goal_cat_perc', 'participants_mth']
ks_particpants_mth.columns=colmns_mth

colmns_wk=['category', 'launched_year', 'launched_week', 'goal_cat_perc', 'participants_wk']
ks_particpants_wk.columns=colmns_wk
#merging the particpants column into the base table
kick_projects = pd.merge(kick_projects, ks_particpants_qtr, on = ['category', 'launched_year', 'launched_quarter','goal_cat_perc'], how = 'left')
kick_projects = pd.merge(kick_projects, ks_particpants_mth, on = ['category', 'launched_year', 'launched_month','goal_cat_perc'], how = 'left')
kick_projects = pd.merge(kick_projects, ks_particpants_wk, on = ['category', 'launched_year', 'launched_week','goal_cat_perc'], how = 'left')
#creating 2 metrics to get average pledge per backer for a category in a year according to the goal bucket it lies in and the success rate ie average pledged to goal ratio for the category and goal bucket in this year
#using pledge_per_backer (computed earlier) and averaging it by category in a launch year
ks_ppb_goal=pd.DataFrame(kick_projects.groupby(['category','launched_year','goal_cat_perc'])['pledge_per_backer','goal_reached'].mean())
#since the above table has all group by columns created as index, converting them into columns
ks_ppb_goal.reset_index(inplace=True)
#renaming column
ks_ppb_goal.columns= ['category','launched_year','goal_cat_perc','avg_ppb_goal','avg_success_rate_goal']

#creating a metric: the success rate ie average pledged to goal ratio for the category in this year
ks_ppb_duration=pd.DataFrame(kick_projects.groupby(['category','launched_year','duration_cat_perc'])['goal_reached'].mean())
#since the above table has all group by columns created as index, converting them into columns
ks_ppb_duration.reset_index(inplace=True)
#renaming column
ks_ppb_duration.columns= ['category','launched_year','duration_cat_perc','avg_success_rate_duration']
#merging the particpants column into the base table
kick_projects = pd.merge(kick_projects, ks_ppb_goal, on = ['category', 'launched_year','goal_cat_perc'], how = 'left')
kick_projects = pd.merge(kick_projects, ks_ppb_duration, on = ['category', 'launched_year','duration_cat_perc'], how = 'left')
#creating 2 metrics: mean and median goal amount
median_goal_cat=pd.DataFrame(kick_projects.groupby(['category','launched_year','duration_cat_perc'])['goal'].median())
#since the above table has all group by columns created as index, converting them into columns
median_goal_cat.reset_index(inplace=True)
#renaming column
median_goal_cat.columns= ['category','launched_year','duration_cat_perc','median_goal_year']

mean_goal_cat=pd.DataFrame(kick_projects.groupby(['category','launched_year','duration_cat_perc'])['goal'].mean())
#since the above table has all group by columns created as index, converting them into columns
mean_goal_cat.reset_index(inplace=True)
#renaming column
mean_goal_cat.columns= ['category','launched_year','duration_cat_perc','mean_goal_year']
#merging the particpants column into the base table
kick_projects = pd.merge(kick_projects, median_goal_cat, on = ['category', 'launched_year','duration_cat_perc'], how = 'left')
kick_projects = pd.merge(kick_projects, mean_goal_cat, on = ['category', 'launched_year','duration_cat_perc'], how = 'left')
print(kick_projects.shape)
kick_projects[:3]
# replacing all 'N,0"' values in the country column with 'NZERO' to avoid discrepancies while one hot encoding
kick_projects = kick_projects.replace({'country': 'N,0"'}, {'country': 'NZERO'}, regex=True)
list(kick_projects)
#selecting the needed fields only
#this will lead to the final features list

#creating a list of columns to be dropped
drop_columns= ['name','launched','deadline','launched_date','deadline_date','pledged','backers','usd_pledged','usd_pledged_real','pledge_per_backer','goal_reached']
#dropping columns above
kick_projects.drop(drop_columns, axis=1, inplace=True)
#these functions will be used on the textual column entries to remove '&','-' or white spaces
def replace_ampersand(val):
    if isinstance(val, str):
        return(val.replace('&', 'and'))
    else:
        return(val)

def replace_hyphen(val):
    if isinstance(val, str):
        return(val.replace('-', '_'))
    else:
        return(val)    
    
def remove_extraspace(val):
        if isinstance(val, str):
            return(val.strip())
        else:
            return(val) 

def replace_space(val):
        if isinstance(val, str):
            return(val.replace(' ', '_'))
        else:
            return(val)         
#apply those functions to all cat columns
#this will remove special characters from the character columns.
#Since these fields will be one-hot encoded, the column names so derived should be compatible with the requied format
kick_projects['category'] = kick_projects['category'].apply(remove_extraspace)
kick_projects['category'] = kick_projects['category'].apply(replace_ampersand)
kick_projects['category'] = kick_projects['category'].apply(replace_hyphen)
kick_projects['category'] = kick_projects['category'].apply(replace_space)

kick_projects['main_category'] = kick_projects['main_category'].apply(remove_extraspace)
kick_projects['main_category'] = kick_projects['main_category'].apply(replace_ampersand)
kick_projects['main_category'] = kick_projects['main_category'].apply(replace_hyphen)
kick_projects['main_category'] = kick_projects['main_category'].apply(replace_space)
#missing value treatment
# Check for nulls.
kick_projects.isnull().sum()
#dropping all rows that have any nulls
kick_projects=kick_projects.dropna() 
# Check for nulls again.
kick_projects.isnull().sum()
#creating a backup copy of the dataset
kick_projects_copy= kick_projects.copy()

kick_projects_copy[:5]
for c in kick_projects.columns:
    #this gives us the list of columns and the respective data types
    col_type = kick_projects[c].dtype
    #looking through all categorical columns in the list above
    if col_type == 'object' :
        a=kick_projects[c].unique()
        keys= range(a.shape[0])
        #initiating a dictionary
        diction={}
        for idx,val in enumerate(a):
        #looping through to create the dictionary with mappings
            diction[idx] = a[idx]
        #the above step maps integers to the values in the column
        # hence inverting the key-value pairs
        diction = {v: k for k, v in diction.items()}
        print(diction)
        # creating a dictionary for mapping the values to integers
        kick_projects_copy[c] = [diction[item] for item in kick_projects_copy[c]] 
        # converting data type to 'category'
        kick_projects_copy[c] = kick_projects_copy[c].astype('category')
# One-Hot encoding to convert categorical columns to numeric
print('start one-hot encoding')

kick_projects_ip = pd.get_dummies(kick_projects, prefix = [ 'category', 'main_category', 'currency','country'],
                             columns = [ 'category', 'main_category', 'currency','country'])
    
#this will have created 1-0 flag columns (like a sparse matrix)    
print('ADS dummy columns made')
#creating 2 arrays: features and response

#features will have all independent variables
features=list(kick_projects_ip)
features.remove('state')
#response has the target variable
response= ['state']
#creating a backup copy of the input dataset
kick_projects_ip_copy= kick_projects_ip.copy()
kick_projects_ip[features].shape
# normalize the data attributes
kick_projects_ip_scaled_ftrs = pd.DataFrame(preprocessing.normalize(kick_projects_ip[features]))
kick_projects_ip_scaled_ftrs.columns=list(kick_projects_ip[features])
kick_projects_ip_scaled_ftrs[:3]
#kick_projects_ip[features].shape
#creating test and train dependent and independent variables
#Split the data into test and train (30-70: random sampling)
#will be using the scaled dataset to split 
train_ind, test_ind, train_dep, test_dep = train_test_split(kick_projects_ip_scaled_ftrs, kick_projects_ip[response], test_size=0.3, random_state=0)
from xgboost import XGBClassifier
from sklearn import model_selection
#def timer(start_time=None):
#    if not start_time:
#        start_time = datetime.now()
#        return start_time
#    elif start_time:
#        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
#        tmin, tsec = divmod(temp_sec, 60)
#        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
# defining the XGBoost model
xgb_model = XGBClassifier(
 n_estimators= 1200,
 learning_rate= 0.08,
 max_depth= 5,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic'
 )


# Tried doing a grid search but commented it out for the amount of time it takes
## Defining parameters
#n_estimators = [500,1000, 1200]
#learning_rate = [0.0001, 0.01,0.1, 0.3]
#param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)

## Starting stratified Kfold
#kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
#random_search = model_selection.RandomizedSearchCV(xgb_model, param_grid, scoring="neg_log_loss", n_jobs=4, cv=kfold.split(train_ind[features], train_dep[response]), n_iter=12)

## fitting the random search
#start_time = timer(None)
#random_result = random_search.fit(train_ind[features], train_dep[response])
#timer(start_time) # timing ends here for "start_time" variable
# model fitting
xgb_model=xgb_model.fit(train_ind[features], train_dep[response])
# Predict the on the train_data
test_ind["Pred_state_XGB_2"] = xgb_model.predict(test_ind[features])

# Predict the on the train_data
train_ind["Pred_state_XGB_2"] = xgb_model.predict(train_ind[features])

# Predict the on the train_data
kick_projects_ip["Pred_state_XGB_2"] = xgb_model.predict(kick_projects_ip_scaled_ftrs)
print ("Test Accuracy :: ",accuracy_score(test_dep[response], xgb_model.predict(test_ind[features])))
print ("Train Accuracy :: ",accuracy_score(train_dep[response], xgb_model.predict(train_ind[features])))
print ("Complete Accuracy  :: ",accuracy_score(kick_projects_ip[response], xgb_model.predict(kick_projects_ip_scaled_ftrs)))
print (" Confusion matrix of complete data is", confusion_matrix(kick_projects_ip[response],kick_projects_ip["Pred_state_XGB_2"]))
## Feature importances
ftr_imp=zip(features,xgb_model.feature_importances_)
for values in ftr_imp:
    print(values)
# creating a dataframe
feature_imp=pd.DataFrame(list(zip(features,xgb_model.feature_importances_)))
column_names= ['features','XGB_imp']
feature_imp.columns= column_names

# sort in descending order of importances
feature_imp= feature_imp.sort_values('XGB_imp',ascending=False)
feature_imp[:15]
from sklearn.ensemble import RandomForestClassifier
import math
features_count = train_ind.shape[1]

parameters_rf = {'n_estimators':[50], 'max_depth':[20], 'max_features': 
                     [math.floor(np.sqrt(features_count)), math.floor(features_count/3)]}

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators=50,criterion='gini' ,max_depth=20, max_features=2)
    clf.fit(features, target)
    return clf
trained_model_RF= random_forest_classifier(train_ind[features], train_dep[response])
# Predict the on the train_data
test_ind["Pred_state_RF"] = trained_model_RF.predict(test_ind[features])

# Predict the on the train_data
train_ind["Pred_state_RF"] = trained_model_RF.predict(train_ind[features])

# Predict the on the train_data
kick_projects_ip["Pred_state_RF"] = trained_model_RF.predict(kick_projects_ip_scaled_ftrs)
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(train_dep[response], trained_model_RF.predict(train_ind[features])))
print ("Test Accuracy  :: ", accuracy_score(test_dep[response], trained_model_RF.predict(test_ind[features])))
print ("Complete Accuracy  :: ", accuracy_score(kick_projects_ip[response], trained_model_RF.predict(kick_projects_ip_scaled_ftrs)))
print (" Confusion matrix of complete data is", confusion_matrix(kick_projects_ip[response],kick_projects_ip["Pred_state_RF"]))
## Feature importances
ftr_imp_rf=zip(features,trained_model_RF.feature_importances_)
for values in ftr_imp_rf:
    print(values)
feature_imp_RF=pd.DataFrame(list(zip(features,trained_model_RF.feature_importances_)))
column_names_RF= ['features','RF_imp']
feature_imp_RF.columns= column_names_RF
feature_imp_RF= feature_imp_RF.sort_values('RF_imp',ascending=False)
feature_imp_RF[:15]
import lightgbm as lgb
#create LGBM classifier model
gbm_model = lgb.LGBMClassifier(
        boosting_type= "dart",
        n_estimators=1300,
        learning_rate=0.08,
        num_leaves=35,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=9,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
)

# LGBM with one-hot encoded features
#fit the model on training data
gbm_model=gbm_model.fit(train_ind[features], 
            train_dep[response], 
              verbose=0)
# Predict the on the train_data
test_ind["Pred_state_LGB"] = gbm_model.predict(test_ind[features])

# Predict the on the train_data
train_ind["Pred_state_LGB"] = gbm_model.predict(train_ind[features])

# Predict the on the train_data
kick_projects_ip["Pred_state_LGB"] = gbm_model.predict(kick_projects_ip_scaled_ftrs)
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(train_dep[response], gbm_model.predict(train_ind[features])))
print ("Test Accuracy  :: ", accuracy_score(test_dep[response], gbm_model.predict(test_ind[features])))
print ("Complete Accuracy  :: ", accuracy_score(kick_projects_ip[response], gbm_model.predict(kick_projects_ip_scaled_ftrs)))
print (" Confusion matrix of complete data is", confusion_matrix(kick_projects_ip[response],kick_projects_ip["Pred_state_LGB"]))
# classification matrix
print('\nClassification metrics')
print(classification_report(y_true=test_dep[response], y_pred=test_ind["Pred_state_LGB"]))
## Feature importances
ftr_imp_lgb=zip(features,gbm_model.feature_importances_)

for values in ftr_imp_lgb:
    print(values)
feature_imp_lgb=pd.DataFrame(list(zip(features,gbm_model.feature_importances_)))
column_names_lgb= ['features','LGB_imp']
feature_imp_lgb.columns= column_names_lgb

feature_imp_lgb= feature_imp_lgb.sort_values('LGB_imp',ascending=False)
feature_imp_lgb
#creating features and response list
features_2=list(kick_projects_copy)
features_2.remove('state')
features_2_numerical = [e for e in features_2 if e not in ('category','main_category','country','currency')]
features_2_categorical = ['category','main_category','country','currency']
response = ['state']
# Assuming same lines from your example
cols_to_norm = features_2_numerical
kick_projects_copy[cols_to_norm] = kick_projects_copy[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#creating test and train dependent and independent variables
#Split the data into test and train (30-70: random sampling)
#will be using the scaled dataset to split 
train_ind_2, test_ind_2, train_dep_2, test_dep_2 = train_test_split(kick_projects_copy[features_2],kick_projects_copy[response], test_size=0.3, random_state=0)
#create LGBM classifier model
gbm_model_2 = lgb.LGBMClassifier(
        boosting_type= "dart",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=38,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=9,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
)

# LGBM with one-hot encoded features
#fit the model on training data
gbm_model_2=gbm_model_2.fit(train_ind_2[features_2], 
            train_dep_2[response], 
            feature_name=features_2,
            categorical_feature= features_2_categorical,                
              verbose=0)
# Predict the on the train_data
test_ind_2["Pred_state_LGB"] = gbm_model_2.predict(test_ind_2[features_2])

# Predict the on the train_data
train_ind_2["Pred_state_LGB"] = gbm_model_2.predict(train_ind_2[features_2])

# Predict the on the train_data
kick_projects_copy["Pred_state_LGB"] = gbm_model_2.predict(kick_projects_copy[features_2])
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(train_dep_2[response], gbm_model_2.predict(train_ind_2[features_2])))
print ("Test Accuracy  :: ", accuracy_score(test_dep_2[response], gbm_model_2.predict(test_ind_2[features_2])))
print ("Complete Accuracy  :: ", accuracy_score(kick_projects_copy[response], gbm_model_2.predict(kick_projects_copy[features_2])))
print (" Confusion matrix of complete data is", confusion_matrix(kick_projects_copy[response],kick_projects_copy["Pred_state_LGB"]))
# classification matrix
print('\nClassification metrics')
print(classification_report(y_true=test_dep_2[response], y_pred=gbm_model_2.predict(test_ind_2[features_2])))
## Feature importances
ftr_imp_lgb_2=zip(features_2,gbm_model_2.feature_importances_)

for values in ftr_imp_lgb_2:
    print(values)
# creating a dataframe to get top features
feature_imp_lgb_2=pd.DataFrame(list(zip(features_2,gbm_model_2.feature_importances_)))
column_names_lgb_2= ['features','LGB_imp_2']
feature_imp_lgb_2.columns= column_names_lgb_2

feature_imp_lgb_2= feature_imp_lgb_2.sort_values('LGB_imp_2',ascending=False)
feature_imp_lgb_2
class LGBMClassifier_GainFE(lgb.LGBMClassifier):
    @property
    def feature_importances_(self):
        if self._n_features is None:
            raise LGBMNotFittedError('No feature_importances found. Need to call fit beforehand.')
        return self.booster_.feature_importance(importance_type='gain')
# defining parameters
lgb_gain = LGBMClassifier_GainFE(boosting_type= "dart",
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=38,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=9,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01)
#fitting the model
lgb_gain.fit(train_ind_2[features_2], 
            train_dep_2[response], 
            feature_name=features_2,
            categorical_feature= features_2_categorical,                
              verbose=0)
# Predict the on the train_data
test_ind_2["Pred_state_LGB_Gain"] = lgb_gain.predict(test_ind_2[features_2])

# Predict the on the train_data
train_ind_2["Pred_state_LGB_Gain"] = lgb_gain.predict(train_ind_2[features_2])

# Predict the on the train_data
kick_projects_copy["Pred_state_LGB_Gain"] = lgb_gain.predict(kick_projects_copy[features_2])
# Train and Test Accuracy
print ("Train Accuracy :: ", accuracy_score(train_dep_2[response], lgb_gain.predict(train_ind_2[features_2])))
print ("Test Accuracy  :: ", accuracy_score(test_dep_2[response], lgb_gain.predict(test_ind_2[features_2])))
print ("Complete Accuracy  :: ", accuracy_score(kick_projects_copy[response], lgb_gain.predict(kick_projects_copy[features_2])))
print (" Confusion matrix of complete data is", confusion_matrix(kick_projects_copy[response],kick_projects_copy["Pred_state_LGB_Gain"]))
## Feature importances
ftr_imp_lgb_gain=zip(features_2,lgb_gain.feature_importances_)

for values in ftr_imp_lgb_gain:
    print(values)
# creating a dataframe to get top 15 features
ftr_imp_lgb_gain=pd.DataFrame(list(zip(features_2,lgb_gain.feature_importances_)))
column_names_lgb_gain= ['features','LGB_gain_imp']
ftr_imp_lgb_gain.columns= column_names_lgb_gain

ftr_imp_lgb_gain= ftr_imp_lgb_gain.sort_values('LGB_gain_imp',ascending=False)
ftr_imp_lgb_gain[:15]
from sklearn import tree
from sklearn import neighbors
import math
#creating 4 models for ensembling: Decision Tree (using gini and entropy), knn and Logistic Regression
model_dtc_g = tree.DecisionTreeClassifier()
model_dtc_e = tree.DecisionTreeClassifier(criterion="entropy")
model_knn = neighbors.KNeighborsClassifier()
model_lr= LogisticRegression(penalty='l1',solver='saga')
#fitting each of the model above
model_dtc_g.fit(train_ind[features], train_dep[response])
model_dtc_e.fit(train_ind[features], train_dep[response])
model_knn.fit(train_ind[features], train_dep[response])
model_lr.fit(train_ind[features], train_dep[response])

#predicting the probabilities
pred_dtc_g=model_dtc_g.predict_proba(test_ind[features])
pred_dtc_e=model_dtc_e.predict_proba(test_ind[features])
pred_knn=model_knn.predict_proba(test_ind[features])
pred_lr=model_lr.predict_proba(test_ind[features])

#averaging the 4 predictions above
finalpred=(pred_dtc_g+pred_dtc_e+pred_knn+pred_lr)/4
#creating the dataframe with predicted probabilities (for 0 and 1)
pred_proba_avg=pd.DataFrame(finalpred)
#the results have 2 probabilities: prob of the state being 0 and state being 1 in that order: hence the 2 columns
col_names=['prob_0','prob_1']
pred_proba_avg.columns=col_names
# if the probability of 0> probability of 1: state is 0 and vice versa
def final_state(c):
    if c['prob_0'] >c['prob_1']:
        return 0
    else:
        return 1
#creating the final predicted state column using the averaging method    
pred_proba_avg['final_state_avg'] = pred_proba_avg.apply(final_state, axis=1)
#appending to base dataframe
test_ind = test_ind.reset_index(drop=True)
pred_proba_avg = pred_proba_avg.reset_index(drop=True)
test_ind=pd.concat([test_ind,pred_proba_avg],axis=1)
print ("Test Accuracy  :: ", accuracy_score(test_dep[response],test_ind['final_state_avg']))
from sklearn.ensemble import AdaBoostClassifier
#creating the ADA Boost classifier using XGBoost
model_ada = AdaBoostClassifier(random_state=1)
model_ada.fit(train_ind[features], train_dep[response])

#accuracy score
model_ada.score(test_ind[features],test_dep[response])