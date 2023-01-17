!pip install rfpimp
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
from rfpimp import importances
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
import seaborn as sns

df_train = pd.read_csv('../input/janatahack-crosssell-prediction/train.csv', header=0)
df_test = pd.read_csv('../input/janatahack-crosssell-prediction/test.csv', header=0)

df_train['train_flag'] = 1
df_test['train_flag'] = 0

df_test['Response'] = -1
print(df_train.shape, df_test.shape)

df_data = pd.concat((df_train, df_test))
print(df_data.shape)
 
df_data.isnull().sum()
df_data['Policy_Sales_Channel'] = df_data['Policy_Sales_Channel'].astype(np.int) 
df_data['Region_Code'] = df_data['Region_Code'].astype(np.int)
df_data['Vintage'] = df_data['Vintage']/365
def policy_bin_fn(row):
    if row['Policy_Sales_Channel'] in [26,124,152,156,157,122,154,163,160,155,25,13,55,7,31,3,30,158,12,125,8,151,52,11,29,4,24,15,150,120,14,23,61,60,10,16,136,153,1,147]:
        return 1
    if row['Policy_Sales_Channel'] in [91,42,59,145,109,44,19,121,116,22,9,36,54,37,131,139,128,21,56,106,138,35,135,103,94,111,127,148,47,113,53,45,140,90,18,86,64,119]:
        return 2
    if row['Policy_Sales_Channel'] in [80,132,133,65,81,114,78,129,92,20,93,17,40,89,130,100,88,101,32,49,107,87,62,66,2,73,69,48,108,97,63,51,68,39,159,110,43,57]:
        return 3
    if row['Policy_Sales_Channel'] in [98,58,27,28,123,46,76,105,33,96,112,71,34,149,95,79,115,83,41,102,6,146,134,74,104,75,67,50,137,99,117,82,118,84,70,38,143,126,144]:
        return 4
    else:
        return 5
df_data = df_data.assign(Policy_Sales_bin = df_data.apply(policy_bin_fn,axis=1))

def policy_bin_2(row):
    if row['Policy_Sales_Channel'] in [38,28,19,4,23,51,24,7,18,3]:
        return 3
    if row['Policy_Sales_Channel'] in [35,39,52,29,41,40,5,	20,	11,	45,	1,	46,	48,	31,	33,	12,	8,	43,	14,	13,	47,	0,	32]:
        return 2
    else:
        return 1
df_data = df_data.assign(Policy_Sales_bin_ER = df_data.apply(policy_bin_2,axis=1))

def region_code_bin(row):
    if row['Region_Code'] in [28,8,41,46,29,3,11,15,30,35,33,36,18,47]:
        return 1
    if row['Region_Code'] in [50,45,39,48,6,37,7,14,38,13,24,12,21]:
        return 2
    if row['Region_Code'] in [23,2,4,10,9,19,43,32,20,27,31,26,17]:
        return 3
    if row['Region_Code'] in [0,40,5,49,16,34,1,25,22,42,44,52,51]:
        return 4
    else:
        return 5
    
df_data = df_data.assign(Region_code_bin = df_data.apply(region_code_bin,axis=1))

def region_code_bin2(row):
    if row['Region_Code'] in [43,	123,	27,	28,	36,	155,	163,	3,	121,	80,	81,	87,	101,	158,	90,	157,	31,	2,	68,	100,	154,	150,	106,	53,	136,	156,	4,	57,	25,	26,	44,	42,	59,	94,	10,	124,	17,	147,	56,	91,	122,	12,	62,	69,	54,	55,	45,	13,	49,	89,	23,	35,	40,	111,	145,	24,	78,	114,	47,	29,	86,	92,	103]:
        return 3
    if row['Region_Code'] in [125,	109,	116,	131,	7,	20,	58,	30,	52,	93,	148,	60,	14,	9,	39,	37,	138,	61,	32,	128,	110,	130,	139,	11,	135,	15,	16,	19]:
        return 2
    else:
        return 1
    
df_data = df_data.assign(Region_code_bin_ER = df_data.apply(region_code_bin2,axis=1))

df_data["Gender"].replace({"Male":0, "Female":1}, inplace=True)
df_data["Vehicle_Age"].replace({"< 1 Year": 0, "1-2 Year":1, "> 2 Years":2}, inplace=True)
df_data["Vehicle_Damage"].replace({"Yes": 0, "No":1}, inplace=True)

df_data['Age'].unique()
def Age_bin_fn(row):
    if row['Age'] <= 25:
        return 1
    elif row['Age'] > 25 and row['Age'] <= 36:
        return 2
    elif row['Age'] >36 and row['Age'] <= 50:
        return 3
    else:
        return 4

df_data = df_data.assign(Age_bin = df_data.apply(Age_bin_fn,axis=1))

def Premium_bin_fn(row):
    if row['Annual_Premium'] <= 24500:
        return 1
    elif row['Annual_Premium'] > 24500 and row['Annual_Premium'] <= 31700:
        return 2
    elif row['Annual_Premium'] > 31700 and row['Annual_Premium'] <= 39400:
        return 3
    else:
        return 4

df_data = df_data.assign(Premium_bin = df_data.apply(Premium_bin_fn,axis=1))       


def encode_AG(main_columns, sub_columns,df,aggregations):
    for main_column in main_columns:  
        for col in sub_columns:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = df[[col, main_column]]
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                df[new_col_name] = df[col].map(temp_df).astype('float32')
                
                print("'"+new_col_name+"'",', ',end='')

# FREQUENCY ENCODE TOGETHER
def encode_FE(df, cols):
    for col in cols:
        df1 = df[col]
        vc = df1.value_counts(normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df[nm] = df[col].map(vc)
        df[nm] = df[nm].astype('float32')
        print(nm,', ',end='')
# COMBINE FEATURES
def encode_CB(col1,col2,df):
    nm = col1+'_'+col2
    df[nm] = df[col1].astype(str)+'_'+df[col2].astype(str)
    le = LabelEncoder()
    df[nm] = le.fit_transform(df[nm])
    print(nm,', ',end='')
encode_FE(df_data,['Previously_Insured','Driving_License','Gender','Premium_bin','Age_bin','Region_code_bin','Policy_Sales_bin','Vehicle_Age','Region_Code','Policy_Sales_Channel'])
encode_AG(['Annual_Premium','Vintage'],['Age_bin','Vehicle_Damage','Vehicle_Age','Policy_Sales_bin','Region_code_bin'],df_data,
          aggregations=['mean','std','min','max'])
encode_AG(['Previously_Insured','Driving_License'],['Vehicle_Damage'],df_data,aggregations=['mean','std'])
encode_CB('Region_Code','Policy_Sales_Channel',df_data)
encode_CB('Vehicle_Damage','Vehicle_Age',df_data)
encode_CB('Previously_Insured','Vehicle_Damage_Vehicle_Age',df_data)
encode_CB('Region_Code_Policy_Sales_Channel','Vehicle_Damage_Vehicle_Age',df_data)

df_data['Avg_premium_per_age'] = (df_data.groupby(['Age'])['Annual_Premium'].transform('mean')*df_data['Age'])/df_data.groupby(['Age'])['id'].transform('nunique')
df_data['Avg_premium_per_age_agg'] = (df_data.groupby(['Age'])['Annual_Premium'].transform('mean'))/df_data['Age']
df_data['Age_Premium_cust_cnt'] = df_data.groupby(['Age','Premium_bin'])['id'].transform('nunique')
df_data['Vintage_Age_avgPremium'] = df_data.groupby(['Vintage','Age'])['Annual_Premium'].transform('nunique')
df_data['Vintage_Age_customer_cnt'] = df_data.groupby(['Vintage','Age'])['id'].transform('nunique')
df_data['Interaction_Vintage_Age'] = df_data['Vintage']*df_data['Age']
df_data['Interaction Vehicle dam and prev insured'] = df_data['Vehicle_Damage']*df_data['Previously_Insured']
df_data['Interaction Region code and Sales channel'] = df_data['Region_Code']* df_data['Policy_Sales_Channel']
df_data['Interaction Age Premium'] = df_data['Age']*df_data['Annual_Premium']
df_data['Multiple Interactions'] = df_data['Interaction Age Premium']*df_data['Interaction Region code and Sales channel']*df_data['Interaction Vehicle dam and prev insured']*df_data['Interaction_Vintage_Age']
df_data['Premium by Age'] = df_data['Annual_Premium'] / df_data['Age']
df_data['log_premium'] = np.log(df_data['Annual_Premium'])
df_data['log_Region_code'] = np.log(df_data['Region_Code'])
df_data['log_sales_channel'] = np.log(df_data['Policy_Sales_Channel'])
df_data['Channel_region_Avgpremium'] = df_data.groupby(['Region_Code','Policy_Sales_Channel'])['Annual_Premium'].transform('mean')
df_data['Age minus Vintage'] = df_data['Age'] - df_data['Vintage']
df_data['Channel_region_AvgVintage'] = df_data.groupby(['Region_Code','Policy_Sales_Channel'])['Vintage'].transform('mean')
df_data['Channel_region_cust_cnt']  =  df_data.groupby(['Region_Code','Policy_Sales_Channel'])['id'].transform('nunique')
df_data['vehicleDamage_region_cust_cnt']  =  df_data.groupby(['Region_Code','Vehicle_Damage'])['id'].transform('nunique')
df_data['Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_cust_cnt'] = df_data.groupby(['Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age'])['id'].transform('nunique')       
df_data['Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_premium'] =  df_data.groupby(['Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age'])['Annual_Premium'].transform('mean')                            
df_data['Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_avgAge'] =  df_data.groupby(['Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age'])['Age'].transform('mean')     
df_data['Sum of vars'] = np.log(df_data['Annual_Premium'] )+ df_data['Region_Code'] + df_data['Policy_Sales_Channel']+ df_data['Vehicle_Damage'] + df_data['Previously_Insured']

feature_cols1 = df_data.columns.tolist()
feature_cols1.remove('id')
feature_cols1.remove('Response')
feature_cols1.remove('train_flag')

label_col = 'Response'
print(feature_cols1)
#exclude_features = ['Vintage_Age_avgPremium',	'Age minus Vintage',	'Channel_region_Avgpremium',	'Annual_Premium',	'Region_Code','Vintage_Policy_Sales_bin_std','Vintage_Region_code_bin_mean','Vintage_Region_code_bin_std',	'Vintage_Vehicle_Age_std',	'log_Region_code',	'Interaction_Vintage_Age',	'Region_Code_FE',	'Vehicle_Damage_Vehicle_Age',	'Annual_Premium_Age_bin_mean',	'Channel_region_AvgVintage',	'Vehicle_Age',	'Previously_Insured_Vehicle_Damage_Vehicle_Age','Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_avgAge','Annual_Premium_Age_bin_min',	'Annual_Premium_Age_bin_max',	'Annual_Premium_Vehicle_Damage_min',	'Annual_Premium_Vehicle_Damage_max',	'Annual_Premium_Vehicle_Age_min',	'Annual_Premium_Vehicle_Age_max',	'Annual_Premium_Policy_Sales_bin_max',	'Annual_Premium_Region_code_bin_min',	'Annual_Premium_Region_code_bin_max',	'Vintage_Age_bin_min',	'Vintage_Age_bin_max',	'Vintage_Vehicle_Damage_min',	'Vintage_Vehicle_Damage_max',	'Vintage_Vehicle_Age_min',	'Vintage_Vehicle_Age_max','Vintage_Policy_Sales_bin_min','Vintage_Region_code_bin_min','Vintage_Region_code_bin_max']
#exclude_features = ['Vehicle_Age',	'Vintage_Vehicle_Damage_std',	'Vintage_Policy_Sales_bin_std',	'Vintage_Policy_Sales_bin_min',	'Vintage_Policy_Sales_bin_max',	'Vintage_Region_code_bin_mean',	'Vintage_Region_code_bin_std',	'Vintage_Region_code_bin_min',	'Vintage_Region_code_bin_max',	'Previously_Insured_Vehicle_Damage_mean',	'Previously_Insured_Vehicle_Damage_std',	'Driving_License_Vehicle_Damage_mean',	'Driving_License_Vehicle_Damage_std',	'Region_Code_Policy_Sales_Channel',	'Vehicle_Damage_Vehicle_Age',	'Previously_Insured_Vehicle_Damage_Vehicle_Age',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age',	'Avg_premium_per_age',	'Avg_premium_per_age_agg',	'Age_Premium_cust_cnt',	'Vintage_Age_avgPremium',	'Vintage_Age_customer_cnt',	'Interaction_Vintage_Age',	'Interaction Vehicle dam and prev insured',	'Interaction Region code and Sales channel',	'Interaction Age Premium',	'Multiple Interactions',	'Premium by Age',	'log_premium',	'log_Region_code',	'log_sales_channel',	'Channel_region_Avgpremium',	'Age minus Vintage',	'Channel_region_AvgVintage',	'Channel_region_cust_cnt',	'vehicleDamage_region_cust_cnt',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_cust_cnt',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_premium',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_avgAge',	'Sum of vars']
#exclude_features = ['Vintage_Age_bin_max',	'Vintage_Vehicle_Age_std',	'Vintage_Vehicle_Damage_max',	'Annual_Premium_Region_code_bin_max',	'Annual_Premium_Age_bin_min',	'Annual_Premium_Age_bin_std',	'Vintage_Vehicle_Age_min',	'Annual_Premium_Age_bin_max',	'Premium_bin',	'Previously_Insured_FE',	'Driving_License_FE',	'Vintage_Vehicle_Damage_min',	'Annual_Premium_Vehicle_Damage_min',	'Vehicle_Age',	'Vintage_Vehicle_Damage_std',	'Vintage_Policy_Sales_bin_std',	'Vintage_Policy_Sales_bin_min',	'Vintage_Policy_Sales_bin_max',	'Vintage_Region_code_bin_mean',	'Vintage_Region_code_bin_std',	'Vintage_Region_code_bin_min',	'Vintage_Region_code_bin_max',	'Previously_Insured_Vehicle_Damage_mean',	'Previously_Insured_Vehicle_Damage_std',	'Driving_License_Vehicle_Damage_mean',	'Driving_License_Vehicle_Damage_std',	'Region_Code_Policy_Sales_Channel',	'Vehicle_Damage_Vehicle_Age',	'Previously_Insured_Vehicle_Damage_Vehicle_Age',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age',	'Avg_premium_per_age',	'Avg_premium_per_age_agg',	'Age_Premium_cust_cnt',	'Vintage_Age_avgPremium',	'Vintage_Age_customer_cnt',	'Interaction_Vintage_Age',	'Interaction Vehicle dam and prev insured',	'Interaction Region code and Sales channel',	'Interaction Age Premium',	'Multiple Interactions',	'Premium by Age',	'log_premium',	'log_Region_code',	'log_sales_channel',	'Channel_region_Avgpremium',	'Age minus Vintage',	'Channel_region_AvgVintage',	'Channel_region_cust_cnt',	'vehicleDamage_region_cust_cnt',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_cust_cnt',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_premium',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_avgAge',	'Sum of vars']
#exclude_features = ['Previously_Insured_Vehicle_Damage_mean',	'Avg_premium_per_age',	'Sum of vars',	'Annual_Premium_Vehicle_Age_max',	'Vintage_Region_code_bin_std',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_cust_cnt',	'vehicleDamage_region_cust_cnt',	'Annual_Premium_Region_code_bin_std',	'Driving_License_Vehicle_Damage_std',	'Gender',	'Vintage_Age_avgPremium',	'Annual_Premium_Region_code_bin_mean',	'Vintage_Vehicle_Age_std',	'Age_bin_FE',	'Annual_Premium_Vehicle_Damage_std',	'Vintage_Age_bin_std',	'Age_bin',	'Multiple Interactions',	'Premium by Age',	'Channel_region_AvgVintage',	'Interaction Region code and Sales channel',	'Annual_Premium_Region_code_bin_max',	'log_Region_code',	'Channel_region_Avgpremium',	'Region_code_bin',	'Region_Code_Policy_Sales_Channel_Vehicle_Damage_Vehicle_Age_premium',	'Region_code_bin_FE',	'Gender_FE',	'Channel_region_cust_cnt',	'Age_Premium_cust_cnt',	'log_sales_channel',	'Vintage_Age_bin_mean',	'Vehicle_Damage_Vehicle_Age',	'Region_Code_Policy_Sales_Channel']
#exclude_features = ['Driving_License_Vehicle_Damage_mean',	'Policy_Sales_bin',	'Vintage_Age_customer_cnt',	'Gender_FE',	'Annual_Premium_Policy_Sales_bin_std',	'Annual_Premium_Vehicle_Damage_std',	'Region_code_bin_FE',	'Driving_License_Vehicle_Damage_std',	'Region_code_bin',	'Annual_Premium_Policy_Sales_bin_mean',	'Annual_Premium',	'Vintage_Vehicle_Age_mean',	'Policy_Sales_bin_ER',	'Annual_Premium_Vehicle_Age_std',	'log_premium',	'Vintage_Age_avgPremium',	'Vintage_Policy_Sales_bin_std',	'Policy_Sales_bin_FE',	'Annual_Premium_Vehicle_Age_max',	'Previously_Insured_Vehicle_Damage_mean',	'Vehicle_Damage_Vehicle_Age',	'Vintage_Vehicle_Damage_std',	'Annual_Premium_Vehicle_Age_mean']
exclude_features = ['Annual_Premium_Age_bin_mean',	'Channel_region_cust_cnt',	'Driving_License',	'Driving_License_FE',	'Vintage_Policy_Sales_bin_mean',	'Premium_bin_FE',	'Annual_Premium_Policy_Sales_bin_std',	'Policy_Sales_bin',	'Vintage_Policy_Sales_bin_std',	'Policy_Sales_bin_ER',	'Vintage',	'id',	'Annual_Premium_Age_bin_min',	'Annual_Premium_Policy_Sales_bin_max',	'Annual_Premium_Policy_Sales_bin_mean',	'Annual_Premium_Policy_Sales_bin_min',	'Annual_Premium_Region_code_bin_max',	'Annual_Premium_Region_code_bin_mean',	'Annual_Premium_Region_code_bin_min',	'Annual_Premium_Region_code_bin_std',	'Annual_Premium_Vehicle_Age_min',	'Annual_Premium_Vehicle_Damage_max',	'Annual_Premium_Vehicle_Damage_min',	'Policy_Sales_bin_FE',	'Region_code_bin_FE',	'Vehicle_Age_FE',	'Vintage_Age_bin_max',	'Vintage_Age_bin_min',	'Vintage_Policy_Sales_bin_max',	'Vintage_Policy_Sales_bin_min',	'Vintage_Region_code_bin_max',	'Vintage_Region_code_bin_min',	'Vintage_Vehicle_Age_max',	'Vintage_Vehicle_Age_min',	'Vintage_Vehicle_Age_std',	'Vintage_Vehicle_Damage_max',	'Vintage_Vehicle_Damage_min']
feature_cols = np.setdiff1d(feature_cols1,exclude_features)
df_data['Region_Code'] = df_data['Region_Code'].astype(np.int) 
df_train, df_test = df_data[df_data.train_flag == 1], df_data[df_data.train_flag == 0]

df_train.drop(['train_flag'], inplace=True, axis=1)
df_test.drop(['train_flag'], inplace=True, axis=1)
df_test.drop([label_col], inplace=True, axis=1)

print(df_train.shape, df_test.shape)
x=df_train[feature_cols]
y=df_train[label_col]

test = df_test[feature_cols]
print(x.shape,y.shape)
!pip install rfpimp
#over = SMOTE(sampling_strategy=0.16)
#under = RandomUnderSampler(sampling_strategy=0.35)
#steps = [('o', over), ('u', under)]
#pipeline = Pipeline(steps=steps)
# transform the dataset
#X_res, y_res = under.fit_resample(x, y)
#print(X_res.shape)
#print(y_res.shape)
# summarize the new class distribution
#counter = Counter(y_res)
#print(counter)

err_lgb = [] 
oof_lgb = []

err_xgb = [] 
oof_xgb = []

models_xgb = []
models = []
feature_importance_df = pd.DataFrame()
feature_importance_df['Feature'] = x.columns
#oof = np.empty((len(df_test), 10))
y_pred_tot_lgm = np.zeros((len(test), 1))
params = {}
params['learning_rate'] = 0.04
params['max_depth'] = 10
params['n_estimators'] = 250
params['objective'] = 'binary'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.9
params['random_state'] = 142
params['colsample_bytree']=0.5
params['min_data_in_leaf'] = 30
params['reg_alpha'] = 2.1
params['reg_lambda'] = 2.1
params['device']= 'gpu'
params['gpu_platform_id']= 0,
params['gpu_device_id']= 0
#params['is_unbalance'] = True
#params['max_bin'] = 64
#params['num_leaves'] = 1024
params['class_weight']: {0: 0.1, 1: 2.5}
#params['scale_pos_weight'] = 7.0

nfolds = 10

fold = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=12323)
i = 1

for train_index, test_index in fold.split(x, y):
    
    x_train, x_val = x.iloc[train_index], x.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    
    print('Fittting model for fold number'+str(i))
    clf = lgb.LGBMClassifier(**params)

    clf.fit(x_train, y_train,eval_set=[(x_val, y_val)],eval_metric='auc',categorical_feature= categorical_list, early_stopping_rounds=150,verbose=True)
    pred_lgb = clf.predict_proba(x_val)[:,1]
    ''''
    model_xgb = xgb.XGBClassifier(learning_rate =0.04,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=30,
     gamma=0,
     subsample=0.7,
     colsample_bytree=0.5,
     objective="binary:logistic",
     #nthread=32,
     scale_pos_weight=2.5,
     reg_alpha = 2.1,reg_lambda = 2.1,
                                  
     seed=47,tree_method = "gpu_hist",n_gpus= 1)
     '''
    
    #model_xgb.fit(x_train, y_train,eval_set=[(x_train,y_train),(x_val, y_val)],eval_metric='auc', early_stopping_rounds=100)
    #pred_xgb = model_xgb.predict_proba(x_val)[:,1]
    
    
    #imp_ = importances(clf,x_val,y_val)
    #print(imp_)
    #imp_.rename(columns={'Importance':i },inplace=True)
    #feature_importance_df = pd.merge(feature_importance_df,imp_,on='Feature')
    models.append(clf)
    #models_xgb.append(model_xgb)
    
    print(i, " err_lgm: ", roc_auc_score(y_val, pred_lgb))
    #print(i, "err_xgb: ", roc_auc_score(y_val, pred_xgb))
    
    err_lgb.append(roc_auc_score(y_val, pred_lgb))
    #err_xgb.append(roc_auc_score(y_val, pred_xgb))
    
    
    #best_iteration = model_xgb.get_booster().best_iteration
    
    
    final_pred_lgb = clf.predict_proba(test)[:,1]
    #final_pred_xgb = model_xgb.predict_proba(test)[:,1]
    
    oof_lgb.append(final_pred_lgb)
    #oof_xgb.append(final_pred_xgb)

    #np.append(oof, np.array([final_pred]).transpose(), axis=1)
    i = i + 1

sum(err_lgb)/nfolds
err_xgb
best1 = oof_lgb[2]
best2 = oof_lgb[0]
best3 = oof_lgb[1]
best4 = oof_lgb[7]


#weighted_off = best1*0.1+ best2*0.2 + best3*0.3 + best4*0.4

#weighted_off = best1*0.1+ best2*0.2 + best3*0.3 + best4*0.4
#finalpreds = oof[2]
weighted_off = ( best1+best2+best3+best4)/4
#finalpreds = np.mean(oof_xgb,0)
finalpreds = np.mean(oof_lgb,0)
#finalpreds = best1
#weighted_off = oof_xgb[9]
submission = pd.DataFrame({'id':df_test['id'], 'Response':finalpreds})
submission.to_csv("lightgbm250.csv",index=False)
plt.rcParams['figure.figsize'] = (150, 50)
lgb.plot_importance(models[2])
plt.show()
feature_important = models_xgb[0].get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data_feature = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data_feature.to_csv("smote XGboost Feature Importance 1:32.csv.csv")
#data_feature.plot(kind='barh')
df_train.shape
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from imblearn.over_sampling import RandomOverSampler

os =  RandomOverSampler(0.70)

#X_train_res, y_train_res = os.fit_sample(df_train[feature_cols],df_train[label_col])

ordered_rank_features=SelectKBest(score_func=chi2,k=50)
ordered_feature=ordered_rank_features.fit(df_train[feature_cols],df_train[label_col])

dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(df_train.columns)

features_rank=pd.concat([dfcolumns,dfscores],axis=1)

features_rank.columns=['Features','Score']

features_rank.to_csv("features based on chisquer.csv",index=False)

features_rank.nlargest(99,'Score')
