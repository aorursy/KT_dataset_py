# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict
from sklearn.preprocessing import Imputer
pd.set_option('display.max_columns',None)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score,GridSearchCV,RandomizedSearchCV
# Any results you write to the current directory are saved as output.
color_dict = OrderedDict({1: 'red', 0: 'green'})
loan_dict = OrderedDict({1: 'Cant repay',0:'Repaid'})
drop_cols = []
def plot_continuos(data,var):
    plt.figure(figsize=(8,8))
    for key,clr in color_dict.items():
        sns.kdeplot(data[train.TARGET==key][var].dropna(),color=clr,label = loan_dict[key])
    plt.xlabel(var)
    plt.ylabel('Density')
    
def plot_bar(data,x,y):
    tempdf = data.groupby(x)[y].mean().reset_index()
    sns.barplot(x = x,y = y,data = tempdf)
    plt.xticks(rotation=90)
    plt.show()
    
def plot_count(data,x,hue):
    sns.countplot(x = x,data = data,hue = hue)
    plt.show()
    
def create_missing_columns_report(data):    
    prop = data.isnull().sum()/len(data)
    missing_df = pd.DataFrame(prop).reset_index().rename(columns = {'index':'columns',0:'%missing'})
    missing_df.sort_index(by = '%missing',ascending=False,inplace=True)
    
    dict_column_datatype = data.dtypes.to_dict()
    missing_df['datatype'] = missing_df['columns'].map(dict_column_datatype)
    return missing_df


def convert_categoricals(data):
    ohe_frame = pd.DataFrame()
    ohe_cols= []
    for col in categorical_cols:
        if(data[col].nunique()==2):            
            data.loc[:,col] = np.where(((data[col]=='no')|(data[col]=='N')|(data[col]=='F')),0,1)
            data[col] = data[col].astype(np.int8)         
    data = pd.get_dummies(data)       
    return data

def drop_rows(data,column,value,**args):
    
    index = data.index[(data[data[column]==value])&(data[AMT_INCOME_TOTAL]>1000000)]
    train_filter = train.drop(index,axis=0)

def check_column_consistency(df1, df2):
        """ Check if columns of train and test data are in same order or not. Should be called after train, valid
        and test has been transformed. If certain columns are missing or are not in order then they are added or ordered
        accordingly
        :param df1: train data frame
        :param df2: test or valid data frame
        :return: consistent data frames
        """
        df1_columns = df1.columns.tolist()
        df2_columns = df2.columns.tolist()

        for df1_col in df1_columns:
            if df1_col not in df2_columns:
                df2[df1_col] = 0
        df2 = df2[df1_columns]
        df1 = df1[df1_columns]
        return df1, df2
    
    
def days_age(data):

    #cols_days = [x for x in data if x.startswith('DAYS_')]
    #for col in cols_days:
    data.loc[:,'Age(years)'] = data['DAYS_BIRTH']*-1/365
    return data


def days_employ_flag(data):
    quart90 = np.percentile(data.DAYS_EMPLOYED, 90)
    index = data[data.DAYS_EMPLOYED>=quart90].index
    data.loc[:,'days_employ_flag'] = np.where(data.DAYS_EMPLOYED>=quart90,1,0)
    days_mean = np.mean(data.loc[~(data.DAYS_EMPLOYED>=quart90),'DAYS_EMPLOYED'].values)
    data.loc[index,'DAYS_EMPLOYED'] = days_mean
    
    return data



def train_eval(feature_train,feature_test,target_train,nfolds,test_ids,return_preds=False):
      
    sfold = StratifiedKFold(n_splits= nfolds,shuffle=True,random_state=100)
    valid_scores_list = []
    test_predictions_df = pd.DataFrame()
    feature_columns = feature_train.columns
    feature_importance = np.zeros(len(feature_columns))
    featuresNames = []
    featureImps =[]

    feature_train_arr = feature_train.values
    feature_test_arr = feature_test.values
    target_train_arr = target_train.values
    
    clf_lgb=lgb.LGBMClassifier(  n_estimators=10000,
                                 n_jobs = -1,
                                 metric = 'None',
                                 random_state=100,
                                 class_weight='balanced')
    for i, (train_index,valid_index) in enumerate(sfold.split(feature_train,target_train)):
        fold_predictions_df = pd.DataFrame()        
        # Training and validation data
        X_train = feature_train_arr[train_index]
        X_valid = feature_train_arr[valid_index]
        y_train = target_train_arr[train_index]
        y_valid = target_train_arr[valid_index]
        
        
        fit_params={"early_stopping_rounds":100,
            "eval_metric" : 'auc', 
            "eval_set" : [(X_train,y_train), (X_valid,y_valid)],
            'eval_names': ['train', 'valid'],
            'verbose': 100,
            'categorical_feature': 'auto'}
        
        clf_lgb.fit(X_train,y_train,**fit_params)
        best_iteration = clf_lgb.best_iteration_
        valid_scores_list.append(clf_lgb.best_score_['valid']['auc'])
        display(f'Fold {i + 1}, Validation Score: {round(valid_scores_list[i], 5)}, Estimators Trained: {clf_lgb.best_iteration_}')
        fold_probabilitites = clf_lgb.predict_proba(feature_test_arr,num_iteration = best_iteration)[:,1]      
        fold_predictions_df['Score'] = fold_probabilitites     
        fold_predictions_df['SK_ID_CURR'] = test_ids
        fold_predictions_df['fold'] = (i+1)
        
        test_predictions_df = test_predictions_df.append(fold_predictions_df)
        valid_scores = np.array(valid_scores_list)
        #print(test_predictions_df.shape)
        fold_feature_importance = clf_lgb.feature_importances_
        fold_feature_importance = 100.0 * (fold_feature_importance / fold_feature_importance.max())
        feature_importance = (feature_importance+fold_feature_importance)/nfolds
        sorted_idx = np.argsort(feature_importance)
        for item in sorted_idx[::-1][:]:
            featuresNames.append(np.asarray(feature_columns)[item])
            featureImps.append(feature_importance[item])
            featureImportance = pd.DataFrame([featuresNames, featureImps]).transpose()
            featureImportance.columns = ['FeatureName', 'Importance']
        
    
    # Average the predictions over folds    
    test_predictions_df = test_predictions_df.groupby('SK_ID_CURR', as_index = False).mean()
    #test_predictions_df['Target'] = test_predictions_df[[0,1]].idxmax(axis = 1)
    #test_predictions_df['Score'] = test_predictions_df[1]   
    test_predictions_df.drop('fold',axis=1,inplace=True)   
        
    
    return test_predictions_df,featureImportance,valid_scores
train = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")
prev_appl = pd.read_csv("../input/previous_application.csv")
bureau = pd.read_csv("../input/bureau.csv")
bureau_bal = pd.read_csv("../input/bureau_balance.csv")
print('Shape of train:{}'.format(train.shape))
train.head(3)

print('Shape of test:{}'.format(test.shape))
test.head(3)
prev_appl.loc[:,'FLAG_LAST_APPL_PER_CONTRACT'] = prev_appl['FLAG_LAST_APPL_PER_CONTRACT'].map({'Y':1,'N':0})
agg_data_prev_appl1 = prev_appl.groupby('SK_ID_CURR',as_index=False)[['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_DOWN_PAYMENT','AMT_GOODS_PRICE','DAYS_DECISION','CNT_PAYMENT','DAYS_FIRST_DUE']].agg(['mean','sum','std'])
agg_data_prev_appl2 = prev_appl.groupby('SK_ID_CURR',as_index=False)[['FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']].agg(['sum'])
agg_data_prev_appl3 = prev_appl.groupby('SK_ID_CURR',as_index=False).size().reset_index().rename(columns = {0:'Count'}).set_index('SK_ID_CURR')                                                               
agg_data_prev_appl1.columns = [' _'.join(col).strip() for col in agg_data_prev_appl1.columns.values]
agg_data_prev_appl2.columns = [' _'.join(col).strip() for col in agg_data_prev_appl2.columns.values]
#print(agg_data_prev_appl.columns)
agg_data_prev_appl = pd.concat([agg_data_prev_appl1,agg_data_prev_appl2,agg_data_prev_appl3],axis=1)                                                                     
agg_data_prev_appl.reset_index(inplace=True)
agg_data_prev_appl.columns = ['Previous_Appl_'+col for col in agg_data_prev_appl.columns.values]
#agg_data_prev_appl.rename(columns = {'Previous_Appl_SK_ID_CURR':'SK_ID_CURR'},inplace=True)
agg_data_prev_appl.head(3)
train = pd.merge(train,agg_data_prev_appl,left_on='SK_ID_CURR',right_on='Previous_Appl_SK_ID_CURR',how='left')
test = pd.merge(test,agg_data_prev_appl,left_on='SK_ID_CURR',right_on='Previous_Appl_SK_ID_CURR',how='left')
print(train.shape,test.shape)
bureau.head(4)
agg_data_bureau1 = bureau.groupby('SK_ID_CURR',as_index=False)[['DAYS_CREDIT','DAYS_CREDIT_ENDDATE','CREDIT_DAY_OVERDUE','AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT_UPDATE','AMT_ANNUITY']].agg(['mean','sum','std'])
agg_data_bureau2 = bureau.groupby('SK_ID_CURR',as_index=False)[['CNT_CREDIT_PROLONG']].agg(['sum'])
agg_data_bureau3 = bureau.groupby('SK_ID_CURR',as_index=False).size().reset_index().rename(columns = {0:'Bureau_Count'}).set_index('SK_ID_CURR')                                                               
agg_data_bureau1.columns = [' _'.join(col).strip() for col in agg_data_bureau1.columns.values]
agg_data_bureau2.columns = [' _'.join(col).strip() for col in agg_data_bureau2.columns.values]
#print(agg_data_prev_appl.columns)
agg_data_bureau = pd.concat([agg_data_bureau1,agg_data_bureau2,agg_data_bureau3],axis=1)                                                                     
agg_data_bureau.reset_index(inplace=True)
agg_data_bureau.columns = ['Bureau_'+col for col in agg_data_bureau.columns.values]
train = pd.merge(train,agg_data_bureau,left_on='SK_ID_CURR',right_on='Bureau_SK_ID_CURR',how='left')
test = pd.merge(test,agg_data_bureau,left_on='SK_ID_CURR',right_on='Bureau_SK_ID_CURR',how='left')
print(train.shape,test.shape)
train.info()
train.select_dtypes('object').nunique()
#Show a head of object columns
train[list(train.select_dtypes('object').columns)].head(2)
#Show a head of float columns
train[list(train.select_dtypes(np.float64).columns)].head(2)
sns.countplot(train.TARGET)
plt.xlabel('Target')
plt.ylabel('Frequency')
train.TARGET.value_counts(normalize=True)
missing_df = create_missing_columns_report(train)
missing_df.head(20)
missing_df[missing_df.datatype == 'object']
train.describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])
test.describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])
train.query('AMT_INCOME_TOTAL>100000000')
train.OCCUPATION_TYPE.unique()
plot_bar(train,'OCCUPATION_TYPE','AMT_INCOME_TOTAL')
train_lab = train.query('OCCUPATION_TYPE=="Laborers"')
train_lab.describe(percentiles= [0.9,0.94,0.98,0.99,0.995,0.999])
train_lab = train.query('OCCUPATION_TYPE=="Laborers"')
train_lab.drop((train_lab.index[train_lab['AMT_INCOME_TOTAL']>1000000]),axis=0,inplace=True)
plot_continuos(train_lab,'AMT_INCOME_TOTAL')
index = train.index[(train.OCCUPATION_TYPE=="Laborers")&(train['AMT_INCOME_TOTAL']>1000000)]
train_filter = train.drop(index,axis=0)
train_filter.shape
train_filter.describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])
plot_continuos(train_filter,'AMT_INCOME_TOTAL')
plot_continuos(train_filter,'DAYS_EMPLOYED')
print('Number of samples:{}'.format(len(train_filter.query('DAYS_EMPLOYED==365243'))))
print('Percentage of samples:{}'.format(len(train_filter.query('DAYS_EMPLOYED==365243'))*100/len(train_filter)))
train_filter2 = days_employ_flag(train_filter.copy())
print('Shape of data before handling DAYS_EMPLOYED:{}'.format(train_filter.shape))
print('Shape of data after handling DAYS_EMPLOYED:{}'.format(train_filter2.shape))
#quart3 = np.percentile(train_filter.DAYS_EMPLOYED, 75)
#iqr = quart3 - quart1

#outlier = train_filter[train_filter['DAYS_EMPLOYED'] > quart3 + 1.5 * iqr].DAYS_EMPLOYED.max()
plot_continuos(train_filter2,'DAYS_EMPLOYED')

train_filter3 = days_age(train_filter2.copy())
print('Shape of data before handling DAYS_BIRTH:{}'.format(train_filter2.shape))
print('Shape of data after handling DAYS_BIRTH:{}'.format(train_filter3.shape))
plot_continuos(train_filter3,'Age(years)')
drop_cols.append('DAYS_BIRTH')
train_filter3[[col for col in train_filter3 if col.startswith('OBS')]].describe(percentiles=[0.9,0.92,0.94,0.96,0.98,0.99,0.995])
train_filter3.OBS_30_CNT_SOCIAL_CIRCLE.isnull().sum()
outlier_30 = np.nanpercentile(train_filter3.OBS_30_CNT_SOCIAL_CIRCLE,100)
outlier_60 = np.nanpercentile(train_filter3.OBS_60_CNT_SOCIAL_CIRCLE,100)
SK_ID_CURR_30 = train_filter3[train_filter3['OBS_30_CNT_SOCIAL_CIRCLE']>=outlier_30]['SK_ID_CURR'].values[0]
SK_ID_CURR_60=train_filter3[train_filter3['OBS_60_CNT_SOCIAL_CIRCLE']>=outlier_60]['SK_ID_CURR'].values[0]

print(SK_ID_CURR_30 ,',',SK_ID_CURR_60)
train_filter3.drop(train_filter3.index[train_filter3.SK_ID_CURR==SK_ID_CURR_30],axis=0,inplace=True)
print('Shape of data after handling outlier row:{}'.format(train_filter3.shape))
plot_continuos(train_filter3,'OBS_30_CNT_SOCIAL_CIRCLE')
plot_continuos(train_filter3,'OBS_60_CNT_SOCIAL_CIRCLE')
train_filter3.head(3)
plot_count(train_filter3,'FLAG_MOBIL',hue='TARGET')
train_filter3[train_filter3.FLAG_MOBIL==0]
plot_count(train_filter3,'FLAG_EMP_PHONE',hue='TARGET')
plot_count(train_filter3,'FLAG_WORK_PHONE',hue='TARGET')
plot_count(train_filter3,'FLAG_CONT_MOBILE',hue='TARGET')
plot_count(train_filter3,'FLAG_PHONE',hue='TARGET')
train_filter3['flag_mob'] = train_filter3['FLAG_MOBIL']+train_filter3['FLAG_EMP_PHONE']+\
                            train_filter3['FLAG_WORK_PHONE']+train_filter3['FLAG_CONT_MOBILE']+train_filter3['FLAG_PHONE']
train_filter3.head(3)
plot_count(train_filter3,'flag_mob',hue='TARGET')
train_filter3['flag_mob'] = train_filter3['flag_mob'].astype('object')
drop_cols.extend(['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE'])
drop_cols
print('Shape of data after handling flag_mobile:{}'.format(train_filter3.shape))
train_filter3.head(3)
test_filter2 = days_employ_flag(test.copy())
print('Shape of data before handling DAYS_EMPLOYED:{}'.format(test.shape))
print('Shape of data after handling DAYS_EMPLOYED:{}'.format(test_filter2.shape))


test_filter3 = days_age(test_filter2.copy())
print('Shape of data before handling DAYS_BIRTH:{}'.format(test_filter2.shape))
print('Shape of data after handling DAYS_BIRTH:{}'.format(test_filter3.shape))



test_filter3['flag_mob'] = test_filter3['FLAG_MOBIL']+test_filter3['FLAG_EMP_PHONE']+\
                            test_filter3['FLAG_WORK_PHONE']+test_filter3['FLAG_CONT_MOBILE']+test_filter3['FLAG_PHONE']
    
#print('Shape of data before handling flag_mobile:{}'.format(test_filter2.shape))
print('Shape of data after handling flag_mobile:{}'.format(test_filter3.shape))
test_filter3['flag_mob'] = test_filter3['flag_mob'].astype('object')

categorical_cols = list(train_filter3.select_dtypes('object').columns)
unique_levels = train_filter3[categorical_cols].apply(lambda x: x.nunique())
print('Total levels in categorical columns:{}'.format(unique_levels.sum()))
unique_levels
train_copy = convert_categoricals(train_filter3.copy())
test_copy = convert_categoricals(test_filter3.copy())

print('Train shape:{},Test shape:{}'.format(train_copy.shape,test_copy.shape))
train_copy.head(2)
test_copy.head(2)
train_copy.info()
test_copy.info()
train_labels = train_copy['TARGET']
test_ids = test.SK_ID_CURR.values
drop_cols.append('SK_ID_CURR')
print(drop_cols)
train_copy.drop(drop_cols+['TARGET'],axis=1,inplace=True)
test_copy.drop(drop_cols,axis=1,inplace=True)
train_copy, test_copy= check_column_consistency(train_copy, test_copy)
#train_copy,test_copy = train_copy.align(test_copy,axis=1,join='inner')
print('Train shape:{},Test shape:{}'.format(train_copy.shape,test_copy.shape))
train_copy.info()
train_copy.head(3)
test_copy.head(3)
test_copy.info()

test_predictions_df,featureImportance,valid_scores = train_eval(train_copy,test_copy,train_labels,10,test_ids,return_preds=False)
test_predictions_df.head(2)
#submission = test_predictions_df[['SK_ID_CURR','Score']]
test_predictions_df.rename(columns = {'Score':'TARGET'},inplace=True)
test_predictions_df.head(2)
test_predictions_df.to_csv('baseline_lgb.csv', index = False)









