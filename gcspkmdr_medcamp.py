import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,15
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split, GridSearchCV ,RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


import warnings
warnings.filterwarnings("ignore")
train_data = pd.read_csv('/kaggle/input/Train/Train/Train.csv')
patient_profile = pd.read_csv('/kaggle/input/Train/Train/Patient_Profile.csv')
health_camp_detail = pd.read_csv('/kaggle/input/Train/Train/Health_Camp_Detail.csv')
first_health_camp = pd.read_csv('/kaggle/input/Train/Train/First_Health_Camp_Attended.csv')
second_health_camp = pd.read_csv('/kaggle/input/Train/Train/Second_Health_Camp_Attended.csv')
third_health_camp = pd.read_csv('/kaggle/input/Train/Train/Third_Health_Camp_Attended.csv')
test_data = pd.read_csv('/kaggle/input/test.csv')
print(train_data.shape)
train_data.head()
print(test_data.shape)
test_data.head()
health_camp_detail.tail()
first_health_camp.head()
first_health_camp.drop(columns=['Unnamed: 4'],inplace=True)
sns.distplot(first_health_camp['Health_Score'])
second_health_camp.head()
sns.distplot(second_health_camp['Health Score'])
third_health_camp.head()
visits = third_health_camp.Number_of_stall_visited.value_counts()
plt.xlabel("No.of visits")
plt.ylabel('Count')
sns.barplot(visits.index , visits.values)
def nullColumns(train_data):
    list_of_nullcolumns =[]
    for column in train_data.columns:
        total= train_data[column].isna().sum()
        try:
            if total !=0:
                print('Total Na values is {0} for column {1}' .format(total, column))
                list_of_nullcolumns.append(column)
        except:
            print(column,"-----",total)
    print('\n')
    return list_of_nullcolumns


def percentMissingFeature(data):
    data_na = (data.isnull().sum() / len(data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    print(missing_data.head(20))
    return data_na


def plotMissingFeature(data_na):
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    if(data_na.empty ==False):
        sns.barplot(x=data_na.index, y=data_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
print('train data')
print(nullColumns(train_data))
print('\n')
print('test_data')
print(nullColumns(test_data))
print('\n')
print('first_health_camp')
print(nullColumns(first_health_camp))
print('\n')
print('second_health_camp')
print(nullColumns(second_health_camp))
print('\n')
print('third_health_camp')
print(nullColumns(third_health_camp))
print('\n')
print('patient_profile')
print(nullColumns(patient_profile))
print('\n')
print('health_camp_detail')
print(nullColumns(health_camp_detail))
print('\n')
combined_data = pd.concat([train_data,test_data],axis = 0)
combined_data = combined_data.reset_index(drop =  True)
combined_data.head()
Number_Of_Prior_Registration = []
combined_data['Number_Of_Prior_Registration'] = 0
for idx, row in combined_data.iterrows():
    patient_id = row['Patient_ID']
    Number_Of_Prior_Registration.append(combined_data.iloc[:idx,:].loc[(combined_data['Patient_ID']==patient_id),:].shape[0])
combined_data['Number_Of_Prior_Registration'] = Number_Of_Prior_Registration
#first_occurence = []
#for val in combined_data['Patient_ID'].unique():
#    first_occurence.append(combined_data[combined_data.Patient_ID == val].first_valid_index())
#combined_data['Old_Patient_ID'] = 1
#combined_data.loc[first_occurence,'Old_Patient_ID'] = 0
combined_data = pd.merge(combined_data,health_camp_detail,on='Health_Camp_ID',how= 'left')
first_health_camp['F_O'] = 1
combined_data = pd.merge(combined_data,first_health_camp[['Patient_ID','Health_Camp_ID','F_O']],on=['Patient_ID','Health_Camp_ID'],how = 'left')
second_health_camp['S_O'] = 1
combined_data = pd.merge(combined_data,second_health_camp[['Patient_ID','Health_Camp_ID','S_O']],on=['Patient_ID','Health_Camp_ID'],how = 'left')
third_health_camp['T_O'] = third_health_camp['Number_of_stall_visited'].apply(lambda x : 1 if x>0 else 0)
combined_data = pd.merge(combined_data,third_health_camp[['Patient_ID','Health_Camp_ID','T_O']],on=['Patient_ID','Health_Camp_ID'],how = 'left')
combined_data = pd.merge(combined_data,patient_profile,on=['Patient_ID'],how = 'left')
combined_data['F_O'] = combined_data['F_O'].fillna(0)
combined_data['S_O'] = combined_data['S_O'].fillna(0)
combined_data['T_O'] = combined_data['T_O'].fillna(0)
combined_data['Outcome'] = combined_data['F_O'] +combined_data['S_O'] +combined_data['T_O']
combined_data['Outcome'] =combined_data['Outcome'].astype('int64')
#combined_data['Health_Camp_Type'] = np.where(combined_data['Health_Camp_ID'].isin(first_health_camp.Health_Camp_ID),1
#                                             ,np.where(combined_data['Health_Camp_ID'].isin(second_health_camp.Health_Camp_ID),2
 #                                                     ,np.where(combined_data['Health_Camp_ID'].isin(third_health_camp.Health_Camp_ID),3,0)))
outcome = combined_data.loc[:75278,"Outcome"].value_counts()
print(outcome)
plt.xlabel("outcome")
plt.ylabel('count')
sns.barplot(outcome.index , outcome.values)
nullColumns(combined_data)
combined_data['Registration_Date'] = combined_data.apply(
    lambda row: row['Camp_Start_Date'] if row['Registration_Date'] != row['Registration_Date'] else row['Registration_Date'],
    axis=1
)

combined_data['City_Type'] = combined_data['City_Type'].fillna(combined_data['City_Type'].mode()[0])

combined_data['Employer_Category'] = combined_data['Employer_Category'].fillna(combined_data['Employer_Category'].mode()[0])

combined_data['Income'] = combined_data['Income'].apply(lambda x : '0' if x == 'None' else x)

combined_data['Education_Score'] = combined_data['Education_Score'].apply(lambda x : '0' if x == 'None' else x)

ages = []

combined_data['Age'].apply(lambda x : ages.append(float(x)) if x!= 'None' else 0).mean()

age_mean = sum(ages)/len(ages)

combined_data['Age'] = combined_data['Age'].apply(lambda x : age_mean if x == 'None' else float(x))

combined_data['Age'] = combined_data['Age'].astype('int64')
def generate_date_features(calendar,colname,prefix):
    
    df = pd.DataFrame()
    
    df[prefix+'_Year'] = pd.to_datetime(calendar[colname]).dt.year

    df[prefix+'_Month'] = pd.to_datetime(calendar[colname]).dt.month

    df[prefix+'_Day'] = pd.to_datetime(calendar[colname]).dt.day

    df[prefix+'_Dayofweek'] = pd.to_datetime(calendar[colname]).dt.dayofweek

    df[prefix+'_DayOfyear'] = pd.to_datetime(calendar[colname]).dt.dayofyear

    df[prefix+'_Week'] = pd.to_datetime(calendar[colname]).dt.week

    #df['Quarter'] = pd.to_datetime(calendar[colname]).dt.quarter 

    #df['Is_month_start'] = pd.to_datetime(calendar[colname]).dt.is_month_start

    #df['Is_month_end'] = pd.to_datetime(calendar[colname]).dt.is_month_end

    #df['Is_quarter_start'] = pd.to_datetime(calendar[colname]).dt.is_quarter_start

    #df['Is_quarter_end'] = pd.to_datetime(calendar[colname]).dt.is_quarter_end

    #df['Is_year_start'] = pd.to_datetime(calendar[colname]).dt.is_year_start

    #df['Is_year_end'] = pd.to_datetime(calendar[colname]).dt.is_year_end

    #df['Semester'] = np.where(df['Quarter'].isin([1,2]),1,2)

    #df['Is_weekend'] = np.where(df['Dayofweek'].isin([5,6]),1,0)

    #df['Is_weekday'] = np.where(df['Dayofweek'].isin([0,1,2,3,4]),1,0)

    return df
date_features1 = generate_date_features(combined_data,'Camp_End_Date','Camp_End')
date_features2 = generate_date_features(combined_data,'Camp_Start_Date','Camp_Start')
#date_features3 = generate_date_features(combined_data,'Registration_Date','Registration')
combined_data = pd.concat([date_features1,
                           date_features2,
                           #date_features3,
                           combined_data],axis =1)
combined_data['Camp_duration'] = (pd.to_datetime(combined_data['Camp_End_Date'])-pd.to_datetime(combined_data['Camp_Start_Date'])).dt.days
#combined_data['Approach'] = (pd.to_datetime(combined_data['First_Interaction'])-pd.to_datetime(combined_data['Camp_Start_Date'])).dt.days
combined_data['Delay'] = (pd.to_datetime(combined_data['Camp_End_Date'])-pd.to_datetime(combined_data['First_Interaction'])).dt.days
combined_data['Eagernes'] = (pd.to_datetime(combined_data['Camp_Start_Date'])-pd.to_datetime(combined_data['Registration_Date'])).dt.days
le = LabelEncoder()
for col in combined_data.columns:
    #print(col)
    if combined_data[col].dtype == 'O':
        combined_data[col] = le.fit_transform(combined_data[col])
        
combined_data['Enthusiasm'] = combined_data.apply(lambda x : 1 if x.LinkedIn_Shared == 1 or x.Facebook_Shared == 1 or x.Twitter_Shared == 1 else 0,axis =1)
#combined_data['Social_Media'] = combined_data['LinkedIn_Shared'] +combined_data['Facebook_Shared'] + combined_data['Twitter_Shared']
#combined_data = combined_data.drop(columns= ['LinkedIn_Shared','Facebook_Shared','Twitter_Shared'])
#combined_data['Income_Age1'] = combined_data['Income']/combined_data['Age']
#combined_data['Income_Age2'] = combined_data['Income']*combined_data['Age']
#combined_data['Education_Age1'] = combined_data['Education_Score']/ combined_data['Age']
#combined_data['Education_Age2'] = combined_data['Education_Score'] * combined_data['Age']
#combined_data['Education_Income1'] = combined_data['Education_Score']/ combined_data['Income']
#combined_data['Education_Income1'] = combined_data['Education_Income1'].fillna(0)
#combined_data['Education_Income2'] = combined_data['Education_Score'] * combined_data['Income']
#combined_data['Category1_squared'] =combined_data['Category1']*combined_data['Category1']
combined_data = combined_data.drop(columns= ['LinkedIn_Shared'
                                             ,'Facebook_Shared'
                                             ,'Twitter_Shared'
                                            #,'Online_Follower'
                                            ])
combined_data = combined_data.drop(columns = ['Patient_ID'
                                              ,'Health_Camp_ID'
                                              ,'Var3'
                                              ,'Var4'
                                              ,'F_O'
                                              ,'S_O'
                                              ,'T_O'
                                              ,'Employer_Category'
                                             #,'Category3'
                                             ])

combined_data = combined_data.drop(columns= ['First_Interaction'
                                             ,'Camp_End_Date'
                                             ,'Camp_Start_Date'
                                             ,'Registration_Date'
                                             #,'Income'
                                             #,'Education_Score'
                                             #,'Age'
                                            ])
target = combined_data['Outcome']
combined_data = combined_data.drop(columns = ['Outcome'])
def create_submission_file(model_list):
    preds = 0
    submission = pd.read_csv('/kaggle/input/sample_submission.csv')
    for model in model_list:
        preds = preds + (model.predict_proba(combined_data.iloc[75278:,:])[:,-1])
        #preds = preds + (model.predict_proba(combined_data[75278:,:])[:,-1])
    submission.loc[:,'Outcome'] = preds/len(model_list)
    !rm './submission.csv'
    submission.to_csv('submission.csv', index = False, header = True)
    print(submission.head())
def auc_cv(model,X,y):
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
    auc = cross_val_score(model, X, y, scoring='roc_auc', cv = rskf)
    return(auc)
X_train = combined_data.iloc[:55000,:]
X_val = combined_data.iloc[55000:75278,:]
y_train = target[:55000]
y_val =target[55000:75278]
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(n_jobs = -1)
X_train, y_train = tl.fit_sample(X_train, y_train)
def feature_importance(model, X_train):

    print(model.feature_importances_)
    names = X_train.columns.values
    ticks = [i for i in range(len(names))]
    plt.bar(ticks, model.feature_importances_)
    plt.xticks(ticks, names,rotation =90)
    plt.show()
weight = float(y_train.value_counts()[0]/y_train.value_counts()[1])
print("Count Majority Class : {0}".format(y_train.value_counts()[0]))
print("Count Minority Class : {0}".format(y_train.value_counts()[1]))
print('weight : {0}'.format(weight))
model_xgb = xgb.XGBClassifier(scale_pos_weight= weight, 
                              colsample_bytree=0.8, gamma=0.045, 
                             learning_rate=0.1, max_depth=10, 
                             n_estimators=1000,
                             reg_alpha=0.45, reg_lambda=0.8,
                             subsample=0.5,
                             random_state =7, nthread = -1,seed=42,n_jobs = -1)
###tooooooo slow

#score = auc_cv(model_xgb,X_train,y_train)
#print('mean : {0} std: {1}'.format(score.mean(),score.std())) 

model_xgb.fit(X_train,y_train,eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=['auc','logloss'],
        early_stopping_rounds = 50,
        verbose=2)
feature_importance(model_xgb,X_train)
create_submission_file([model_xgb])
model_lgb = lgb.LGBMClassifier(scale_pos_weight= weight, bagging_frequency=4, boosting_type='gbdt'
                               ,colsample_bytree=0.8, feature_fraction=0.5,
                               importance_type='split', learning_rate=0.1, max_depth=10,
                               min_split_gain=0.0001, n_estimators=1000, n_jobs=-1,random_state=101, reg_alpha=0.45,
                               reg_lambda=0.8, subsample=1.0)

##Cross-Validation

#score = auc_cv(model_lgb,X_train,y_train)
#print('mean : {0} std: {1}'.format(score.mean(),score.std()))

model_lgb.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric=['auc','logloss'],
        early_stopping_rounds = 100,
        verbose=2)
print(model_lgb.best_score_['valid_1'])
feature_importance(model_lgb,X_train)
create_submission_file([model_lgb])
rfc = RandomForestClassifier(n_estimators=500 ,
                             max_depth=10, min_samples_split=2, 
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                             n_jobs=-1, random_state=123, verbose=3, 
                             class_weight='balanced')

##Cross-Validation
#score = auc_cv(rfc,combined_data.iloc[:75278,:],target[:75278])
#print('mean : {0} std: {1}'.format(score.mean(),score.std()))

rfc.fit(X_train,y_train)
feature_importance(rfc,X_train)
create_submission_file([rfc,model_lgb])
etc = ExtraTreesClassifier(n_estimators=500 ,
                             max_depth=6, min_samples_split=2, 
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                             max_leaf_nodes=None, min_impurity_decrease=0.0,
                             bootstrap=True, oob_score=False, n_jobs=-1, random_state=123, verbose=3, 
                             class_weight='balanced')
##Cross-Validation

#score = auc_cv(etc,combined_data.iloc[:75278,:],target[:75278])
#print('mean : {0} std: {1}'.format(score.mean(),score.std()))
etc.fit(X_train,y_train)
create_submission_file([etc,rfc,model_lgb,model_xgb])