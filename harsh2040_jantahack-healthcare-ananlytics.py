import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *

import gc
from tqdm import tqdm, tqdm_notebook
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
pwd
path = '../input/healthcare-ananlytics/'
train = pd.read_csv(path+'Train.csv')
test = pd.read_csv(path+'test_l0Auv8Q.csv')
health_camp = pd.read_csv(path+'Health_Camp_Detail.csv')
camp_1 = pd.read_csv(path+'First_Health_Camp_Attended.csv')
camp_2 = pd.read_csv(path+'Second_Health_Camp_Attended.csv')
camp_3 = pd.read_csv(path+'Third_Health_Camp_Attended.csv')
patient_profile = pd.read_csv(path+'Patient_Profile.csv')
ss = pd.read_csv(path+'sample_submmission.csv')
data_dict = pd.read_excel(path+'Data_Dictionary.xlsx')
train.head()
train.columns
patient_profile.columns
test['Patient_ID'].describe()
np.intersect1d(train['Patient_ID'], test['Patient_ID']).shape[0]/train['Patient_ID'].nunique()
patient_profile[['Income', 'Education_Score', 'Age']] = patient_profile[['Income', 'Education_Score', 'Age']].apply(lambda x: x.str.replace('None', 'NaN').astype('float'))
patient_profile[['City_Type',	'Employer_Category']] = patient_profile[['City_Type',	'Employer_Category']].apply(lambda x: pd.factorize(x)[0])
for df_tmp in [train, test]:
  for c in ['Health_Camp_ID']:
    # mapper = train
    df_tmp[c + '_freq'] = df_tmp[c].map(df_tmp[c].value_counts(normalize=True))

train = pd.merge(train, patient_profile, on = 'Patient_ID', how = 'left')
test = pd.merge(test, patient_profile, on = 'Patient_ID', how = 'left')
#### Getting the target

for c in [camp_1, camp_2, camp_3, train]:
  c['id'] = c['Patient_ID'].astype('str') + c['Health_Camp_ID'].astype('str')
camp_3 = camp_3[camp_3['Number_of_stall_visited'] > 0]

all_patients_in_camp = pd.Series(camp_1['id'].tolist() + camp_2['id'].tolist() + camp_3['id'].tolist()).unique()

train['target'] = 0
train.loc[train['id'].isin(all_patients_in_camp), 'target'] = 1
train['target'].value_counts(normalize=True)
health_camp['Category1'] = health_camp['Category1'].map({'First': 1, 'Second': 2, 'Third': 3})
health_camp['Category2'] = pd.factorize(health_camp['Category2'])[0]

health_camp['Camp_Start_Date'] = pd.to_datetime(health_camp['Camp_Start_Date'])
health_camp['Camp_End_Date'] = pd.to_datetime(health_camp['Camp_End_Date'])
health_camp['total_days_of_campaign'] = (health_camp['Camp_End_Date'] - health_camp['Camp_Start_Date']).dt.days

health_camp['starting_day'] = pd.to_datetime(health_camp['Camp_Start_Date']).dt.day
health_camp['ending_day'] = pd.to_datetime(health_camp['Camp_End_Date']).dt.day
health_camp['starting_week'] = pd.to_datetime(health_camp['Camp_Start_Date']).dt.week
health_camp['ending_week'] = pd.to_datetime(health_camp['Camp_End_Date']).dt.week
health_camp['starting_month'] = pd.to_datetime(health_camp['Camp_Start_Date']).dt.month
health_camp['ending_month'] = pd.to_datetime(health_camp['Camp_End_Date']).dt.month
health_camp['Is_weekend_start'] = np.where(health_camp['Camp_Start_Date'].isin([5,6]),1,0)
health_camp['Is_weekday_start'] = np.where(health_camp['Camp_Start_Date'].isin([0,1,2,3,4]),1,0)
health_camp['Dayofweek_start'] = pd.to_datetime(health_camp['Camp_Start_Date']).dt.dayofweek
health_camp['Dayofweek_end'] = pd.to_datetime(health_camp['Camp_End_Date']).dt.dayofweek

# health_camp['difference_to_next_campaign'] = (health_camp['Camp_End_Date'] - health_camp['Camp_Start_Date'].shift(-1)).dt.days
# health_camp['difference_to_prev_campaign'] = (health_camp['Camp_Start_Date'] - health_camp['Camp_End_Date'].shift(1)).dt.days
train = pd.merge(train, health_camp, on = 'Health_Camp_ID', how = 'left')
test = pd.merge(test, health_camp, on = 'Health_Camp_ID', how = 'left')
D_COL = 'Registration_Date'
for df_tmp in [train, test]:
  df_tmp[D_COL] = pd.to_datetime(df_tmp[D_COL])
test_min_date = test[D_COL].min()
np.intersect1d(train['Patient_ID'], test['Patient_ID']).shape
train.columns
train.head(1)
var_names = ['Var'+str(i) for i in range(1,6)]
sums_tr = [] 
subs_tr = [] 
mul_tr = [] 
div_tr = []
sums_ts = []
subs_ts = []
mul_ts = []
div_ts = []
for row in train[var_names].itertuples():
    sums_tr.append(row[1]+row[2]+row[3]+row[4]+row[5])
    subs_tr.append(row[1]-row[2]-row[3]-row[4]-row[5])
    mul_tr.append(row[1]*row[2]*row[3]*row[4]*row[5])
    #div_tr.append(row[1]/row[2]/row[3]/row[4]/row[5])
    
for row2 in test[var_names].itertuples():
    sums_ts.append(row[1]+row[2]+row[3]+row[4]+row[5])
    subs_ts.append(row[1]-row[2]-row[3]-row[4]-row[5])
    mul_ts.append(row[1]*row[2]*row[3]*row[4]*row[5])
    #div_ts.append(row[1]/row[2]/row[3]/row[4]/row[5])
    
    
train['sum_VARS'] = sums_tr
train['sub_VARS'] = subs_tr
train['mul_VARS'] = mul_tr
#train['div_VARS'] = div_tr
test['sum_VARS'] = sums_ts
test['sub_VARS'] = subs_ts
test['mul_VARS'] = mul_ts
#test['div_VARS'] = div_ts

### Getting a train and validation split, similar to test data

trn = train[train[D_COL] < test_min_date]
val = train[train[D_COL] >= test_min_date]
trn['target']
TARGET_COL = 'target'
features = [c for c in trn.columns if c not in ['Health_Camp_ID', 'Registration_Date', TARGET_COL, 'id', 'Camp_Start_Date', 'Camp_End_Date', 'First_Interaction']]
print(len(features))
print(features)
trn[features].dtypes
from sklearn.model_selection import RandomizedSearchCV
gridParams = {
    'learning_rate': np.arange(0.01,0.05,0.01),
    'n_estimators': np.arange(200,1400,100),
    'num_leaves': np.arange(50,150,10),
    'max_depth' : np.arange(4,15),
    'colsample_bytree' : np.arange(0.4,0.8,0.1),
    'subsample' : np.arange(0.4,0.8,0.1),
    'min_split_gain' :  np.arange(0.1,0.8,0.1),
    'min_data_in_leaf':np.arange(4,15),
    'metric':['auc']
    }
clf = LGBMClassifier()
grid = RandomizedSearchCV(clf,gridParams,verbose=10,cv=5,n_jobs = -1,n_iter=10)
grid.fit(train[features],train['target'])
print(grid.best_params_)
clf = LGBMClassifier(n_estimators=1100, learning_rate=0.01, verbose=1,
                     random_state=1, colsample_bytree=0.7, reg_alpha=0, reg_lambda=0,
                     min_data_in_leaf=7,min_split_gain=0.2,max_depth=10,num_leaves=130,subsample=0.4)

clf.fit(trn[features], trn[TARGET_COL], eval_set=[(val[features], val[TARGET_COL])], verbose=50,
        eval_metric = 'auc', early_stopping_rounds = 50)

preds = clf.predict_proba(test[features])[:, 1]
trn[features].columns
fi = pd.Series(index = features, data = clf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind = 'barh')
ss['Outcome'] = preds
SUB_FILE_NAME = 'submission.csv'
ss.to_csv(SUB_FILE_NAME, index=False)
xgb1 = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgb1.fit(trn[features], trn[TARGET_COL], eval_set=[(val[features], val[TARGET_COL])], verbose=50,
        eval_metric = 'auc',early_stopping_rounds=50)
pre = xgb1.predict_proba(test[features])[:, 1]
ss['Outcome'] = pre
SUB_FILE_NAME = 'sub.csv'
ss.to_csv(SUB_FILE_NAME, index=False)
model = CatBoostClassifier(iterations=1000,verbose=1, eval_metric='AUC',)
model.fit(train[features],train['target'],verbose=True)
cat = model.predict_proba(test[features])[:, 1]
ss['Outcome'] = cat
SUB_FILE_NAME = 'submission_jantahack_healthcare3.csv'
ss.to_csv(SUB_FILE_NAME, index=False)
reds = 0
for seed_val in [1,3,10,15,20,33,333,1997,2020,2021]:
    print (seed_val)
    m=LGBMClassifier(n_estimators=450,learning_rate=0.03,random_state=seed_val,colsample_bytree=0.5,reg_alpha=2,reg_lambda=2)
    m.fit(train[features],train['target'])
    predict=m.predict_proba(test[features])[:,1]
    preds += predict
preds = preds/10
ss['Outcome'] = preds
SUB_FILE_NAME = 'submission_jantahack.csv'
ss.to_csv(SUB_FILE_NAME, index=False)
