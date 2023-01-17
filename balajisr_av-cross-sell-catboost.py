# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing, metrics, model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from catboost import Pool, CatBoostClassifier, cv
train = pd.read_csv('/kaggle/input/avcrosssell/train.csv')
test = pd.read_csv('/kaggle/input/avcrosssell/test.csv')
train.head()
train['Vehicle_Damage_int'] = np.where(train['Vehicle_Damage']=="Yes",1,0)
test['Vehicle_Damage_int'] = np.where(test['Vehicle_Damage']=="Yes",1,0)

train['Vehicle_Age_int'] = np.where(train['Vehicle_Age']=="< 1 Year",0.5
                                ,np.where(train['Vehicle_Age']=="1-2 Year",1.5,2.5))
test['Vehicle_Age_int'] = np.where(test['Vehicle_Age']=="< 1 Year",0.5
                                ,np.where(test['Vehicle_Age']=="1-2 Year",1.5,2.5))
train_test = train.append(test,sort=False)

stats=train_test.groupby('Policy_Sales_Channel').agg({'Previously_Insured':['mean']}).reset_index()
stats.columns=['Policy_Sales_Channel','PSC_Mean_Insured']
train = pd.merge(train,stats,how='left',on='Policy_Sales_Channel')
test = pd.merge(test,stats,how='left',on='Policy_Sales_Channel')

stats=train_test.groupby('Region_Code').agg({'Previously_Insured':['mean']}).reset_index()
stats.columns=['Region_Code','Region_Mean_Insured']
train = pd.merge(train,stats,how='left',on='Region_Code')
test = pd.merge(test,stats,how='left',on='Region_Code')

stats=train_test.groupby('Policy_Sales_Channel').agg({'Vehicle_Damage_int':['mean']}).reset_index()
stats.columns=['Policy_Sales_Channel','PSC_Mean_Vehicle_Damage']
train = pd.merge(train,stats,how='left',on='Policy_Sales_Channel')
test = pd.merge(test,stats,how='left',on='Policy_Sales_Channel')

stats=train_test.groupby('Region_Code').agg({'Vehicle_Damage_int':['mean']}).reset_index()
stats.columns=['Region_Code','Region_Mean_Vehicle_Damage']
train = pd.merge(train,stats,how='left',on='Region_Code')
test = pd.merge(test,stats,how='left',on='Region_Code')

stats=train_test.groupby('Policy_Sales_Channel').agg({'Vehicle_Age_int':['mean']}).reset_index()
stats.columns=['Policy_Sales_Channel','PSC_Mean_Vehicle_Age']
train = pd.merge(train,stats,how='left',on='Policy_Sales_Channel')
test = pd.merge(test,stats,how='left',on='Policy_Sales_Channel')

stats=train_test.groupby('Region_Code').agg({'Vehicle_Age_int':['mean']}).reset_index()
stats.columns=['Region_Code','Region_Mean_Vehicle_Age']
train = pd.merge(train,stats,how='left',on='Region_Code')
test = pd.merge(test,stats,how='left',on='Region_Code')


stats=train_test.groupby('Vehicle_Damage').agg({'Previously_Insured':['mean']}).reset_index()
stats.columns=['Vehicle_Damage','Vehicle_Damage_Mean_Insured']
train = pd.merge(train,stats,how='left',on='Vehicle_Damage')
test = pd.merge(test,stats,how='left',on='Vehicle_Damage')

stats=train_test.groupby('Policy_Sales_Channel').agg({'Annual_Premium':['mean']}).reset_index()
stats.columns=['Policy_Sales_Channel','PSC_Mean_Premium']
train = pd.merge(train,stats,how='left',on='Policy_Sales_Channel')
test = pd.merge(test,stats,how='left',on='Policy_Sales_Channel')

stats=train_test.groupby('Region_Code').agg({'Annual_Premium':['mean']}).reset_index()
stats.columns=['Region_Code','Region_Mean_Premium']
train = pd.merge(train,stats,how='left',on='Region_Code')
test = pd.merge(test,stats,how='left',on='Region_Code')

stats=train_test.groupby('Policy_Sales_Channel').agg({'Age':['mean']}).reset_index()
stats.columns=['Policy_Sales_Channel','PSC_Mean_Age']
train = pd.merge(train,stats,how='left',on='Policy_Sales_Channel')
test = pd.merge(test,stats,how='left',on='Policy_Sales_Channel')

stats=train_test.groupby('Region_Code').agg({'Age':['mean']}).reset_index()
stats.columns=['Region_Code','Region_Mean_Age']
train = pd.merge(train,stats,how='left',on='Region_Code')
test = pd.merge(test,stats,how='left',on='Region_Code')
train.head()
train_id = train['id']
test_id=test['id']
target = train['Response']
train.drop(['Response','id'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)
train['vehicle_damaged_no_insurance']=np.where((train['Vehicle_Damage']=='Yes') & (train['Previously_Insured']<=0),"Vehicle_Damaged_No_Insurance","Others")
test['vehicle_damaged_no_insurance']=np.where((test['Vehicle_Damage']=='Yes') & (test['Previously_Insured']<=0),"Vehicle_Damaged_No_Insurance","Others")
train['vehicle_damaged_insurance']=np.where((train['Vehicle_Damage']=='Yes') & (train['Previously_Insured']>0),"Vehicle_Damaged_Insurance","Others")
test['vehicle_damaged_insurance']=np.where((test['Vehicle_Damage']=='Yes') & (test['Previously_Insured']>0),"Vehicle_Damaged_Insurance","Others")
train['vehicle_damaged_vehicle_age']=np.where((train['Vehicle_Damage']=='Yes') & (train['Vehicle_Age']=='> 2 Years'),1,0)
test['vehicle_damaged_vehicle_age']=np.where((test['Vehicle_Damage']=='Yes') & (test['Vehicle_Age']=='> 2 Years'),1,0)

train['Policy_Sales_Channel'] = train['Policy_Sales_Channel'].astype(str)
test['Policy_Sales_Channel'] = test['Policy_Sales_Channel'].astype(str)
train['Region_Code'] = train['Region_Code'].astype(str)
test['Region_Code'] = test['Region_Code'].astype(str)
train['Previously_Insured'] = train['Previously_Insured'].astype(str)
test['Previously_Insured'] = test['Previously_Insured'].astype(str)
train['Driving_License'] = train['Driving_License'].astype(str)
test['Driving_License'] = test['Driving_License'].astype(str)
train['damage_insurance']=train['Vehicle_Damage']+ "-" + train['Previously_Insured']
test['damage_insurance']=test['Vehicle_Damage']+ "-" + test['Previously_Insured']
train['vehicle_damage_age']=train['Vehicle_Damage']+ "-" + train['Vehicle_Age']
test['vehicle_damage_age']=test['Vehicle_Damage']+ "-" + test['Vehicle_Age']
train['insurance_vehicle_age']=train['Previously_Insured']+ "-" + train['Vehicle_Age']
test['insurance_vehicle_age']=test['Previously_Insured']+ "-" + test['Vehicle_Age']
train['insurance_psc']=train['Previously_Insured']+ "-" + train['Policy_Sales_Channel']
test['insurance_psc']=test['Previously_Insured']+ "-" + test['Policy_Sales_Channel']
train['insurance_region']=train['Previously_Insured']+ "-" + train['Region_Code']
test['insurance_region']=test['Previously_Insured']+ "-" + test['Region_Code']
train['damage_psc']=train['Vehicle_Damage']+ "-" + train['Policy_Sales_Channel']
test['damage_psc']=test['Vehicle_Damage']+ "-" + test['Policy_Sales_Channel']
train['damage_region']=train['Vehicle_Damage']+ "-" + train['Region_Code']
test['damage_region']=test['Vehicle_Damage']+ "-" + test['Region_Code']
train['Channel_Region']=train['Policy_Sales_Channel']+ "-" + train['Region_Code']
test['Channel_Region']=test['Policy_Sales_Channel']+ "-" + test['Region_Code']

train['vehicle_damage_age']=train['Vehicle_Damage']+ "-" + train['Vehicle_Age']
test['vehicle_damage_age']=test['Vehicle_Damage']+ "-" + test['Vehicle_Age']
fold_cnt=5
fold=StratifiedKFold(n_splits=fold_cnt,shuffle=True,random_state=88)
cv_scores = []
pred_test_full = 0
pred_val_full = np.zeros(train.shape[0])
fold_no=np.zeros(train.shape[0])
enc_test = 0
print("Build Model...")
j=0
cat_columns=train.select_dtypes(include='object').columns.tolist()
feature_importance_df = pd.DataFrame()
for train_index,val_index in fold.split(train,target):
    j=j+1
    train_x,val_x = train.iloc[train_index,:],train.iloc[val_index,:]
    train_y,val_y=target[train_index],target[val_index]
    model = CatBoostClassifier(iterations=10000,loss_function='Logloss',eval_metric='AUC',learning_rate=0.05,od_type='Iter'
                               ,od_wait=200)
    cat_model = model.fit(train_x, train_y, eval_set=(val_x, val_y),cat_features=cat_columns, use_best_model=True,verbose_eval=200)
    pred_val = cat_model.predict_proba(val_x)[:,-1]
    pred_test = cat_model.predict_proba(test)[:,-1]
    pred_val_full[val_index] = pred_val
    fold_no[val_index]=j
    pred_test_full = pred_test_full + pred_test
    loss =  metrics.roc_auc_score(val_y, pred_val)
    cv_scores.append(loss)
    print(cv_scores)
pred_test_full /= fold_cnt
print(metrics.roc_auc_score(target, pred_val_full))
test_df = pd.DataFrame({"id":test_id})
test_df["Response"] = (pred_test_full-pred_test_full.min())/(pred_test_full.max()-pred_test_full.min())
test_df.to_csv("sub1.csv", index=False)