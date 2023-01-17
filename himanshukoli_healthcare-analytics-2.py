
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
import seaborn as sns
from collections import Counter
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 10)

df_train = pd.read_csv('../input/janatahack-healthcare-analytics-ii/Train/train.csv', header=0)
df_test = pd.read_csv('../input/janatahack-healthcare-analytics-ii/test.csv', header=0)

le_stay = LabelEncoder()

df_train['Stay'] = le_stay.fit_transform(df_train['Stay'])

df_train['train_flag'] = 1
df_test['train_flag'] = 0

df_test['Stay'] = -1
print(df_train.shape, df_test.shape)



df_data = pd.concat((df_train, df_test))
print(df_data.shape)

df_data['Bed Grade'] = np.where(df_data['Bed Grade'].isna(),99,df_data['Bed Grade'])
df_data['City_Code_Patient'] = np.where(df_data['City_Code_Patient'].isna(),99,df_data['City_Code_Patient'])

df_data['Age'].unique()
train_majority_age = ['21-30','11-20','31-40','51-60']
test_majority_age = ['21-30','31-40','51-60','41-50','61-70','71-80']


df_data['Age_maj_train'] = df_data['Age'].apply(lambda x : 1 if x in train_majority_age else 0 )
df_data['Age_maj_test'] = df_data['Age'].apply(lambda x : 1 if x in test_majority_age else 0 )

def amt_bands(row):
    if row['Admission_Deposit'] <= 3000:
        return 1
    elif row['Admission_Deposit'] > 3000 and row['Admission_Deposit'] <= 4200:
        return 2
    elif row['Admission_Deposit'] > 4200 and row['Admission_Deposit'] <=4700:
        return 3
    elif row['Admission_Deposit'] > 4700 and row['Admission_Deposit'] <=5500:
        return 4
    elif row['Admission_Deposit'] > 5500:
        return 5
    
df_data = df_data.assign(Admission_band = df_data.apply(amt_bands,axis=1))

df_data['cases_cnt_per_patientId']=df_data.groupby(['patientid'])['case_id'].transform('nunique')

df_data['cases_cnt_per_patientId per hospital']=df_data.groupby(['patientid','Hospital_code'])['case_id'].transform('nunique')

df_data['cases_cnt_per_patientId per severity']=df_data.groupby(['patientid','Severity of Illness'])['case_id'].transform('nunique')

df_data['cases_cnt_per_patientId per admissiontype']=df_data.groupby(['patientid','Type of Admission'])['case_id'].transform('nunique')

df_data['cases_cnt_per_patientId per department']=df_data.groupby(['patientid','Department'])['case_id'].transform('nunique')


#amount_band and case id

df_data['amount band cnt patient by hospital code'] = df_data.groupby(['patientid','Hospital_code'])['Admission_band'].transform('nunique')

df_data['amount band age group cases'] = df_data.groupby(['Admission_band','Age'])['case_id'].transform('nunique')
df_data['amount band age group admission type case ids'] = df_data.groupby(['Admission_band','Type of Admission'])['case_id'].transform('nunique')

df_data['cases cnt by amount band'] = df_data.groupby(['Admission_band'])['case_id'].transform('nunique')
df_data['amout band hospital group'] = df_data.groupby(['Admission_band'])['Hospital_code'].transform('nunique')
df_data['amout band severity group'] = df_data.groupby(['Admission_band'])['Severity of Illness'].transform('nunique')

df_data['amount band bed grade group'] = df_data.groupby(['Admission_band'])['Bed Grade'].transform('mean')

df_data['severity_cnt_per_patientid'] = df_data.groupby(['patientid','Hospital_code'])['Severity of Illness'].transform('nunique')

df_data['addmissiontype_cnt_per_patientid'] = df_data.groupby(['patientid','Hospital_code'])['Type of Admission'].transform('nunique')

df_data['department_cnt_per_patientid'] = df_data.groupby(['patientid','Hospital_code'])['Department'].transform('nunique')

df_data['avg visitors per patient per hospital'] = df_data.groupby(['patientid','Hospital_code'])['Visitors with Patient'].transform('mean')

df_data['avg visitors per patient per hospital per admission'] = df_data.groupby(['patientid','Hospital_code','Type of Admission'])['Visitors with Patient'].transform('mean')

df_data['avg visitors per patient per hospital per severity'] = df_data.groupby(['patientid','Hospital_code','Severity of Illness'])['Visitors with Patient'].transform('mean')

df_data['Total money spent per patient per hospital']=df_data.groupby(['patientid','Hospital_code'])['Admission_Deposit'].transform('sum')

df_data['Total money spent per patient per hospital per dept'] =  df_data.groupby(['patientid','Hospital_code','Department'])['Admission_Deposit'].transform('sum')

df_data['Total money spent per patient per hospital per severity'] =  df_data.groupby(['patientid','Hospital_code','Severity of Illness'])['Admission_Deposit'].transform('sum')

df_data['Total money spent per patient per hospital per admission type'] =  df_data.groupby(['patientid','Hospital_code','Type of Admission'])['Admission_Deposit'].transform('sum')

#ageband vars

df_data['Age group patients_id'] = df_data.groupby(['Age'])['patientid'].transform('nunique')
df_data['Age group amt deposit'] = df_data.groupby(['Age'])['Admission_Deposit'].transform('mean')
df_data['Age group hospital code'] = df_data.groupby(['Age'])['Hospital_code'].transform('nunique')
df_data['Age group severity '] = df_data.groupby(['Age'])['Severity of Illness'].transform('nunique')
df_data['Age group admission type'] = df_data.groupby(['Age'])['Type of Admission'].transform('nunique')
df_data['Age group admission type case ids'] = df_data.groupby(['Age','Type of Admission'])['case_id'].transform('nunique')



df_data['number of hospital codes per patient per severity'] = df_data.groupby(['patientid','Severity of Illness'])['Hospital_code'].transform('nunique')

df_data['avg bed grade per patient per hospital'] =  df_data.groupby(['patientid','Hospital_code'])['Bed Grade'].transform('mean')

df_data['avg bed grade per patient per hospital severity of ilnness'] =  df_data.groupby(['patientid','Hospital_code','Severity of Illness'])['Bed Grade'].transform('mean')

df_data['avg bed grade per patient per hospital Department'] =  df_data.groupby(['patientid','Hospital_code','Department'])['Bed Grade'].transform('mean')

df_data['extra beds per hospital per patient avg']=  df_data.groupby(['patientid','Hospital_code'])['Available Extra Rooms in Hospital'].transform('mean')

df_data['extra beds per hospital per patient per dept avg']=  df_data.groupby(['patientid','Hospital_code','Department'])['Available Extra Rooms in Hospital'].transform('mean')

df_data['extra beds per hospital per patient per severity avg']=  df_data.groupby(['patientid','Hospital_code', 'Severity of Illness'])['Available Extra Rooms in Hospital'].transform('mean')


df_data.columns
df_data.isnull().sum()
df_data["City_Code_Patient"] = df_data["City_Code_Patient"].astype(np.int) 



categorical = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type',
       'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness',
       'Age','Hospital_code','City_Code_Hospital','City_Code_Patient']

df_data = pd.get_dummies(data=df_data, columns = categorical)

feature_cols = df_data.columns.tolist()
feature_cols.remove('patientid')
feature_cols.remove('case_id')
feature_cols.remove('Stay')
feature_cols.remove('train_flag')

label_col = 'Stay'
print(feature_cols)
from catboost import Pool, CatBoostClassifier
model = CatBoostClassifier(
    iterations=759,
    learning_rate=0.06,
    random_strength=0.1,
    depth=10,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    leaf_estimation_method='Newton',
    task_type="GPU",
    devices='0:1'
)


df_train, df_test = df_data[df_data.train_flag == 1], df_data[df_data.train_flag == 0]

df_train.drop(['train_flag'], inplace=True, axis=1)
df_test.drop(['train_flag'], inplace=True, axis=1)
df_test.drop([label_col], inplace=True, axis=1)

print(df_train.shape, df_test.shape)

df_train_, df_eval = train_test_split(df_train, test_size=0.25, random_state=42, shuffle=True, stratify=df_train[label_col])
# fit the model with the training data
model.fit(df_train_[feature_cols], df_train_[label_col],cat_features = categorical,plot=True,
          eval_set=[ (df_eval[feature_cols], df_eval[label_col])]
         )
print('\n Model Trainied')


eval_score = accuracy_score(df_eval[label_col], model.predict(df_eval[feature_cols]))

print('Eval ACC: {}'.format(eval_score))

best_iter = model.best_iteration_
params['n_estimators'] = best_iter
print(params)
model = CatBoostClassifier(
    iterations=610,
    learning_rate=0.06,
    random_strength=0.1,
    depth=10,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    leaf_estimation_method='Newton',
    task_type="GPU",
    devices='0:1'
)
df_train_full = pd.concat((df_train_, df_eval))

model.fit(df_train_full[feature_cols], df_train_full[label_col], cat_features = categorical,plot=True)

eval_score_acc = accuracy_score(df_train_full[label_col], model.predict(df_train_full[feature_cols]))


print('ACC: {}'.format(eval_score_acc))

preds = model.predict(df_test[feature_cols])

pred_actual = le_stay.inverse_transform(preds)
#catboost submission
submission = pd.DataFrame({'case_id':df_test['case_id'], 'Stay':pred_actual})
submission.to_csv("catboost submission .csv",index=False)


params = {}
params['learning_rate'] = 0.08
params['max_depth'] = 60
params['n_estimators'] = 750
params['objective'] = 'multiclass'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.7
params['random_state'] = 42
params['colsample_bytree']=0.75
params['min_data_in_leaf'] = 55
params['reg_alpha'] = 1.1
params['reg_lambda'] = 1.1
params['device']= 'gpu'
params['gpu_platform_id']= 0,
params['gpu_device_id']= 0
params['is_unbalance'] = True
#params['max_bin'] = 64
#params['num_leaves'] = 512
#params['class_weight']: {0: 0.44, 1: 0.4, 2: 0.37}

clf = lgb.LGBMClassifier(**params)
    
clf.fit(df_train_[feature_cols], df_train_[label_col], early_stopping_rounds=100, eval_set=[(df_train_[feature_cols], df_train_[label_col]), (df_eval[feature_cols], df_eval[label_col])],
        eval_metric='multi_error', verbose=True)

eval_score = accuracy_score(df_eval[label_col], clf.predict(df_eval[feature_cols]))

print('Eval ACC: {}'.format(eval_score))

best_iter = clf.best_iteration_
params['n_estimators'] = best_iter
print(params)

best_iter
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
model_xgb = xgb.XGBClassifier(learning_rate =0.08,
 n_estimators=500,
 max_depth=20,
 min_child_weight=55,
 gamma=0,
 subsample=0.7,
 colsample_bytree=0.75,
 objective="multi:softprob",
 nthread=8,
 scale_pos_weight=1,
 reg_alpha = 1.1,
 seed=47,silent=1,num_class=10 ,tree_method = "gpu_hist",n_gpus= 1)

model_xgb.fit(df_train_[feature_cols], df_train_[label_col], early_stopping_rounds=100, eval_set=[(df_train_[feature_cols], df_train_[label_col]), (df_eval[feature_cols], df_eval[label_col])],
        eval_metric='mlogloss', verbose=True)

eval_score = accuracy_score(df_eval[label_col], model_xgb.predict(df_eval[feature_cols]))

print('Eval ACC XGB: {}'.format(eval_score))




df_train_full = pd.concat((df_train_, df_eval))

model_xgb = xgb.XGBClassifier(learning_rate =0.08,
 n_estimators=204,
 max_depth=20,
 min_child_weight=55,
 gamma=0,
 subsample=0.7,
 colsample_bytree=0.75,
 objective="multi:softprob",
 nthread=8,
 scale_pos_weight=1,
 reg_alpha = 1.1,
 seed=47,silent=1,num_class=10 ,tree_method = "gpu_hist",n_gpus= 1)

#clf = lgb.LGBMClassifier(**params)

model_xgb.fit(df_train_full[feature_cols], df_train_full[label_col], eval_metric='mlogloss', verbose=True)

# eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))
eval_score_acc = accuracy_score(df_train_full[label_col], model_xgb.predict(df_train_full[feature_cols]))


print('ACC: {}'.format(eval_score_acc))

preds_xgb = model_xgb.predict(df_test[feature_cols])

pred_actual = le_stay.inverse_transform(preds_xgb)

df_train_full = pd.concat((df_train_, df_eval))

clf = lgb.LGBMClassifier(**params)

clf.fit(df_train_full[feature_cols], df_train_full[label_col], eval_metric='multi_error', verbose=True)

# eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))
eval_score_acc = accuracy_score(df_train_full[label_col], clf.predict(df_train_full[feature_cols]))


print('ACC: {}'.format(eval_score_acc))

preds_lgb = clf.predict(df_test[feature_cols])

pred_actual = le_stay.inverse_transform(preds_lgb)
len(feature_cols)

submission = pd.DataFrame({'case_id':df_test['case_id'], 'Stay':pred_actual})
submission.to_csv("XGBOOSTING latest final submission .csv",index=False)

#from xgboost import plot_importance
#plot_importance(model_xgb)
#from matplotlib import pyplot
#pyplot.show()
#Feature_imp = pd.Series(model_xgb.feature_importances_,index=df_data.columns).sort_values(ascending=False)

#dfm_xgboost = pd.DataFrame(Feature_imp)
#dfm_xgboost.to_csv('xgboosting_importancefinal_60.csv')


plt.rcParams['figure.figsize'] = (150, 50)
lgb.plot_importance(clf)
plt.show()

pred_cat = model.predict(df_test[feature_cols])

pred_cat
preds_xgb = model_xgb.predict(df_test[feature_cols])
pred_ligbm = clf.predict(df_test[feature_cols])
preds_xgb
preds_lgb
voted_pred = pd.DataFrame()
voted_pred['pred_lgb'] = preds_lgb
voted_pred['pred_xgb'] = preds_xgb
voted_pred['pred_cat']= pred_cat
results = np.argmax(voted_pred)#
voted_pred
final_preds = voted_pred.mode(axis=1)
preds_lgb
final_preds_numeric = final_preds[0].astype(np.int) 
final_actual = le_stay.inverse_transform(final_preds_numeric)

submission = pd.DataFrame({'case_id':df_test['case_id'], 'Stay':final_actual})
submission.to_csv("Voting_submission.csv",index=False)
