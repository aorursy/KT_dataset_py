# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sample = pd.read_csv('/kaggle/input/health-care-analysis/sample_submmission.csv')
pp = pd.read_csv('/kaggle/input/health-care-analysis/Patient_Profile.csv')
pp.head()
pp.shape
pp.describe()
pp.apply(lambda x: len(x.unique()))
pp.dtypes
pp.isna().sum()
pp.isna().sum() / pp.shape[0]
pp.head()
cat_columns=['Online_Follower', 'Online_Follower','Twitter_Shared','Facebook_Shared','Income']
figure, ax = plt.subplots(1, 5,figsize=(15, 4))
sns.countplot(pp['Online_Follower'], ax=ax[0])
sns.countplot(pp['Online_Follower'], ax=ax[1])
sns.countplot(pp['Twitter_Shared'], ax=ax[2])
sns.countplot(pp['Facebook_Shared'], ax=ax[3])
sns.countplot(pp['Income'], ax=ax[4])
figure.show()
pp['Income'].value_counts()
pp[['Income', 'Education_Score', 'Age']] = pp[['Income', 'Education_Score', 'Age']].apply(lambda x: x.str.replace('None', 'NaN').astype('float'))
pp.head()
health = pd.read_csv('/kaggle/input/health-care-analysis/Health_Camp_Detail.csv')
health.head()
health.shape
health.isna().sum()
health.describe()
health.dtypes
health.apply(lambda x: len(x.unique()))
cat_columns=['Category1', 'Category2','Category3']
figure, ax = plt.subplots(1, 3,figsize=(15, 3))
sns.countplot(health['Category1'], ax=ax[0])
sns.countplot(health['Category2'], ax=ax[1])
sns.countplot(health['Category3'], ax=ax[2])
figure.show()
health['Camp_Start_Date'] = pd.to_datetime(health['Camp_Start_Date'])
health['Camp_End_Date'] = pd.to_datetime(health['Camp_End_Date'])
health['Number of days'] = (health['Camp_End_Date'] - health['Camp_Start_Date']).dt.days
health.head()
health['Category1'] = health['Category1'].map({'First': 1, 'Second': 2, 'Third': 3})

 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
health['Category2']= label_encoder.fit_transform(health['Category2']) 
  
health['Category2'].unique() 
health.head()
train = pd.read_csv('/kaggle/input/health-care-analysis/Train.csv')
train
train.shape
train.apply(lambda x:len(x.unique()))
train['id'] = train['Patient_ID'].astype('str') + "-"+ train['Health_Camp_ID'].astype('str')
train.head()
test = pd.read_csv('/kaggle/input/health-care-analysis/test_l0Auv8Q.csv')
test
test['id'] = test['Patient_ID'].astype('str') + "-"+ test['Health_Camp_ID'].astype('str')
first = pd.read_csv('/kaggle/input/health-care-analysis/First_Health_Camp_Attended.csv')
second = pd.read_csv('/kaggle/input/health-care-analysis/Second_Health_Camp_Attended.csv')
third = pd.read_csv('/kaggle/input/health-care-analysis/Third_Health_Camp_Attended.csv')
del first['Unnamed: 4']
print(first.columns,second.columns,third.columns)
first.rename(columns = {'Health_Score':'Health Score'},inplace = True)
first['id'] = first['Patient_ID'].astype('str') + "-"+ first['Health_Camp_ID'].astype('str')
second['id'] = second['Patient_ID'].astype('str') + "-"+ second['Health_Camp_ID'].astype('str')
third['id'] = third['Patient_ID'].astype('str') + "-"+ third['Health_Camp_ID'].astype('str')
third = third[third['Number_of_stall_visited'] > 0]
camps = (first.append(second)).append(third)
camps
train['data'] = 'train'
test['data'] = 'test'
df = train.append(test)

df['target'] = 0
df.loc[df['id'].isin(camps['id']), 'target'] = 1
df
df[df['data'] == 'train']['target'].value_counts(normalize=True)
train.columns
health.columns
temp = pd.merge(pd.merge(df,pp,on = 'Patient_ID',how = 'left'),health, on = 'Health_Camp_ID',how = 'left')
temp.columns
temp['Registration_Date'] = pd.to_datetime(temp['Registration_Date'])
trains = temp[temp['data'] == 'train']
tests = temp[temp['data'] == 'test']
trains
tests
a = list(sample['Patient_ID'])
a.sort()

b = list(tests['Patient_ID'])
b.sort()

print(len(a))
print(len(b))
a == b
trains.columns
train_data = trains[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','target','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days']]
train_data
tests['target'].value_counts()
test_data = tests[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days']]
from lightgbm import LGBMClassifier
clf = LGBMClassifier(n_estimators=550, learning_rate=0.05, random_state=1, colsample_bytree=0.5, reg_alpha=2, reg_lambda=2)

clf.fit(train_data.drop('target',axis = 1), train_data['target'])

preds = clf.predict_proba(test_data)[:, 1]
fi = pd.Series(index = test_data.columns, data = clf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind = 'barh')
submission = tests[['Patient_ID', 'Health_Camp_ID']]
submission['Outcome'] = preds
submission.to_csv('LGBM.csv',index = False)
X_train = train_data.drop('target',axis=1)
y_train = train_data['target']
test_data
import lightgbm as lgb
splits = 5
folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=22)
oof_preds = np.zeros((len(test_data), 3))
feature_importance_df = pd.DataFrame()
final_preds = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("Fold {}".format(fold_))
        X_trn,y_trn = X_train.iloc[trn_idx],y_train.iloc[trn_idx]
        X_val,y_val = X_train.iloc[val_idx],y_train.iloc[val_idx]
        clf = lgb.LGBMClassifier(random_state=22,n_jobs=-1,max_depth=-1,min_data_in_leaf=24,num_leaves=49,bagging_fraction=0.01,
                        colsample_bytree=1.0,lambda_l1=1,lambda_l2=11,learning_rate=0.1,n_estimators=5000)
        clf.fit(X_trn, y_trn, eval_metric='auc', eval_set=[(X_val,y_val)], verbose=False,early_stopping_rounds=100)
        y_val_preds = clf.predict_proba(X_val)
        final_preds.append(f1_score(y_pred=[np.argmax(x) for x in y_val_preds],y_true=y_val,average='weighted'))
        print(final_preds)
#         predictions += clf.predict_proba(X_valid)
        oof_preds = (clf.predict_proba(test_data)[:, 1])
#         counter = counter + 1
oof_preds = oof_preds/splits
print(sum(final_preds)/5)
submission['Outcome'] = oof_preds
submission.to_csv('LGBM(II).csv',index = False)
from catboost import CatBoostClassifier
# Set up folds
K = 5
skf = StratifiedKFold(n_splits = K, random_state = 7, shuffle = True)
cat_columns=X_train.select_dtypes(include='object').columns.tolist()
X = X_train
y = y_train
X_test = test_data
y_valid_pred = 0*y
y_test_pred = 0
accuracy = 0
result={}
#fitting catboost classifier model
j=1
model = CatBoostClassifier(n_estimators=1000,verbose=False,learning_rate=0.1)
for train_index, test_index in skf.split(X, y):  
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", j)
    
    # Run model for this fold
    fit_model = model.fit( X_train, y_train, eval_set=(X_valid, y_valid),cat_features=cat_columns, use_best_model=True)
    print( "  N trees = ", model.tree_count_ )
    pred = fit_model.predict(X_valid)
    y_valid_pred.iloc[test_index] = pred.reshape(-1)
    print(accuracy_score(y_valid,pred))
    accuracy+=accuracy_score(y_valid,pred)
    # Accumulate test set predictions
    y_test_pred += fit_model.predict(X_test)
    result[j]=fit_model.predict(X_test)
    j+=1
results = y_test_pred / K  # Average test set predictions
print(accuracy/5)
submission['Outcome'] = list(fit_model.predict_proba(X_test)[:,1])
submission.to_csv('Cat.csv',index = False)

from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
trains['City_Type'] = trains['City_Type'].astype(str)
tests['City_Type'] = tests['City_Type'].astype(str)

trains['Employer_Category'] = trains['Employer_Category'].astype(str)
tests['Employer_Category'] = tests['Employer_Category'].astype(str)

# Encode labels in column 'species'. 
trains['City_Type']= label_encoder.fit_transform(trains['City_Type']) 
trains['Employer_Category']= label_encoder.fit_transform(trains['Employer_Category']) 

tests['City_Type']= label_encoder.fit_transform(tests['City_Type']) 
tests['Employer_Category']= label_encoder.fit_transform(tests['Employer_Category']) 
  


train_data = trains[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days','City_Type',
       'Employer_Category','target']]
test_data = tests[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days','City_Type',
       'Employer_Category']]
clf = LGBMClassifier(n_estimators=550, learning_rate=0.05, random_state=1, colsample_bytree=0.5, reg_alpha=2, reg_lambda=2)

clf.fit(train_data.drop('target',axis = 1), train_data['target'])

preds = clf.predict_proba(test_data)[:, 1]

fi = pd.Series(index = test_data.columns, data = clf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind = 'barh')

submission = tests[['Patient_ID', 'Health_Camp_ID']]
submission['Outcome'] = preds
submission.to_csv('LGBM_all.csv',index = False)
cat_columns=X_train.select_dtypes(include='object').columns.tolist()
X = X_train
y = y_train
X_test = test_data
y_valid_pred = 0*y
y_test_pred = 0
accuracy = 0
result={}
#fitting catboost classifier model
j=1
model = CatBoostClassifier(n_estimators=1000,verbose=False,learning_rate=0.1)
for train_index, test_index in skf.split(X, y):  
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", j)
    
    # Run model for this fold
    fit_model = model.fit( X_train, y_train, eval_set=(X_valid, y_valid),cat_features=cat_columns, use_best_model=True)
    print( "  N trees = ", model.tree_count_ )
    pred = fit_model.predict(X_valid)
    y_valid_pred.iloc[test_index] = pred.reshape(-1)
    print(accuracy_score(y_valid,pred))
    accuracy+=accuracy_score(y_valid,pred)
    # Accumulate test set predictions
    y_test_pred += fit_model.predict(X_test)
    result[j]=fit_model.predict(X_test)
    j+=1
results = y_test_pred / K  # Average test set predictions
print(accuracy/5)
submission['Outcome'] = list(fit_model.predict_proba(X_test)[:,1])
submission.to_csv('Cat_all.csv',index = False)

temp['patient_count'] = temp.groupby('Patient_ID')['Patient_ID'].transform('count')
temp['camp_count'] = temp.groupby('Health_Camp_ID')['Health_Camp_ID'].transform('count')
#temp['patient_camp_count'] = temp.groupby('id')['id'].transform('count')
trains = temp[temp['data'] == 'train']
tests = temp[temp['data'] == 'test']
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
trains['City_Type'] = trains['City_Type'].astype(str)
tests['City_Type'] = tests['City_Type'].astype(str)

trains['Employer_Category'] = trains['Employer_Category'].astype(str)
tests['Employer_Category'] = tests['Employer_Category'].astype(str)

# Encode labels in column 'species'. 
trains['City_Type']= label_encoder.fit_transform(trains['City_Type']) 
trains['Employer_Category']= label_encoder.fit_transform(trains['Employer_Category']) 

tests['City_Type']= label_encoder.fit_transform(tests['City_Type']) 
tests['Employer_Category']= label_encoder.fit_transform(tests['Employer_Category']) 
  


train_data = trains[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days','City_Type',
       'Employer_Category','patient_count','camp_count','target']]
test_data = tests[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days','City_Type',
       'Employer_Category','patient_count','camp_count']]
clf = LGBMClassifier(n_estimators=550, learning_rate=0.05, random_state=1, colsample_bytree=0.5, reg_alpha=2, reg_lambda=2)

clf.fit(train_data.drop('target',axis = 1), train_data['target'])

preds = clf.predict_proba(test_data)[:, 1]

fi = pd.Series(index = test_data.columns, data = clf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind = 'barh')

submission = tests[['Patient_ID', 'Health_Camp_ID']]
submission['Outcome'] = preds
submission.to_csv('LGBM_all_all.csv',index = False)
all_camps = first.append(second).append(third)
all_camps.drop('Last_Stall_Visited_Number',axis = 1, inplace = True)
data = pd.merge(temp,all_camps,on =['Patient_ID', 'Health_Camp_ID','id'],how = 'left')
data.shape
final_train = data[data['data'] == 'train']
final_test = data[data['data'] == 'test']
final_train.columns
train_data = final_train[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days','patient_count','camp_count','Donation', 'Health Score', 'Number_of_stall_visited','target']]
test_data = final_test[['Var1', 'Var2',
       'Var3', 'Var4', 'Var5','Category1', 'Category2', 'Category3', 'Online_Follower', 'LinkedIn_Shared',
       'Twitter_Shared', 'Facebook_Shared', 'Income', 'Education_Score', 'Age','Number of days','patient_count','camp_count','Donation', 'Health Score', 'Number_of_stall_visited']]
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
train_data['City_Type'] = train_data['City_Type'].astype(str)
test_data['City_Type'] = test_data['City_Type'].astype(str)

train_data['Employer_Category'] = train_data['Employer_Category'].astype(str)
test_data['Employer_Category'] = test_data['Employer_Category'].astype(str)

# Encode labels in column 'species'. 
train_data['City_Type']= label_encoder.fit_transform(train_data['City_Type']) 
train_data['Employer_Category']= label_encoder.fit_transform(train_data['Employer_Category']) 

test_data['City_Type']= label_encoder.fit_transform(test_data['City_Type']) 
test_data['Employer_Category']= label_encoder.fit_transform(test_data['Employer_Category']) 
  
train_data
clf = LGBMClassifier(n_estimators=10000, learning_rate=0.05, random_state=42, colsample_bytree=0.5, reg_alpha=2, reg_lambda=2)

clf.fit(train_data.drop('target',axis = 1), train_data['target'])

preds = clf.predict_proba(test_data)[:, 1]

fi = pd.Series(index = test_data.columns, data = clf.feature_importances_)
fi.sort_values(ascending=False)[-20:][::-1].plot(kind = 'barh')

submission = tests[['Patient_ID', 'Health_Camp_ID']]
submission['Outcome'] = preds
submission.to_csv('Light.csv',index = False)
cat_columns=X_train.select_dtypes(include='object').columns.tolist()
X = X_train
y = y_train
X_test = test_data
y_valid_pred = 0*y
y_test_pred = 0
accuracy = 0
result={}
#fitting catboost classifier model
j=1
model = CatBoostClassifier(n_estimators=1000,verbose=False,learning_rate=0.1)
for train_index, test_index in skf.split(X, y):  
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", j)
    
    # Run model for this fold
    fit_model = model.fit( X_train, y_train, eval_set=(X_valid, y_valid),cat_features=cat_columns, use_best_model=True)
    print( "  N trees = ", model.tree_count_ )
    pred = fit_model.predict(X_valid)
    y_valid_pred.iloc[test_index] = pred.reshape(-1)
    print(accuracy_score(y_valid,pred))
    accuracy+=accuracy_score(y_valid,pred)
    # Accumulate test set predictions
    y_test_pred += fit_model.predict(X_test)
    result[j]=fit_model.predict(X_test)
    j+=1
results = y_test_pred / K  # Average test set predictions
print(accuracy/5)
submission['Outcome'] = list(fit_model.predict_proba(X_test)[:,1])
submission.to_csv('Cat_all_pay.csv',index = False)

train_data
from xgboost import XGBClassifier
def Xg_boost(Xtrain,Ytrain,Xtest):
    xg = XGBClassifier()
    xg.fit(Xtrain, Ytrain) 
    xg_prediction = xg.predict(Xtest)
    return xg_prediction
print(train_data.shape,y_train.shape,test_data.shape)
pred_xg = Xg_boost(train_data.drop('target',axis = 1),train_data['target'],test_data)
submission['Outcome'] = pred_xg
submission.to_csv('XG.csv',index = False)
train_data.fillna(-1,inplace = True)
test_data.fillna(-1,inplace = True)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(loss='exponential', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', 
                                  min_samples_split=2, 
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=10, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, 
                                  init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, 
                                  presort='deprecated', 
                                  validation_fraction=0.1, n_iter_no_change=None, tol=0.0001).fit(train_data.drop('target',axis = 1),train_data['target'])
prediction_of_gbc = gbc.predict(test_data)
submission['Outcome'] = prediction_of_gbc
submission.to_csv('prediction_of_gbc.csv',index = False)