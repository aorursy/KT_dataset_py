# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/train_fNxu4vz.csv')
test = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/test_fjtUOL8.csv')
sample = pd.read_csv('/kaggle/input/janatahack-machine-learning-for-banking/sample_submission_HSqiq1Q.csv')
print(train.shape)
print(test.shape)
print(sample.shape)
train.columns
test.columns
categorical_columns = train.select_dtypes(exclude=['int', 'float']).columns
categorical_columns
numerical_columns = train.select_dtypes(include=['int', 'float']).columns
numerical_columns
print(train['Interest_Rate'].value_counts())
sns.countplot(train['Interest_Rate'])
train.isna().sum()
test.isna().sum()
train.isna().sum()/train.shape[0]
test.isna().sum()/test.shape[0]
train['Interest_Rate'] = train['Interest_Rate'].astype(int)
print(train['Months_Since_Deliquency'].skew())
print(train['Months_Since_Deliquency'].kurtosis())
sns.distplot(train['Months_Since_Deliquency'])
df = train.append(test)
df.shape

df.plot(kind = 'box',figsize=(20,5))
df.columns
df['Loan_Amount_Requested'] #Loan applied by the borrower
df['Loan_Amount_Requested'].isna().sum()
df['Loan_Amount_Requested'].dtypes
df['Loan_Amount_Requested'] = df['Loan_Amount_Requested'].str.replace(',', '').astype(int)
print(df['Loan_Amount_Requested'].skew())
print(df['Loan_Amount_Requested'].kurtosis())
sns.distplot(df['Loan_Amount_Requested'])
print(df['Loan_Amount_Requested'].min())
print(df['Loan_Amount_Requested'].max())
fig, ax = plt.subplots(figsize=(5,5))
plt.suptitle('')
df.boxplot(column=['Loan_Amount_Requested'], by='Interest_Rate', ax=ax)
pd.cut(df['Loan_Amount_Requested'],bins = 3)
df['Loan_label'] = pd.cut(x=df['Loan_Amount_Requested'], bins= 3, labels=['Low','Medium','High'], right=True)

print(df['Loan_Amount_Requested'])

print(df['Loan_label'].unique())
dic = {'Low':1,'Medium':2,'High':3}
df['Loan_label'] = df['Loan_label'].map(dic)
print(df['Loan_Amount_Requested'].corr(df['Interest_Rate']))
print(df['Loan_label'].corr(df['Interest_Rate']))
df['Length_Employed']
df['Length_Employed'].value_counts()
df['Length_Employed'] = df['Length_Employed'].replace('10+ years','10 years')
df['Length_Employed'] = df['Length_Employed'].replace('< 1 year','0 years')
df.head()
plt.figure(figsize=(10,5))
sns.countplot(df['Length_Employed'])
df['Length_Employed'].fillna(df['Length_Employed'].mode()[0],inplace = True)
df['Length_Employed'].isna().sum()
plt.figure(figsize=(10,5))
sns.countplot(df['Length_Employed'])
df[['A','B']] = df['Length_Employed'].str.split(" ",expand = True)
df['Length_Employed'] = df['A']
del df['A']
del df['B']
df['Length_Employed'] = df['Length_Employed'].astype(int)
df['Home_Owner'].value_counts()
df.groupby('Home_Owner')['Interest_Rate'].unique()
pd.crosstab(train['Home_Owner'],train['Interest_Rate'])
df['Home_Owner'].isna().sum()/df.shape[0]
sns.countplot(df['Home_Owner'])
df['Home_Owner'].fillna(df['Home_Owner'].fillna(df['Home_Owner'].mode()[0]),inplace = True)
sns.countplot(df['Home_Owner'])
print(df['Annual_Income'].skew())
print(df['Annual_Income'].kurtosis())
sns.distplot(df['Annual_Income'])
df['Annual_Income'].isna().sum()/df['Annual_Income'].shape[0]
print(df['Annual_Income'].min())
print(df['Annual_Income'].max())
df['Annual_Income'].fillna(df['Annual_Income'].median(),inplace = True)
df['Annual_Income']
num = [200000,500000,100000]

print(type(num[0]))


df['Income_label'] = pd.cut(x=df['Annual_Income'], bins= 3, labels=['Low','Medium','High'], right=True)

print(df['Annual_Income'])

print(df['Income_label'].unique())
dic = {'Low':1,'Medium':2,'High':3}
df['Income_label'] = df['Income_label'].map(dic)
df['Income_Verified'].value_counts()
df['Income_Verified'].isna().sum()
sns.countplot(df['Income_Verified'])
df['Purpose_Of_Loan'].isna().sum()
plt.figure(figsize=(25,8))
sns.countplot(df['Purpose_Of_Loan'])
df['Debt_To_Income'].isna().sum()
print(df['Debt_To_Income'].skew())
print(df['Debt_To_Income'].kurtosis())
sns.distplot(df.Debt_To_Income)
df['Inquiries_Last_6Mo'].isna().sum()
sns.countplot(df['Inquiries_Last_6Mo'])
df['Months_Since_Deliquency']
deli = []
for i in df['Months_Since_Deliquency']:
    if pd.isnull(i) == True:
        deli.append(0)
    else:
        deli.append(1)
df['Deliquency'] = deli
df['Deliquency'].value_counts()
df.drop('Months_Since_Deliquency',axis = 1,inplace = True)
df['Number_Open_Accounts'].describe()
df['Number_Open_Accounts'].dtype
plt.figure(figsize = (20,5))
sns.countplot(df['Number_Open_Accounts'])
df['Number_Open_Accounts'].isna().sum()
df['Total_Accounts'].dtype
plt.figure(figsize = (30,5))
sns.countplot(df['Total_Accounts'])
df['Total_Accounts'].isna().sum()
print(df['Total_Accounts'].corr(df['Interest_Rate']))
df['Gender'].value_counts()
df['Gender'].isna().sum()
sns.countplot(df['Gender'])
df['Gender'].isna().sum()
df["Number_Invalid_Acc"] = df["Total_Accounts"] - df["Number_Open_Accounts"]
df["Number_Years_To_Repay_Debt"] = df["Loan_Amount_Requested"]/df["Annual_Income"]

df = df[['Loan_ID','Loan_Amount_Requested','Loan_label','Length_Employed','Home_Owner','Annual_Income','Income_label','Income_Verified','Purpose_Of_Loan','Debt_To_Income','Inquiries_Last_6Mo','Number_Open_Accounts','Total_Accounts','Deliquency','Gender','Number_Invalid_Acc','Number_Years_To_Repay_Debt','Interest_Rate']]
df.head()
trains = df[df['Interest_Rate'].isna() == False] 
tests = df[df['Interest_Rate'].isna() == True]
trains['Interest_Rate'] = trains['Interest_Rate'].astype(int)
trains
trains.groupby('Length_Employed')['Loan_Amount_Requested'].agg(['count','min','max','mean','median','std'])
trains.groupby('Home_Owner')['Loan_Amount_Requested'].agg(['count','min','max','mean','median','std'])
plt.figure(figsize=(12,6))
sns.scatterplot(x=trains['Loan_Amount_Requested'],y=trains['Annual_Income'])
trains.groupby('Purpose_Of_Loan')['Loan_Amount_Requested'].agg(['count','min','max','mean','median','std'])
trains.groupby('Interest_Rate')['Loan_Amount_Requested'].agg(['count','min','max','mean','median','std'])
trains.groupby('Interest_Rate')['Annual_Income'].agg(['count','min','max','mean','median','std'])
df_new = trains.append(tests)
trains.drop('Loan_ID',axis =1,inplace = True )
tests.drop('Loan_ID',axis =1,inplace = True )
X_train, Y = trains.drop(["Interest_Rate"], axis=1).values, trains["Interest_Rate"].astype(int).values
X_test = tests.values

X_train.shape, Y.shape, X_test.shape
trains.head()
kfold, scores = KFold(n_splits=5, shuffle=True, random_state=0), list()
for train, test in kfold.split(X_train):
    x_train, x_test = X_train[train], X_train[test]
    y_train, y_test = Y[train], Y[test]
    
    model = CatBoostClassifier(random_state=27, max_depth=4, n_estimators=1000, verbose=500)
    model.fit(x_train, y_train, cat_features=[1,2,3,5,6,7,12,13])
    preds = model.predict(x_test)
    score = f1_score(y_test, preds, average="weighted")
    scores.append(score)
    print(score)
print("Average: ", sum(scores)/len(scores))
model = CatBoostClassifier(random_state=27, n_estimators=1000, max_depth=4, verbose=500)
model.fit(X_train, Y, cat_features=[1,2,3,5,6,7,12,13])
preds1 = model.predict(X_test)
feat_imp = pd.Series(model.feature_importances_, index=trains.drop(["Interest_Rate"], axis=1).columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))
sample['Interest_Rate'] = preds1
sample.to_csv('Solution_with_Cat.csv',index=False)
cat_columns = ['Home_Owner','Income_Verified','Purpose_Of_Loan','Gender']
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 

for columns in cat_columns:
    df_new[columns]= le.fit_transform(df_new[columns]) 
df_new.head()
col = ['Loan_label','Length_Employed','Home_Owner','Income_label','Income_Verified','Purpose_Of_Loan','Inquiries_Last_6Mo','Deliquency','Gender']
df_new = pd.get_dummies(df_new)
train_df = df_new[df_new['Interest_Rate'].isna() == False] 
test_df = df_new[df_new['Interest_Rate'].isna() == True]
train_df
test_df
train_df.drop('Loan_ID',axis = 1,inplace = True)
test_df.drop('Loan_ID',axis = 1,inplace = True)
train_df['Interest_Rate'] = train_df['Interest_Rate'].astype(int)
test_df.drop('Interest_Rate',axis = 1,inplace = True)
import h2o
h2o.init()

train1 = h2o.H2OFrame(train_df)
test1 = h2o.H2OFrame(test_df)
train1.columns
y = 'Interest_Rate'
x = train1.col_names
x.remove(y)
train1['Interest_Rate'] = train1['Interest_Rate'].asfactor()
train1['Interest_Rate'].levels()
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models = 30, max_runtime_secs=1000, seed = 1)
aml.train(x = x, y = y, training_frame = train1)

preds = aml.predict(test1)
preds
ans=h2o.as_list(preds) 

sample['Interest_Rate'] = ans['predict']
sample.to_csv('Solution_with_autoML.csv',index=False)
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split,RandomizedSearchCV
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#%matplotlib inline 
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

Y = train_df['Interest_Rate']
X = train_df.drop('Interest_Rate',axis = 1)
X1 = pd.get_dummies(X)
X_test = test_df
Y = Y-1
Y.value_counts()
evals_result = {}
feature_imp = pd.DataFrame()
features = [feat for feat in X1.columns]
folds = StratifiedKFold(n_splits=3, shuffle=False, random_state =8736)
param = {
    'boost_from_average':'false',
    'boosting_type': 'gbdt',
    'feature_fraction': 0.54,
    'learning_rate': 0.005,
    'max_depth': -1,  
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 16.0,
    'num_leaves': 40,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'multiclass',
    'num_class': 3,
    'verbosity': 1,
    "n_jobs":-1,
    "metric" : "multi_logloss",
}

predictions = np.zeros((len(X1),3))
predictions_test = np.zeros((len(X_test),3))
X
X_test
for fold_, (train_idx,val_idx) in enumerate(folds.split(X1.values,Y.values)):
    print("Fold {}".format(fold_+1))
    d_train = lgb.Dataset(X1.iloc[train_idx][features], label=Y.iloc[train_idx])
    d_val = lgb.Dataset(X1.iloc[val_idx][features],label=Y.iloc[val_idx])
    num_round = 1000000
    clf = lgb.train(param,d_train,num_round,valid_sets=[d_train,d_val],verbose_eval=1000, early_stopping_rounds=5000,evals_result=evals_result)
    oof = clf.predict(X1.iloc[val_idx][features],num_iteration=clf.best_iteration)
    fold_imp = pd.DataFrame()
    fold_imp["Feature"] = features
    fold_imp["importance"] = clf.feature_importance()
    fold_imp["fold"] = fold_ +1
    feat_imp_df = pd.concat([feature_imp,fold_imp], axis=0)
    predictions += clf.predict(X1, num_iteration=clf.best_iteration)
    predictions_test += clf.predict(X_test, num_iteration=clf.best_iteration)
    pred_lab = pd.DataFrame([np.argmax(pr) for pr in predictions])
    oof_lab = pd.DataFrame([np.argmax(pr) for pr in oof])
    acc_score = accuracy_score(Y,pred_lab)
    oof_acc = accuracy_score(Y.iloc[val_idx],oof_lab)
    print("OOF Accuracy {} and Training Accuracy {}".format(oof_acc,acc_score))
prediction_test_lab = pd.DataFrame([np.argmax(pr) for pr in predictions_test])
prediction_test_lab = prediction_test_lab+1
prediction_test_lab
test = list(df[df["Interest_Rate"].isnull()]["Loan_ID"])
sub = pd.DataFrame({"Loan_ID":test,"Interest_Rate":prediction_test_lab[0]})

train['Interest_Rate'].value_counts()
sub['Interest_Rate'].value_counts()
train_df
test_df
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator
def extra_tree(Xtrain,Ytrain,Xtest):
    extra = ExtraTreesClassifier()
    extra.fit(Xtrain, Ytrain) 
    extra_prediction = extra.predict(Xtest)
    return extra_prediction
def Xg_boost(Xtrain,Ytrain,Xtest):
    xg = XGBClassifier(loss='exponential', learning_rate=0.05, n_estimators=1000, subsample=1.0, criterion='friedman_mse', 
                                  min_samples_split=2, 
                                  min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_depth=10, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, 
                                  init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None, warm_start=False, 
                                  presort='deprecated', 
                                  validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    xg.fit(Xtrain, Ytrain) 
    xg_prediction = xg.predict(Xtest)
    return xg_prediction
def LGBM(Xtrain,Ytrain,Xtest):
    lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=40,
                            max_depth=5, learning_rate=0.05, n_estimators=1000, subsample_for_bin=200, objective='binary', 
                            min_split_gain=0.0, min_child_weight=0.001, min_child_samples=10,
                            subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0,
                            reg_lambda=0.0, random_state=None, n_jobs=1, silent=True, importance_type='split')
    #lgbm = LGBMClassifier(n_estimators= 500)
    lgbm.fit(X_train, Y_train)
    lgbm_preds = lgbm.predict(X_test)
    return lgbm_preds
Y_train = train_df['Interest_Rate']
X_train = train_df.drop('Interest_Rate',axis = 1)
X_test = test_df
pred_xg = Xg_boost(X_train,Y_train,X_test)
pred_et = extra_tree(X_train,Y_train,X_test)
pred_l = LGBM(X_train,Y_train,X_test)
sample['Interest_Rate'] = pred_xg
print(sample['Interest_Rate'].unique())
sample.to_csv('XG.csv',index = False)
sample['Interest_Rate'] = pred_et
print(sample['Interest_Rate'].unique())
sample.to_csv('ET.csv',index = False)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
ans = clf.predict(X_test)
sample['Interest_Rate'] = ans
print(sample['Interest_Rate'].unique())
sample.to_csv('LR.csv',index = False)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10).fit(X_train, Y_train)
prediction_of_rf = rf.predict(X_test)
sample['Interest_Rate'] = prediction_of_rf
print(sample['Interest_Rate'].unique())
sample.to_csv('RF.csv',index = False)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train)

# Predicted class
nri = neigh.predict(X_test)
sample['Interest_Rate'] = nri
print(sample['Interest_Rate'].unique())
sample.to_csv('KNN.csv',index = False)