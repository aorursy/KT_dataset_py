# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv("../input/av-janatahack-healthcare-hackathon-ii/Data/train.csv")
df_test=pd.read_csv("../input/av-janatahack-healthcare-hackathon-ii/Data/test.csv")
df_train.head(30)
df_train.shape
df_test.shape
df_train.columns
df_train.describe()
df_train.info()
df_train.isnull().sum()
df_train.nunique()
#Data visualisation
plt.figure(figsize=(8,6))
plt.title('Department')
sns.countplot(df_train['Department'])
plt.figure(figsize=(8,5))
plt.title('Admission_deposit Distribution')
sns.kdeplot(data=df_train["Admission_Deposit"])
plt.figure(figsize=(9,6))
plt.title('Age Distribution')
sns.countplot(df_train['Age'])
plt.figure(figsize=(8,6))
sns.barplot(x=df_train["Age"],y=df_train["Visitors with Patient"])
plt.figure(figsize=(15,8))
plt.title('Stay Distribution')
sns.countplot(df_train['Stay'])
plt.figure(figsize=(8,6))
sns.barplot(x=df_train["Hospital_region_code"],y=df_train["Admission_Deposit"])
plt.figure(figsize=(15,8))
plt.title('Hospital_type_code Distribution')
sns.countplot(df_train['Hospital_type_code'])
plt.figure(figsize=(15,8))
plt.title('Hospital_region_code Distribution')
sns.countplot(df_train['Hospital_region_code'])
plt.figure(figsize=(15,8))
plt.title('Available Extra Rooms in Hospital')
sns.countplot(df_train['Available Extra Rooms in Hospital'])
df_train["Available Extra Rooms in Hospital"].unique()
plt.figure(figsize=(15,8))
plt.title('Ward_type distribution')
sns.countplot(df_train['Ward_Type'])
plt.figure(figsize=(15,8))
plt.title('Ward_Facility_Code distribution')
sns.countplot(df_train['Ward_Facility_Code'])
plt.figure(figsize=(15,8))
plt.title('Bed Grade distribution')
sns.countplot(df_train['Bed Grade'])
plt.figure(figsize=(15,8))
plt.title('Visitors with Patient distribution')
sns.countplot(df_train['Visitors with Patient'])
df_train["Visitors with Patient"].unique()
plt.figure(figsize=(15,8))
plt.title('Admission_Deposit distribution')
sns.boxplot(df_train['Admission_Deposit'])
plt.figure(figsize=(15,8))
plt.title('Visitors with Patient')
sns.boxplot(df_train['Visitors with Patient'])
plt.figure(figsize=(15,8))
plt.title('Available Extra Rooms in Hospital distribution')
sns.boxplot(df_train['Available Extra Rooms in Hospital'])
age=df_train.groupby("Age")
age.describe()
#Handling missing values
df_train['is_train'] = 1
df_test['is_train'] = 0
df_total=pd.concat([df_train,df_test])
df_total.isnull().sum()
df_total.City_Code_Patient.head(20)
df_total.City_Code_Patient=df_total.City_Code_Patient.fillna(method="ffill",axis=0)
df_total["Bed Grade"].head(20)
df_total["Bed Grade"]=df_total["Bed Grade"].fillna(0)
df_total.isnull().sum()
df_total
#Binning
#Feature generation
#df_total['Bill_per_patient'] = df_total.groupby('patientid')['Admission_Deposit'].transform('sum')
df_total
#Scaling 
from sklearn import preprocessing
df_total["Admission_Deposit"]
scaler=preprocessing.MinMaxScaler()
df_total["Admission_Deposit"]=scaler.fit_transform(df_total[["Admission_Deposit"]])
df_total["Admission_Deposit"]
#Handling categorical variables
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
df_total["Hospital_region_code"] =la.fit_transform(df_total["Hospital_region_code"])
df_total["Department"] = la.fit_transform(df_total["Department"])
df_total["Ward_Type"] = la.fit_transform(df_total["Ward_Type"])
df_total["Ward_Facility_Code"] = la.fit_transform(df_total["Ward_Facility_Code"])
df_total["Type of Admission"] = la.fit_transform(df_total["Type of Admission"])
df_total["Severity of Illness"] = la.fit_transform(df_total["Severity of Illness"])
df_total["Hospital_type_code"]= la.fit_transform(df_total["Hospital_type_code"])
df_total["Age"] = la.fit_transform(df_total["Age"])
#df_total = pd.get_dummies(df_total, columns=["Hospital_region_code"])
#df_total = pd.get_dummies(df_total, columns=["Department"])
#df_total = pd.get_dummies(df_total, columns=["Ward_Type"])
#df_total = pd.get_dummies(df_total, columns=["Ward_Facility_Code"])
#df_total = pd.get_dummies(df_total, columns=["City_Code_Patient"])
#df_total = pd.get_dummies(df_total, columns=["Type of Admission"])
#df_total = pd.get_dummies(df_total, columns=["Severity of Illness"])
#df_total = pd.get_dummies(df_total, columns=["Hospital_type_code"])
#df_total = pd.get_dummies(df_total, columns=["Bed Grade"])
#df_total = pd.get_dummies(df_total, columns=["City_Code_Hospital"])
#df_total = pd.get_dummies(df_total, columns=["Available Extra Rooms in Hospital"])
#df_total = pd.get_dummies(df_total, columns=["Age"])
#df_total = pd.get_dummies(df_total, columns=["Hospital_code"])
#df_total = pd.get_dummies(df_total, columns=["Visitors with Patient"])
#Unmerging train and test data 
df_train_final = df_total[df_total['is_train']==1]
df_test_final = df_total[df_total['is_train']== 0]
df_train_final["Stay"] = la.fit_transform(df_train_final["Stay"])
#Handling outliers
#Deploying the model
x=df_train_final.drop(["Stay","case_id","patientid","is_train"],axis="columns")
y=df_train_final.Stay
x_test=df_test_final.drop(["Stay","case_id","patientid","is_train"],axis="columns")
y_test=df_test_final.Stay
from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.2)
#from sklearn.svm import SVC
#sv=SVC(C=1,kernel='poly',gamma='auto',degree=3)
#sv.fit(x,y)
#import lightgbm as lgb
#import optuna
#def objective(trail):
    
#    dtrain=lgb.Dataset(x_train,label=np.ravel(y_train))
    
 #   param={
        
#        "verbosity": -1,
 #       "boosting_type": "gbdt",
  #      "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
   #     "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    #    "num_leaves": trial.suggest_int("num_leaves", 2, 256),
     #   "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
      #  "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
       # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        #"min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
#    }
 #   
  #  gbm=lgb.train(param,dtrain)
   # preds=gbm.predict(x_valid)
    #pred_labels=np.rint(preds)
    #accuracy=sklearn.metrics.accuracy_score(y_valid,pred_labels)
    #return accuracy
#opt_GS=optuna.create_study(direction="maximise")
#opt_GS.optimize(objective,n_trails=250)

#print("Number of finished trials: {}".format(len(opt_GS.trails)))

#print("Best_trail:")
#trail=opt_GS.best_trial

#print("Value: {}".format(trial.value)
      
#print("Params: ")
#for key, value in trial.params.items():
#    print("{}: {}".format(key, value))
import xgboost as xgb
import optuna 
import sklearn.metrics
def objective(trial):
    dtrain = xgb.DMatrix(x_train, label=np.ravel(y_train))
    dvalid = xgb.DMatrix(x_valid, label=np.ravel(y_valid))

    param = {
#        "silent": 1,
#        "objective": "binary:logistic",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_valid, pred_labels)
    return accuracy



study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

xgb_clf = xgb.XGBClassifier(booster="gbtree",reg_lambda=3.873345576688548e-08,reg_alpha=0.00844791893600649,max_depth=9,
                           eta=0.11,gamma=0.05291265311735101,grow_policy="lossguide",sample_type="weighted",normalize_type="forest",
                           rate_drop=0.000422577972569911,skip_drop=1.8910304509839257e-07)
xgb_clf.fit(x, np.ravel(y))
y_pred = xgb_clf.predict(x_test)
submission_df = pd.DataFrame({'case_id':df_test_final['case_id'], 'Stay':y_pred})
submission_df.to_csv('Sample Submission.csv', index=False)
