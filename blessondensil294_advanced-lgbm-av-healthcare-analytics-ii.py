# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#Basic Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data Visualization
import seaborn as sns # Advance Data Visualization
%matplotlib inline

#OS packages
import os

#Encoding Packages
import pandas as pd
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Scaling Packages
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()

#Multicolinearity VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Data Modelling Packages
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler
sm = RandomOverSampler(random_state=294,sampling_strategy='not majority')

import sklearn.metrics
from sklearn.model_selection import train_test_split

#Model Packages
import lightgbm as lgb
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df_Train = pd.read_csv('../input/av-janatahack-healthcare-hackathon-ii/Data/train.csv')
df_Test = pd.read_csv('../input/av-janatahack-healthcare-hackathon-ii/Data/test.csv')
# Checking Percentage(%) of Common Case ID's  between train and test data using Unique train values :

print(np.intersect1d(df_Train['case_id'], df_Test['case_id']).shape[0]/df_Train['case_id'].nunique())
common_ids = len(set(df_Test['case_id'].unique()).intersection(set(df_Train['case_id'].unique())))

print("Common IDs : ",common_ids)

# No - Data Leak between Train and Test !

print("Unique IDs : ",df_Test.shape[0] - common_ids)
# Checking Percentage(%) of Common ID's  between train and test data using Unique train values :

print(np.intersect1d(df_Train['patientid'], df_Test['patientid']).shape[0]/df_Train['patientid'].nunique())
common_ids = len(set(df_Test['patientid'].unique()).intersection(set(df_Train['patientid'].unique())))

print("Common IDs : ",common_ids)

# No - Data Leak between Train and Test !

print("Unique IDs : ",df_Test.shape[0] - common_ids)
#To find the head of the Data
df_Train.head()
#Information of the Dataset Datatype
df_Train.info()
#Information of the Dataset Continuous Values
df_Train.describe()
#Columns List
df_Train.columns
#Shape of the Train and Test Data
print('Shape of Train Data: ', df_Train.shape)
print('Shape of Test Data: ', df_Test.shape)
#Null values in the Train Dataset
print('Null values in Train Data: \n', df_Train.isnull().sum())
#Null Values in the Test Dataset
print('Null Values in Test Data: \n', df_Test.isnull().sum())
print('Total Count of the Prediction Output Column Stay Variable: \n', df_Train['Stay'].value_counts())
#Counting Hospital Stay
df_Train['Stay'].value_counts()
#Counting Hospital Stay
sns.countplot(x='Stay',data=df_Train)
plt.xlabel("Stay")
plt.ylabel("Count")
plt.title("Stay Duration")
plt.show()
#Counting Hospital Code
df_Train['Hospital_code'].value_counts()
#Counting Hospital Code
sns.countplot(x='Hospital_code',data=df_Train)
plt.xlabel("Hospital Code")
plt.ylabel("Count")
plt.title("Hospital Code Count")
plt.show()
#Counting Hospital Type Code
df_Train['Hospital_type_code'].value_counts()
#Counting Hospital Type Code
sns.countplot(x='Hospital_type_code',data=df_Train)
plt.xlabel("Hospital Type Code")
plt.ylabel("Count")
plt.title("Hospital Type Code Count")
plt.show()
#Counting City Code Hospital
df_Train['City_Code_Hospital'].value_counts()
#Counting Hospital Type Code
sns.countplot(x='City_Code_Hospital',data=df_Train)
plt.xlabel("City Code Hospital")
plt.ylabel("Count")
plt.title("City Code Hospital Count")
plt.show()
#Counting Hospital Region Code
df_Train['Hospital_region_code'].value_counts()
#Counting Hospital Region Code
sns.countplot(x='Hospital_region_code',data=df_Train)
plt.xlabel("Hospital Region Code")
plt.ylabel("Count")
plt.title("Hospital Region Code Count")
plt.show()
#Counting Hospital Region Code
df_Train['Available Extra Rooms in Hospital'].value_counts()
#Counting Available Extra Rooms in Hospital
sns.countplot(x='Available Extra Rooms in Hospital',data=df_Train)
plt.xlabel("Available Extra Rooms in Hospital")
plt.ylabel("Count")
plt.title("Available Extra Rooms in Hospital Count")
plt.show()
#Counting Department
df_Train['Department'].value_counts()
#Counting Department
sns.countplot(x='Department',data=df_Train)
plt.xlabel("Department")
plt.ylabel("Count")
plt.title("Department Count")
plt.show()
#Counting Ward Type
df_Train['Ward_Type'].value_counts()
#Counting Ward Type
sns.countplot(x='Ward_Type',data=df_Train)
plt.xlabel("Ward Type")
plt.ylabel("Count")
plt.title("Ward Type Count")
plt.show()
#Counting Ward Facility Code
df_Train['Ward_Facility_Code'].value_counts()
#Counting Ward Facility Code
sns.countplot(x='Ward_Facility_Code',data=df_Train)
plt.xlabel("Ward Facility Code")
plt.ylabel("Count")
plt.title("Ward Facility Code Count")
plt.show()
#Counting Bed Grade
df_Train['Bed Grade'].value_counts()
#Counting Bed Grade
sns.countplot(x='Bed Grade',data=df_Train)
plt.xlabel("Bed Grade")
plt.ylabel("Count")
plt.title("Bed Grade Count")
plt.show()
#Counting patientid
df_Train['patientid'].value_counts()
#No of Unique Data in the Patient ID Column
df_Train['patientid'].nunique()
#Unique Data in the Patient ID Column
df_Train['patientid'].unique()
#Counting patientid
#sns.countplot(x='patientid',data=df_Train)
#plt.xlabel("patientid")
#plt.ylabel("Count")
#plt.title("patientid Count")
#plt.show()
#Counting City Code Patient
df_Train['City_Code_Patient'].value_counts()
#Counting City_Code_Patient
sns.countplot(x='City_Code_Patient',data=df_Train)
plt.xlabel("City Code Patient")
plt.ylabel("Count")
plt.title("City Code Patient Count")
plt.show()
#Counting Type of Admission
df_Train['Type of Admission'].value_counts()
#Counting Type of Admission
sns.countplot(x='Type of Admission',data=df_Train)
plt.xlabel("Type of Admission")
plt.ylabel("Count")
plt.title("Type of Admission Count")
plt.show()
#Counting Severity of Illness
df_Train['Severity of Illness'].value_counts()
#Counting Severity of Illness
sns.countplot(x='Severity of Illness',data=df_Train)
plt.xlabel("Severity of Illness")
plt.ylabel("Count")
plt.title("Severity of Illness Count")
plt.show()
#Counting Visitors with Patient
df_Train['Visitors with Patient'].value_counts()
#Counting Visitors with Patient
sns.countplot(x='Visitors with Patient',data=df_Train)
plt.xlabel("Visitors with Patient")
plt.ylabel("Count")
plt.title("Visitors with Patient Count")
plt.show()
#Counting Age
df_Train['Age'].value_counts()
#Counting Age
sns.countplot(x='Age',data=df_Train)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Count")
plt.show()
#Admission Deposit Price
sns.boxplot(x=df_Train['Admission_Deposit'])
plt.xlabel("Admission Deposit")
plt.title("Admission_Deposit")
plt.show()
df_Train.drop_duplicates(keep='first', inplace=True)
# We will concat both train and test data set
df_Train['is_train'] = 1
df_Test['is_train'] = 0

#df_Frames = [df_Train,df_Test]
df_Total = pd.concat([df_Train, df_Test])
#Null values in the Total Dataset
print('Null values in Total Data: \n', df_Total.isnull().sum())
#using Forward Fill to fill missing Values
df_Total['Bed Grade']=df_Total['Bed Grade'].fillna(method="ffill",axis=0)
df_Total['City_Code_Patient']=df_Total['City_Code_Patient'].fillna(method="ffill",axis=0)
df_Total['Bill_per_patient'] = df_Total.groupby('patientid')['Admission_Deposit'].transform('sum')
df_Total['Min_Severity_of_Illness'] = df_Total.groupby('patientid')['Severity of Illness'].transform('min')
#Bill Per Patient
sns.boxplot(x=df_Total['Bill_per_patient'])
plt.xlabel("Bill_per_patient")
plt.title("Bill_per_patient")
plt.show()
df_Total.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_Total['Hospital_code'] = le.fit_transform(df_Total['Hospital_code'])
df_Total['Hospital_type_code'] = le.fit_transform(df_Total['Hospital_type_code'])
df_Total['City_Code_Hospital'] = le.fit_transform(df_Total['City_Code_Hospital'])
df_Total['Hospital_region_code'] = le.fit_transform(df_Total['Hospital_region_code'])
df_Total['Available Extra Rooms in Hospital'] = le.fit_transform(df_Total['Available Extra Rooms in Hospital'])
df_Total['Department'] = le.fit_transform(df_Total['Department'])
df_Total['Ward_Type'] = le.fit_transform(df_Total['Ward_Type'])
df_Total['Ward_Facility_Code'] = le.fit_transform(df_Total['Ward_Facility_Code'])
df_Total['Bed Grade'] = le.fit_transform(df_Total['Bed Grade'])
df_Total['patientid'] = le.fit_transform(df_Total['patientid'])
df_Total['City_Code_Patient'] = le.fit_transform(df_Total['City_Code_Patient'])
df_Total['Type of Admission'] = le.fit_transform(df_Total['Type of Admission'])
df_Total['Severity of Illness'] = le.fit_transform(df_Total['Severity of Illness'])
df_Total['Visitors with Patient'] = le.fit_transform(df_Total['Visitors with Patient'])
df_Total['Age'] = le.fit_transform(df_Total['Age'])
df_Total['Min_Severity_of_Illness'] = le.fit_transform(df_Total['Min_Severity_of_Illness'])
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
df_Total[['Admission_Deposit']] = mm_scaler.fit_transform(df_Total[['Admission_Deposit']])
df_Total[['Bill_per_patient']] = mm_scaler.fit_transform(df_Total[['Bill_per_patient']])
df_Total['Admission_Deposit'].describe()
#Un-Merge code
df_Train_final = df_Total[df_Total['is_train'] == 1]
df_Test_final = df_Total[df_Total['is_train'] == 0]
df_Train_final
df_Train_final.columns
x = df_Train_final
x = x.drop(['case_id'], axis=1)
#x = x.drop(['patientid'], axis=1)
x = x.drop(['is_train'], axis=1)
x = x.drop(['Stay'], axis=1)
y = df_Train_final['Stay']
x_pred = df_Test_final
x_pred = x_pred.drop(['case_id'], axis=1)
#x_pred = x_pred.drop(['patientid'], axis=1)
x_pred = x_pred.drop(['is_train'], axis=1)
x_pred = x_pred.drop(['Stay'], axis=1)
#y = le.fit_transform(y) #for Optuna hyperparameter tuning only
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20)
import lightgbm as lgb
import optuna
def objective(trial):
    dtrain = lgb.Dataset(x_train, label=np.ravel(y_train))

    param = {
        #"objective": "multiclass",
        #"metric": "multi_logloss",
        #"num_class": 11,
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100)
        #"n_estimators":trial.suggest_int("n_estimators", 0, 1000),
        #"learning_rate":trial.suggest_int("n_estimators", 0, 99)
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(x_valid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(y_valid, pred_labels)
    return accuracy
opt_GS = optuna.create_study(direction="maximize")
opt_GS.optimize(objective, n_trials=300)

print("Number of finished trials: {}".format(len(opt_GS.trials)))

print("Best trial:")
trial = opt_GS.best_trial

print("Value: {}".format(trial.value))

print("Params: ")
for key, value in trial.params.items():
    print("{}: {}".format(key, value))
import lightgbm as lgb
lgb_cl = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, n_estimators=50000, importance_type='gain', objective='multiclass', num_boost_round=100,
                            min_child_samples=70, num_leaves=246, #max_depth=5, 
                            lambda_l1=9.62, lambda_l2=0.006, feature_fraction=0.73, bagging_fraction=0.82, bagging_freg=6,
                            #max_bin=60, bagging_faction=0.9, feature_fraction=0.9, subsample_freq=2, scale_pos_weight=2.5, 
                            random_state=294, n_jobs=-1, silent=False) #score accuracy 42.70
#lgb_cl.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_valid, y_valid)], verbose=50, eval_metric='auc', early_stopping_rounds=100)
lgb_cl.fit(x, np.ravel(y))
y_pred = lgb_cl.predict(x_pred)
y_pred
submission_df = pd.DataFrame({'case_id':df_Test['case_id'], 'Stay':y_pred})
submission_df.to_csv('Sample Submission LGB v04.csv', index=False)
df_Total.columns()
categorical_features = ["Hospital_code", "Hospital_type_code", "City_Code_Hospital", "Hospital_region_code", "Available Extra Rooms in Hospital",
                        "Department", "Ward_Type", "Ward_Facility_Code", "Bed Grade", "patientid", "City_Code_Patient", "Type of Admission", 
                        "Visitors with Patient", "Severity of Illness", "Age", "Admission_Deposit","Bill_per_patient", "Min_Severity_of_Illness"]


param_lgb = LGBMClassifier(
    boosting_type='gbdt'
    ,learning_rate=0.1
    ,n_estimators=50000
    ,min_child_samples=21
    ,random_state = 294
    ,n_jobs=-1
    ,silent=False
    )


# Apply Stratified K-Fold Cross Validation where K=5 or n_splits=5 :
kf = StratifiedKFold(n_splits=10,shuffle=True)
preds={}
acc_score=0

# Pass predictor_train,target_train for Cross Validation :
for i,(train_idx,val_idx) in enumerate(kf.split(X)):    
    X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx, :], y.iloc[val_idx]
    print('\nFold: {}\n'.format(i+1))
    lg=LGBMClassifier(device="gpu", boosting_type='gbdt',learning_rate=0.04,depth=8,objective='multi_class',num_class=11,
                      n_estimators=50000,
                     metric='multi_error',colsample_bytree=0.5,reg_alpha=2,reg_lambda=2,random_state=294,n_jobs=-1)    
    
    # lg.fit(X_train,y_train)
    lg.fit(X_train, y_train
                        # ,categorical_feature = categorical_features
                        ,eval_metric='multi_error'
                        ,eval_set=[(X_train, y_train),(X_val, y_val)]
                        ,early_stopping_rounds=100
                        ,verbose=50
                       )
    
    print(accuracy_score(y_val,lg.predict(X_val)))
    acc_score+=accuracy_score(y_val,lg.predict(X_val))
    preds[i+1]=lg.predict(X_main_test)
    
print('mean accuracy score: {}'.format(acc_score/10))
# #Permutation Importance of Features using eli5
# perm = PermutationImportance(lg,random_state=100).fit(X_val, y_val)
# eli5.show_weights(perm,feature_names=X_val.columns.tolist())
#Finding the most frequently classified categories
d = pd.DataFrame()
for i in range(1, 10):
    d = pd.concat([d,pd.DataFrame(preds[i])],axis=1)
d.columns=['1','2','3','4','5','6','7','8','9']
re = d.mode(axis=1)[0]
submission_df['Stay']=le.inverse_transform(re.astype(int))

sub_file_name = "BEST_1_43.27_GPU-LGBM_NO-early_stopping.csv"

submission_df.to_csv(sub_file_name,index=False)
submission_df.head(5)

from google.colab import files
files.download(sub_file_name)