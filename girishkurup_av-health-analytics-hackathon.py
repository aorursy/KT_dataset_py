# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/avhealthcare/Train.csv')
patient=pd.read_csv('/kaggle/input/avhealthcare/Patient_Profile.csv')
camp=pd.read_csv('/kaggle/input/avhealthcare/Health_Camp_Detail.csv')
FHC=pd.read_csv('/kaggle/input/avhealthcare/First_Health_Camp_Attended.csv')
SHC=pd.read_csv('/kaggle/input/avhealthcare/Second_Health_Camp_Attended.csv')
THC=pd.read_csv('/kaggle/input/avhealthcare/Third_Health_Camp_Attended.csv')
FHC=FHC.iloc[:, :-1]
SHC.info()
train1=pd.merge(train,patient, on='Patient_ID', how='left')
train12=pd.merge(train1,camp, on='Health_Camp_ID',how='left')
train123=pd.merge(train12,FHC, on=['Patient_ID','Health_Camp_ID'], how='left')
train1234=pd.merge(train123,SHC, on=['Patient_ID','Health_Camp_ID'], how='left')
train12345=pd.merge(train1234,THC, on=['Patient_ID','Health_Camp_ID'], how='left')
train12345
train12345['Outcome'] = train12345.apply(lambda x: 1 if (x['Health_Score'] > 0 
                                                                 or x['Health Score'] > 0 
                                                                 or x['Number_of_stall_visited'] > 0) 
                                                     else 0,axis=1)
train12345['Outcome'].value_counts()
train12345.isnull().sum()
train12345.Income.value_counts()
train12345['Income'] = train12345['Income'].replace('None', np.nan)
train12345['Income'].fillna(train12345['Income'].mode()[0], inplace=True)
train12345['Education_Score'] = train12345['Education_Score'].replace('None', np.nan)
train12345['Education_Score'].fillna(train12345['Education_Score'].mode()[0], inplace=True)
train12345['Age'] = train12345['Age'].replace('None', np.nan)
train12345['Age'].fillna(train12345['Age'].mode()[0], inplace=True)
train12345['Registration_Date'].fillna(train12345['Registration_Date'].mode()[0], inplace=True)
train12345['Income'] = train12345['Income'].astype("int16")
train12345['Education_Score'] = train12345['Education_Score'].astype("float64")
train12345['Age'] = train12345['Age'].astype("int16")

train12345['City_Type'] = train12345['City_Type'].fillna('Unknown')
train12345['Employer_Category'] = train12345['Employer_Category'].fillna('Unknown')
catcols = ['Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared', 'Category1', 'Category2', 'Category3', 'City_Type', 'Employer_Category']
for col in catcols:
    train12345[col] = train12345[col].astype('category')
    train12345[col] = train12345[col].cat.codes.astype("int16")
train12345=train12345.drop(['Patient_ID', 'Health_Camp_ID', 'Donation', 'Health_Score', 'Health Score', 'Number_of_stall_visited', 'Last_Stall_Visited_Number'],axis=1)
train12345['Registration_Date']=pd.to_datetime(train12345['Registration_Date'],dayfirst=True)
train12345['First_Interaction']=pd.to_datetime(train12345['First_Interaction'],dayfirst=True)
train12345['Camp_Start_Date']=pd.to_datetime(train12345['Camp_Start_Date'],dayfirst=True)
train12345['Camp_End_Date']=pd.to_datetime(train12345['Camp_End_Date'],dayfirst=True)
train12345['Campduration']=(train12345['Camp_End_Date']-train12345['Camp_Start_Date']).dt.days
train12345['registrationgapwithcampstart']=(train12345['Registration_Date']-train12345['Camp_Start_Date']).dt.days
train12345['registrationgapwithcampend']=(train12345['Camp_End_Date']-train12345['Registration_Date']).dt.days 
train12345['interactiongapwithcampstart']=(train12345['First_Interaction']-train12345['Camp_Start_Date']).dt.days
train12345['interactiongapwithcampend']=(train12345['Camp_End_Date']-train12345['First_Interaction']).dt.days
train12345['interactiongapwithregistration']=(train12345['Registration_Date']-train12345['First_Interaction']).dt.days
train12345.columns
train12345['Regquarter'] = train12345['Registration_Date'].dt.quarter
train12345['Regyear'] = train12345['Registration_Date'].dt.year
train12345['Regmonth'] = train12345['Registration_Date'].dt.month
train12345['Regdate'] = train12345['Registration_Date'].dt.day
train12345['Regweek_day'] = train12345['Registration_Date'].dt.dayofweek
train12345['Intgquarter'] = train12345['First_Interaction'].dt.quarter
train12345['Intyear'] = train12345['First_Interaction'].dt.year
train12345['Intgmonth'] = train12345['First_Interaction'].dt.month
train12345['Intdate'] = train12345['First_Interaction'].dt.day
train12345['Intweek_day'] = train12345['First_Interaction'].dt.dayofweek
train12345['Campstquarter'] = train12345['Camp_Start_Date'].dt.quarter
train12345['Campstyear'] = train12345['Camp_Start_Date'].dt.year
train12345['Campstgmonth'] = train12345['Camp_Start_Date'].dt.month
train12345['Campstdate'] = train12345['Camp_Start_Date'].dt.day
train12345['Campstweek_day'] = train12345['Camp_Start_Date'].dt.dayofweek
train12345['Campendquarter'] = train12345['Camp_End_Date'].dt.quarter
train12345['Campendyear'] = train12345['Camp_End_Date'].dt.year
train12345['Campendgmonth'] = train12345['Camp_End_Date'].dt.month
train12345['Campenddate'] = train12345['Camp_End_Date'].dt.day
train12345['Campendweek_day'] = train12345['Camp_End_Date'].dt.dayofweek
train12345=train12345.drop(['Registration_Date','First_Interaction','Camp_Start_Date','Camp_End_Date'], axis=1)
train12345.isnull().sum()
train12345.isnull().info()
X= train12345.drop(['Outcome'],axis=1)
y= train12345['Outcome']
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 22)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
import gc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, mean_absolute_error,accuracy_score, classification_report
kfold = KFold(n_splits=10, random_state=7)
#rf = RandomForestRegressor(n_estimators = 200)
#rf.fit(X_train, y_train,eval_metric = 'auc', early_stopping_rounds = 100)
#RF=rf.fit(X_train, y_train)

num_trees = 200
max_features = 3
modelRF = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
resultsRF = cross_val_score(modelRF,X_train, y_train,cv=kfold)
print("Random Forest",resultsRF.mean()*100)
RF=modelRF.fit(X_train, y_train)
y_pred_randomforest= RF.predict(X_train)
print(100*np.sqrt(mean_squared_log_error(y_train, y_pred_randomforest)))

y_pred_randomforest= RF.predict(X_test)
print(100*np.sqrt(mean_squared_log_error(y_test, y_pred_randomforest)))

#print('RMSLE score is', 100*(np.sqrt(np.mean(np.power(np.log1p(y_test)-np.log1p(test_pred), 2)))))
resultsRF_test = cross_val_score(modelRF,X_test, y_test,cv=kfold)
print("Random Forest",resultsRF_test.mean()*100)
sorted(zip(RF.feature_importances_, X_train), reverse = True)
predict_Prob_RF=RF.predict_proba(X_train)[:,1]
predict_Prob_RF
from lightgbm import LGBMRegressor

from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
#rmsle = 0
#for i in ratio:
#  x_train,y_train,x_val,y_val = train_test_split(i)
#lgbc=LGBMRegressor(boosting_type='gbdt',n_estimators=800, learning_rate=0.12,objective= 'regression',n_jobs=-1,random_state=100)
#LGB=lgbc.fit(X_train,y_train)

lgbc = LGBMClassifier(n_estimators=550,
                     learning_rate=0.03,
                     min_child_samples=40,
                     random_state=1,
                     colsample_bytree=0.5,
                     reg_alpha=2,
                     reg_lambda=2)

resultsLGB = cross_val_score(lgbc,X_train, y_train,cv=kfold)
print("LightGBM",resultsLGB.mean()*100)

LGB=lgbc.fit(X_train,y_train)
y_predict_LGBM = LGB.predict(X_train)
print(100*(np.sqrt(mean_squared_log_error(np.exp(y_train), np.exp(y_predict_LGBM)))))
y_predict_LGBM = LGB.predict(X_test)
print(100*(np.sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_predict_LGBM)))))
resultsLGB_test = cross_val_score(lgbc,X_test, y_test,cv=kfold)
print("LightGBM",resultsLGB_test.mean()*100)
sorted(zip(LGB.feature_importances_, X_train), reverse = True)
predict_Prob_LGBM=LGB.predict_proba(X_train)[:,1]
predict_Prob_LGBM
from catboost import CatBoostRegressor 
from catboost import  CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_auc_score

#cb = CatBoostRegressor(
    #n_estimators = 1000,
    #learning_rate = 0.11,
    #iterations=1000,
    #loss_function = 'RMSE',
    #eval_metric = 'RMSE',
    #verbose=0)
    
cb= CatBoostClassifier(
    iterations=100, 
    learning_rate=0.1, 
    #loss_function='CrossEntropy'
)

#rmsle = 0
#for i in ratio:
 # x_train,y_train,x_val,y_val = train_test_split(i)

#CAT=cb.fit(X_train,y_train)
#resultsCAT = cross_val_score(cb,X_train, y_train,cv=kfold)
#print("CAT",resultsCAT.mean()*100)
                        
cb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50,early_stopping_rounds = 100)
resultsCAT = cross_val_score(cb,X_train, y_train,cv=kfold)
print("CAT",resultsCAT.mean()*100)
y_predict_CAT = cb.predict(X_train)
print(100*(np.sqrt(mean_squared_log_error(np.exp(y_train), np.exp(y_predict_CAT)))))
resultsCAT_train = cross_val_score(cb,X_train, y_train,cv=kfold)
print("CAT",resultsCAT_train.mean()*100)
y_predict_CAT = cb.predict(X_test)
print(100*(np.sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_predict_CAT)))))
resultsCAT_test = cross_val_score(cb,X_test, y_test,cv=kfold)
print("CAT",resultsCAT_test.mean()*100)
sorted(zip(cb.feature_importances_, X_train), reverse = True)
predict_Prob_CAT=cb.predict_proba(X_train)[:,1]
predict_Prob_CAT
test=pd.read_csv('/kaggle/input/avhealthcare1/test_l0Auv8Q.csv') 
test.columns
test.info()
test1=pd.merge(test,patient, on='Patient_ID', how='left')
test12=pd.merge(test1,camp, on='Health_Camp_ID',how='left')
test123=pd.merge(test12,FHC, on=['Patient_ID','Health_Camp_ID'], how='left')
test1234=pd.merge(test123,SHC, on=['Patient_ID','Health_Camp_ID'], how='left')
test12345=pd.merge(test1234,THC, on=['Patient_ID','Health_Camp_ID'], how='left')
test12345.isnull().sum()
test12345['Income']=test12345['Income'].replace('None',np.nan)
test12345['Income'].fillna(test12345['Income'].mode()[0],inplace=True)
test12345['Education_Score']=test12345['Education_Score'].replace('None',np.nan)
test12345['Education_Score'].fillna(test12345['Education_Score'].mode()[0],inplace=True)
test12345['Age']=test12345['Age'].replace('None',np.nan)
test12345['Age'].fillna(test12345['Age'].mode()[0],inplace=True)

test12345['Income'] = test12345['Income'].astype("int16")
test12345['Education_Score'] = test12345['Education_Score'].astype("float64")
test12345['Age'] = test12345['Age'].astype("int16")

test12345.info()
test12345['City_Type'].fillna(test12345['City_Type'].mode()[0],inplace=True)
test12345
ForSubmission=test12345
test12345=test12345.drop(['Patient_ID', 'Health_Camp_ID', 'Donation', 'Health_Score', 'Health Score', 'Number_of_stall_visited', 'Last_Stall_Visited_Number'],axis=1)
test12345
test12345['City_Type'] = test12345['City_Type'].fillna('Unknown')
test12345['Employer_Category'] = test12345['Employer_Category'].fillna('Unknown')
test12345
catcols = ['Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared', 'Category1', 'Category2', 'Category3', 'City_Type', 'Employer_Category']
for col in catcols:
    test12345[col] = test12345[col].astype('category')
    test12345[col] = test12345[col].cat.codes.astype("int16")
test12345
test12345['Registration_Date']=pd.to_datetime(test12345['Registration_Date'],dayfirst=True)
test12345['First_Interaction']=pd.to_datetime(test12345['First_Interaction'],dayfirst=True)
test12345['Camp_Start_Date']=pd.to_datetime(test12345['Camp_Start_Date'],dayfirst=True)
test12345['Camp_End_Date']=pd.to_datetime(test12345['Camp_End_Date'],dayfirst=True)
test12345['Registration_Date'].fillna(test12345['Registration_Date'].mode()[0], inplace=True)
test12345['Campduration']=(test12345['Camp_End_Date']-test12345['Camp_Start_Date']).dt.days
test12345['registrationgapwithcampstart']=(test12345['Registration_Date']-test12345['Camp_Start_Date']).dt.days
test12345['registrationgapwithcampend']=(test12345['Camp_End_Date']-test12345['Registration_Date']).dt.days 

test12345['interactiongapwithcampstart']=(test12345['First_Interaction']-test12345['Camp_Start_Date']).dt.days
test12345['interactiongapwithcampend']=(test12345['Camp_End_Date']-test12345['First_Interaction']).dt.days


test12345['interactiongapwithregistration']=(test12345['Registration_Date']-test12345['First_Interaction']).dt.days

test12345['Regquarter'] = test12345['Registration_Date'].dt.quarter
test12345['Regyear'] = test12345['Registration_Date'].dt.year
test12345['Regmonth'] = test12345['Registration_Date'].dt.month
test12345['Regdate'] = test12345['Registration_Date'].dt.day
test12345['Regweek_day'] = test12345['Registration_Date'].dt.dayofweek

test12345['Intgquarter'] = test12345['First_Interaction'].dt.quarter
test12345['Intyear'] = test12345['First_Interaction'].dt.year
test12345['Intgmonth'] = test12345['First_Interaction'].dt.month
test12345['Intdate'] = test12345['First_Interaction'].dt.day
test12345['Intweek_day'] = test12345['First_Interaction'].dt.dayofweek

test12345['Campstquarter'] = test12345['Camp_Start_Date'].dt.quarter
test12345['Campstyear'] = test12345['Camp_Start_Date'].dt.year
test12345['Campstgmonth'] = test12345['Camp_Start_Date'].dt.month
test12345['Campstdate'] = test12345['Camp_Start_Date'].dt.day
test12345['Campstweek_day'] = test12345['Camp_Start_Date'].dt.dayofweek

test12345['Campendquarter'] = test12345['Camp_End_Date'].dt.quarter
test12345['Campendyear'] = test12345['Camp_End_Date'].dt.year
test12345['Campendgmonth'] = test12345['Camp_End_Date'].dt.month
test12345['Campenddate'] = test12345['Camp_End_Date'].dt.day
test12345['Campendweek_day'] = test12345['Camp_End_Date'].dt.dayofweek

test12345.info()
test12345=test12345.drop(['Registration_Date','First_Interaction','Camp_Start_Date','Camp_End_Date'], axis=1)
predict_Prob_LGBM=LGB.predict_proba(test12345)[:,1]
predict_Prob_LGBM
df_solution = pd.DataFrame()
df_solution['Patient_ID'] = ForSubmission.Patient_ID
df_solution['Health_Camp_ID'] =ForSubmission.Health_Camp_ID
df_solution['Outcome'] = predict_Prob_LGBM
df_solution
df_solution.Outcome.value_counts()
df_solution.to_csv("LGB_Implementation_Health_Analytics_Submission.csv", index=False)