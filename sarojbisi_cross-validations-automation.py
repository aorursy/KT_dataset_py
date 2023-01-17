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
import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('/kaggle/input/covidwk1/ca_train1.csv')

test = pd.read_csv('/kaggle/input/covidwk1/ca_test1.csv')

train.tail()
test.head()
train.rename(columns={'Id': 'ForecastId'}, inplace=True)
train.head(1)
# Preprocessing of Train and test Data together

# Concatenate Train and Test Data with a new column to identify Train (Type=0) VS Test Data(Type=1)  

train['Type']=pd.DataFrame(np.zeros(len(train)).astype(int))

test['Type']=pd.DataFrame(np.ones(len(test)).astype(int))



print('Original Train Data shape:{} and Test Data shape:{}'.format(train.shape,test.shape))



features_Data=pd.concat([train.drop(columns=['ConfirmedCases','Fatalities']),test])

features_Data.drop(columns=['ForecastId','Province/State','Country/Region','Lat','Long'],inplace=True)

print('features_Data shape after dropping unnecesary columns ',features_Data.shape)



# Lets deal with Date columns

# First convert the  column to Date time data type

features_Data['Date']=pd.to_datetime(features_Data['Date'])



# We can also extract week, day, dayofweek,dayofyear and create separate columns

features_Data.insert(1,'Week',features_Data['Date'].dt.week)

features_Data.insert(2,'Day',features_Data['Date'].dt.day)

features_Data.insert(3,'DayofWeek',features_Data['Date'].dt.dayofweek)

features_Data.insert(4,'DayofYear',features_Data['Date'].dt.dayofyear)



# Check for Null Values

print('Null Value Check:\n',features_Data.isnull().sum())



train_Data=features_Data[features_Data.Type==0].drop(columns=['Date','Type'])

test_Data=features_Data[features_Data.Type==1].drop(columns=['Date','Type'])

del features_Data

print('Final Train Data shape:{} and Test Data shape:{}'.format(train_Data.shape,test_Data.shape))
train_Data.head()
# Extract features and label

features=train_Data.values

label1=train.ConfirmedCases.values

label2=train.Fatalities.values
# Lets apply Stratified K-Fold Cross Validation

from sklearn.model_selection import StratifiedKFold

#from sklearn.metrics import f1_score,classification_report,confusion_matrix

from sklearn.metrics import mean_squared_error,median_absolute_error,r2_score



def stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y):

    global df_model_selection

    

    skf = StratifiedKFold(n_splits, random_state=12,shuffle=True)

    

    weighted_r2_score = []

    

    for train_index, test_index in skf.split(X,y):

        X_train, X_test = X[train_index], X[test_index] 

        y_train, y_test = y[train_index], y[test_index]

        

        

        model_obj.fit(X_train, y_train)

        test_ds_predicted = model_obj.predict( X_test )  

           

        weighted_r2_score.append(round(r2_score(y_true=y_test, y_pred=test_ds_predicted),2))

        

    sd_weighted_r2_score = np.std(weighted_r2_score, ddof=1)

    range_of_r2_scores = "{}-{}".format(min(weighted_r2_score),max(weighted_r2_score))    

    df_model_selection = pd.concat([df_model_selection,pd.DataFrame([[process,model_name,sorted(weighted_r2_score),range_of_r2_scores,sd_weighted_r2_score]], columns =COLUMN_NAMES) ])
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor

from xgboost import XGBRFRegressor

from sklearn.neighbors import KNeighborsRegressor
COLUMN_NAMES = ["Process","Model Name", "r2 Scores","Range of r2 Scores","Std Deviation of r2 Scores"]

df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)



process='ConfirmedCases Prediction'

n_splits = 5

X=features

y=label1





# 1.LinearRegression

model_LR=LinearRegression()

model_obj=model_LR

model_name='LinearRegression'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 2.BayesianRidge

model_BR=BayesianRidge()

model_obj=model_BR

model_name='BayesianRidgeRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 3.DecisionTreeRegressor

model_DTR=DecisionTreeRegressor()

model_obj=model_DTR

model_name='DecisionTreeRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 4.RandomForestRegressor

model_RFR=RandomForestRegressor()

model_obj=model_RFR

model_name='RandomForestRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 5.GradientBoostingRegressor

model_GBR=GradientBoostingRegressor()

model_obj=model_GBR

model_name='GradientBoostingRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # # 6.XGBRegressor

model_XGBR=XGBRegressor()

model_obj=model_XGBR

model_name='XGBRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # # 7.XGBRFRegressor

model_XGBRFR=XGBRFRegressor()

model_obj=model_XGBRFR

model_name='XGBRFRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# 8.KNeighborsRegressor

model_KNNR=KNeighborsRegressor()

model_obj=model_KNNR

model_name='KNeighborsRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)





df_model_selection
## From Above Results GradientBoostingRegressor,XGBRFRegressor,RandomForestRegressor seems to be predicting better 



# Now lets try to get the Scores using StratifiedKFold Cross Validation



#Initialize the algo

model=GradientBoostingRegressor()



#Initialize StratifiedKFold Method

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, 

              random_state=1,

              shuffle=True)



#Initialize For Loop 



i=0

for train,test in kfold.split(features,label1):

    i = i+1

    X_train,X_test = features[train],features[test]

    y_train,y_test = label1[train],label1[test]

    

    model.fit(X_train,y_train)

    test_ds_predicted=model.predict(X_test)

    train_ds_predicted=model.predict(X_train)

    

    test_r2_score=round(r2_score(y_true=y_test, y_pred=test_ds_predicted ),2)

    train_r2_score=round(r2_score(y_true=y_train, y_pred=train_ds_predicted ),2)

    

    print("Train r2-Score: {}, Test r2-score: {}, for Sample Split: {}".format(train_r2_score,test_r2_score,i))
#Lets extract the Train and Test sample for split 1

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, 

              random_state=1,

              shuffle=True)

i=0

for train,test in kfold.split(features,label1):

    i = i+1

    if i == 1:

        X_train,X_test,y_train,y_test = features[train],features[test],label1[train],label1[test]





########################################################################################################

#Final Model

finalModel=GradientBoostingRegressor()

finalModel.fit(X_train,y_train)



test_ds_predicted=finalModel.predict(X_test)

train_ds_predicted=finalModel.predict(X_train)





test_r2_score=round(r2_score(y_true=y_test, y_pred=test_ds_predicted ),2)

train_r2_score=round(r2_score(y_true=y_train, y_pred=train_ds_predicted ),2)



print("Train r2-Score: {}, Test r2-score: {}".format(train_r2_score,test_r2_score))





train_score=np.round(finalModel.score(X_train,y_train),2)

test_score=np.round(finalModel.score(X_test,y_test),2)

print('Train Accuracy Score is:{} and  Test Accuracy Score:{}'.format(train_score,test_score))

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

df_submission=pd.DataFrame(finalModel.predict(test_Data),columns=['ConfirmedCases'])

df_submission.insert(0,'ForecastId',test.ForecastId)
COLUMN_NAMES = ["Process","Model Name", "r2 Scores","Range of r2 Scores","Std Deviation of r2 Scores"]

df_model_selection = pd.DataFrame(columns=COLUMN_NAMES)



process='Fatalities Prediction'

n_splits = 5

X=features

y=label2





# 1.LinearRegression

model_LR=LinearRegression()

model_obj=model_LR

model_name='LinearRegression'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 2.BayesianRidge

model_BR=BayesianRidge()

model_obj=model_BR

model_name='BayesianRidgeRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 3.DecisionTreeRegressor

model_DTR=DecisionTreeRegressor()

model_obj=model_DTR

model_name='DecisionTreeRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 4.RandomForestRegressor

model_RFR=RandomForestRegressor()

model_obj=model_RFR

model_name='RandomForestRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # 5.GradientBoostingRegressor

model_GBR=GradientBoostingRegressor()

model_obj=model_GBR

model_name='GradientBoostingRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # # 6.XGBRegressor

model_XGBR=XGBRegressor()

model_obj=model_XGBR

model_name='XGBRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# # # 7.XGBRFRegressor

model_XGBRFR=XGBRFRegressor()

model_obj=model_XGBRFR

model_name='XGBRFRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)



# 8.KNeighborsRegressor

model_KNNR=KNeighborsRegressor()

model_obj=model_KNNR

model_name='KNeighborsRegressor'

stratified_K_fold_validation(model_obj, model_name, process, n_splits, X, y)





df_model_selection
## From Above Results GradientBoostingRegressor seems to be predicting better 

# Now lets try to get the Scores using StratifiedKFold Cross Validation



#Initialize the algo

model=GradientBoostingRegressor()



#Initialize StratifiedKFold Method

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, 

              random_state=1,

              shuffle=True)



#Initialize For Loop 



i=0

for train,test in kfold.split(features,label2):

    i = i+1

    X_train,X_test = features[train],features[test]

    y_train,y_test = label2[train],label2[test]

    

    model.fit(X_train,y_train)

    test_ds_predicted=model.predict(X_test)

    train_ds_predicted=model.predict(X_train)

    

    test_r2_score=round(r2_score(y_true=y_test, y_pred=test_ds_predicted ),2)

    train_r2_score=round(r2_score(y_true=y_train, y_pred=train_ds_predicted ),2)

    

    print("Train r2-Score: {}, Test r2-score: {}, for Sample Split: {}".format(train_r2_score,test_r2_score,i))
#Lets extract the Train and Test sample for split 5

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, #n_splits should be equal to no of cv value in cross_val_score

              random_state=1,

              shuffle=True)

i=0

for train,test in kfold.split(features,label2):

    i = i+1

    if i == 5:

        X_train,X_test,y_train,y_test = features[train],features[test],label2[train],label2[test]





########################################################################################################

#Final Model

finalModel=GradientBoostingRegressor()

finalModel.fit(X_train,y_train)



test_ds_predicted=finalModel.predict(X_test)

train_ds_predicted=finalModel.predict(X_train)





test_r2_score=round(r2_score(y_true=y_test, y_pred=test_ds_predicted ),2)

train_r2_score=round(r2_score(y_true=y_train, y_pred=train_ds_predicted ),2)



print("Train r2-Score: {}, Test r2-score: {}".format(train_r2_score,test_r2_score))





train_score=np.round(finalModel.score(X_train,y_train),2)

test_score=np.round(finalModel.score(X_test,y_test),2)

print('Train Accuracy Score is:{} and  Test Accuracy Score:{}'.format(train_score,test_score))
df_submission.insert(2,'Fatalities',pd.DataFrame(finalModel.predict(test_Data),columns=['Fatalities']))

df_submission.astype('int').head()
df_submission.astype('int').to_csv('submission.csv',index=False)
