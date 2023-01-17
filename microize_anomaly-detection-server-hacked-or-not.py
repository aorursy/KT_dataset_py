# importing the basic libraries

import pandas as pd

import numpy as np

import datetime



# importing visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



#for balancing dataset

from imblearn.over_sampling import SMOTE



# importing sklearn libraries

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.pipeline import Pipeline

import xgboost as xgb

from sklearn.metrics import f1_score,confusion_matrix, classification_report,accuracy_score,recall_score



# display setting

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
# reading the train data

# from problem statement, wkt 'INCIDENT_ID' is the index col and have a 'DATE' column

train=pd.read_csv('../input/hacked/train.csv',index_col=['INCIDENT_ID'],parse_dates=['DATE'])

test=pd.read_csv('../input/hacked/test.csv',index_col=['INCIDENT_ID'],parse_dates=['DATE'])



display(train.head(2))
#shape of train and test data

print("Shape of Train Data: {} \nShape of Test Data: {}".format(train.shape,test.shape))
# Creating new column

train['DAYOFWEEK']=train['DATE'].dt.dayofweek

train['WEEK']=train['DATE'].dt.week



test['DAYOFWEEK']=test['DATE'].dt.dayofweek

test['WEEK']=test['DATE'].dt.week



#creating new dataset by merging train and test

merge=train.append(test)
#checking datatypes of the column

train.info()
#Target Variable counts

train.MULTIPLE_OFFENSE.value_counts(normalize=True)
#5 number summary statistics

merge.describe().T
# Exploring the Unique values

for x in ['X_1','X_2','X_3','X_4','X_5','X_6','X_7','X_8','X_9','X_10','X_11','X_12','X_13','X_14','X_15','DAYOFWEEK','WEEK']:

    print("Number of Unique values in ",x," is ",merge[x].nunique())

    print("The list of unique values is \n",np.sort(merge[x].unique()),"\n")
# Plotting frequency of Unique Values in Each Column

columns_label=['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9','X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15','DAYOFWEEK','WEEK']

for col in columns_label:

    plt.figure(figsize=(10,7))

    sns.countplot(merge[col])
# checking the NAN values in 'X_12' belongs to which target class

train[np.isnan(train['X_12'])]['MULTIPLE_OFFENSE'].value_counts()
# Number of Duplicate values after removing Date Related columns.

train_dup_size=train.drop(['DATE','DAYOFWEEK','WEEK'],axis=1).duplicated().sum()

test_dup_size=test.drop(['DATE','DAYOFWEEK','WEEK'],axis=1).duplicated().sum()



train_dup_size,test_dup_size
# removing duplicates

mod_train=train.drop(['DATE','DAYOFWEEK','WEEK'],axis=1).drop_duplicates()

mod_test=test.drop(['DATE','DAYOFWEEK','WEEK'],axis=1)

mod_merge=train.append(test)

print("The shape of the Train and test dataframe after is",mod_train.shape,mod_test.shape)



#checking value distribution of target column

print(mod_train.MULTIPLE_OFFENSE.value_counts())

mod_train.MULTIPLE_OFFENSE.value_counts(normalize=True).round(4)
df=mod_train[np.isnan(mod_train['X_12'])]

df['MULTIPLE_OFFENSE'].value_counts()
# Assigning X and y

y=mod_train['MULTIPLE_OFFENSE']

X=mod_train.drop(['MULTIPLE_OFFENSE','X_12'],axis=1)



# Splitting Dataframe into Train and Test Dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)



# Oversampling

sampler = SMOTE(sampling_strategy='minority')

X_train_smote, y_train_smote = sampler.fit_sample(X_train,y_train)



# parameter grid

xg_cl_param_grid={'n_estimators':[400,500,600,700],'max_depth':[2,3,4,5]}



# Instantiate XGBoost Classifier

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# Grid Search CV

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='accuracy',verbose=1,n_jobs=-1)



# Training the Model

grid_mse.fit(X_train_smote, y_train_smote)



# Predicting

y_predict=grid_mse.predict(X_test)

y_predict_1a=grid_mse.predict(mod_test.drop(['X_12'],axis=1))



# Different Metrics

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Best Parameter",grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print("Recall Score",res)

print("F1 Score",f1s)

print("Confusion Matrix\n",cm)

print("Classification Score\n",cr)
# submission File

test['MULTIPLE_OFFENSE']=y_predict_1a

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_1a.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)
# Parameter Dictionary for GridSearch CV

xg_cl_param_grid={'n_estimators':[200,300,400,500,600],'max_depth':[2,3,4,5]}



# Instantiate XGBoost Classifier

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# Grid Search CV

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='accuracy',verbose=1,n_jobs=-1)



# Training the Model

grid_mse.fit(X_train, y_train)



# Predicting

y_predict=grid_mse.predict(X_test)

y_predict_1b=grid_mse.predict(mod_test.drop(['X_12'],axis=1))



# Different Metrics

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print("Best Parameter",grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print("Recall score",res)

print("F1 Score",f1s)

print("Confusion Matrix\n",cm)

print("Classification Score\n",cr)
# Model created with best hyperparameter

xg_cl1=xgb.XGBClassifier(objective='reg:logistic',seed=123,n_estimators=500,max_depth=2)

xg_cl1.fit(X_train, y_train)

y_predict=xg_cl1.predict(X_test)

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print("F1 Score",f1s)

print("Confusion Matrix\n",cm)

print("Classification Score\n",cr)
# Ploting Important Feature

xgb.plot_importance(xg_cl1)
# submission File

test['MULTIPLE_OFFENSE']=y_predict_1b

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_1b.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)
# Assigning X and y

y=train['MULTIPLE_OFFENSE']

X=train.drop(['MULTIPLE_OFFENSE','X_12','DATE'],axis=1)



# Train-test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)



# Oversampling

sampler = SMOTE(sampling_strategy='minority')

X_train_smote, y_train_smote = sampler.fit_sample(X_train,y_train)



# Parameter dictionary

xg_cl_param_grid={'n_estimators':[50,100,200,300],'max_depth':[2,3,4,5]}



# Instantiate XGBoost

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# Grid Search CV

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='recall',verbose=1)



# Fit the model

grid_mse.fit(X_train_smote, y_train_smote)



# Predict

y_predict=grid_mse.predict(X_test)

y_predict_2a=grid_mse.predict(test.drop(['DATE','X_12'],axis=1))



# Metrics

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print(grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print(f1s)

print(cm)

print(cr)
# Submission file

test['MULTIPLE_OFFENSE']=y_predict_2a

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_2a.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)
# Assigning X and y

y=train['MULTIPLE_OFFENSE']

X=train.drop(['MULTIPLE_OFFENSE','X_12','DATE'],axis=1)



# Train-test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)



# Parameter dictionary

xg_cl_param_grid={'n_estimators':[50,100,200,300],'max_depth':[2,3,4,5]}



# Instantiate XGBoost

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# Grid Search CV

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='recall',verbose=1)



# Fit the model

grid_mse.fit(X_train, y_train)



# Predict

y_predict=grid_mse.predict(X_test)

y_predict_2b=grid_mse.predict(test.drop(['DATE','X_12'],axis=1))



# Metrics

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print(grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print(f1s)

print(cm)

print(cr)
# Model created with best hyperparameter

xg_cl1=xgb.XGBClassifier(objective='reg:logistic',seed=123,n_estimators=200,max_depth=2)



xg_cl1.fit(X_train, y_train)

y_predict=xg_cl1.predict(X_test)

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print("F1 Score",f1s)

print("Confusion Matrix\n",cm)

print("Classification Score\n",cr)
# Ploting Important Feature

xgb.plot_importance(xg_cl1)
# submission File

test['MULTIPLE_OFFENSE']=y_predict_2b

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_2b.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)
# Assigning X and y

y=train['MULTIPLE_OFFENSE']

X=train.drop(['MULTIPLE_OFFENSE','DATE'],axis=1)

X=X.fillna(0)

 

# Train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)



# Oversampling

sampler = SMOTE(sampling_strategy='minority')

X_train_smote, y_train_smote = sampler.fit_sample(X_train,y_train)



# Parameter dictionary

xg_cl_param_grid={'n_estimators':[50,100,200,300],'max_depth':[2,3,4,5]}



# Instantiate XGBClassifier

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# Gridsearch cv

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='recall',verbose=1)



# Train model

grid_mse.fit(X_train_smote, y_train_smote)



# Predict

y_predict=grid_mse.predict(X_test)

y_predict_3a=grid_mse.predict(test.drop(['DATE'],axis=1))



# Metrics

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print(grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print(f1s)

print(cm)

print(cr)
# Submission file

test['MULTIPLE_OFFENSE']=y_predict_3a

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_3a.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)
# Assigning X and y 

y=train['MULTIPLE_OFFENSE']

X=train.drop(['MULTIPLE_OFFENSE','DATE'],axis=1)



# Train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)



# Parameter dictionary

xg_cl_param_grid={'n_estimators':[50,100,200,300],'max_depth':[2,3,4,5]}



# Instantiate XGBClassifier

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# Grid Search CV

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='recall',verbose=1)



# Train the model

grid_mse.fit(X_train,y_train)



# Predict the model

y_predict=grid_mse.predict(X_test)

y_predict_3b=grid_mse.predict(test.drop(['DATE'],axis=1))



# Metric

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print(grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print(f1s)

print(cm)

print(cr)
# Model created with best hyperparameter

xg_cl1=xgb.XGBClassifier(objective='reg:logistic',seed=123,n_estimators=200,max_depth=2)



xg_cl1.fit(X_train, y_train)

y_predict=xg_cl1.predict(X_test)

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print("F1 Score",f1s)

print("Confusion Matrix\n",cm)

print("Classification Score\n",cr)
# Ploting Important Feature

xgb.plot_importance(xg_cl1)
# Submission file

test['MULTIPLE_OFFENSE']=y_predict_3b

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_3b.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)
# Creating New columns

merge['DAYOFWEEK']=merge['DATE'].dt.dayofweek

merge['WEEK']=merge['DATE'].dt.week



# Creating New Train and Test dataset. To predict NAN values by utilising other columns

nan_test=merge[merge['X_12'].isnull()==True]

nan_train=merge[merge['X_12'].isnull()==False]



# Train_test_split foor NAN values

b=nan_train['X_12']

A=nan_train.drop(['DATE','X_12','MULTIPLE_OFFENSE'],axis=1)

A_train,A_test,b_train,b_test=train_test_split(A,b,test_size=0.3)



# XGBClassifier

xb=xgb.XGBClassifier(n_estimator=1000,objective='reg:logistic',seed=123)



# Train the Model

xb.fit(A_train,b_train)



# Predict 

b_predict=xb.predict(A_test)

b_test_predict=xb.predict(nan_test.drop(['DATE','X_12','MULTIPLE_OFFENSE'],axis=1))

print(classification_report(b_test,b_predict))
# Predicted NAN Values

b_test_predict
# Creating New columns

train['DAYOFWEEK']=train['DATE'].dt.dayofweek

train['WEEK']=train['DATE'].dt.week

test['DAYOFWEEK']=test['DATE'].dt.dayofweek

test['WEEK']=test['DATE'].dt.week



# Filling NAN values

train['X_12']=train['X_12'].fillna(0)

test['X_12']=test['X_12'].fillna(0)



# Assigning X and y

y=train['MULTIPLE_OFFENSE']

X=train.drop(['MULTIPLE_OFFENSE','DATE'],axis=1)



# Train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)



# Oversampling

sampler = SMOTE(sampling_strategy='minority')

X_train_smote, y_train_smote = sampler.fit_sample(X_train,y_train)



# Parameter Dictionary

xg_cl_param_grid={'n_estimators':[50,100,200,300,400],'max_depth':[2,3,4,5]}



# XGBCLassifier

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# GridSearchCV

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='recall',verbose=1,n_jobs=-1)

grid_mse.fit(X_train_smote, y_train_smote)

y_predict=grid_mse.predict(X_test)

y_predict_4a=grid_mse.predict(test.drop(['DATE'],axis=1))

# metrics

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print(grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print(f1s)

print(cm)

print(cr)
# Submission file

test['MULTIPLE_OFFENSE']=y_predict_4a

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_4a.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)
# Creating New columns

train['DAYOFWEEK']=train['DATE'].dt.dayofweek

train['WEEK']=train['DATE'].dt.week

test['DAYOFWEEK']=test['DATE'].dt.dayofweek

test['WEEK']=test['DATE'].dt.week



# Filling NAN values

train['X_12']=train['X_12'].fillna(0)

test['X_12']=test['X_12'].fillna(0)



# Assign X and y

y=train['MULTIPLE_OFFENSE']

X=train.drop(['MULTIPLE_OFFENSE','DATE'],axis=1)



# Train test Split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)



# Parameter grid

xg_cl_param_grid={'n_estimators':[50,100,200,300,400],'max_depth':[2,3,4,5]}



# XGBClassifier

xg_cl=xgb.XGBClassifier(objective='reg:logistic',seed=123)



# GridSearch CV

grid_mse=GridSearchCV(xg_cl,param_grid=xg_cl_param_grid,cv=3,scoring='recall',verbose=1,n_jobs=-1)



# Train the model

grid_mse.fit(X_train,y_train)



# Predict

y_predict=grid_mse.predict(X_test)

y_predict_4b=grid_mse.predict(test.drop(['DATE'],axis=1))



# Metrics

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print(grid_mse.best_params_)

print(grid_mse.best_score_)

print('\n')

print(f1s)

print(cm)

print(cr)
# Model created with best hyperparameter

xg_cl1=xgb.XGBClassifier(objective='reg:logistic',seed=123,n_estimators=200,max_depth=2)

xg_cl1.fit(X_train, y_train)

y_predict=xg_cl1.predict(X_test)

f1s=f1_score(y_test,y_predict)

cm=confusion_matrix(y_test,y_predict)

cr=classification_report(y_test,y_predict)

res=recall_score(y_test,y_predict)



print("Recall score",res)

print("F1 Score",f1s)

print("Confusion Matrix\n",cm)

print("Classification Score\n",cr)
# Ploting Important Feature

xgb.plot_importance(xg_cl1)
# 200th tree

xgb.plot_tree(xg_cl1,num_trees=199)
# Submission file

test['MULTIPLE_OFFENSE']=y_predict_4b

sub=test['MULTIPLE_OFFENSE']

sub.to_csv('sub_4b.csv')

test=test.drop(['MULTIPLE_OFFENSE'],axis=1)