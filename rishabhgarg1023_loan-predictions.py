# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
start=time.time()
train_df=pd.read_csv("../input/train_AV3.csv")
test_df=pd.read_csv("../input/test_AV3.csv")
end=time.time()
print("Time taken by above cell is {}.".format((end-start)))
start=time.time()
print("First five rows of train_df:-")
print(train_df.head())
print("First five rows of test_df:-")
print(test_df.head())
print("Shape of train_df:-")
print(train_df.shape)
print("Shape of test_df:-")
print(test_df.shape)
print("Columns of train_df:-")
print(train_df.columns)
end=time.time()
print("Time taken by above cell is {}.".format((end-start)))
print("Info of train_df:-")
train_df.info()
print("NO. of values in different varibles for train_df")
print(train_df.count())
print("NO. of values in different varibles for test_df")
print(test_df.count())

print("no. of null values in train_df:-")
print(train_df.isnull().sum())
print("no. of null values in test_df:-")
print(test_df.isnull().sum())

def regender(srs):
    if(srs.Gender=="Male"):
        srs.Gender=1
    elif(srs.Gender=="Female"):
        srs.Gender=2
    return srs

train_df=train_df.apply(regender, axis='columns')
test_df=test_df.apply(regender,axis='columns')
print(train_df.head())
print(test_df.head())

# taking married as 1 and unmarried as 2
def remarried(srs):
    if(srs.Married=="Yes"):
        srs.Married=1
    elif(srs.Married=="No"):
        srs.Married=2
    return srs

train_df=train_df.apply(remarried, axis='columns')
test_df=test_df.apply(remarried, axis='columns')
print(train_df.head())
print(test_df.head())
def reducate(srs):
    if(srs.Education=="Graduate"):
        srs.Education=1
    elif(srs.Education=="Not Graduate"):
        srs.Education=2
    return srs

train_df=train_df.apply(reducate, axis='columns')
test_df=test_df.apply(reducate, axis='columns')
train_df
def remployed(srs):
    if(srs.Self_Employed=="Yes"):
        srs.Self_Employed=1
    elif(srs.Self_Employed=="No"):
        srs.Self_Employed=2
    return srs

train_df=train_df.apply(remployed, axis='columns')
test_df=test_df.apply(remployed, axis='columns')
train_df
def reproperty(srs):
    if(srs.Property_Area=="Urban"):
        srs.Property_Area=1
    elif(srs.Property_Area=="Semiurban"):
        srs.Property_Area=2
    elif(srs.Property_Area=="Rural"):
        srs.Property_Area=3
    return srs

train_df=train_df.apply(reproperty, axis='columns')
test_df=test_df.apply(reproperty, axis='columns')
train_df
def reloan(srs):
    if(srs.Loan_Status=="Y"):
        srs.Loan_Status=1
    elif(srs.Loan_Status=="N"):
        srs.Loan_Status=2
  
    return srs

train_df=train_df.apply(reloan, axis='columns')
train_df
def redependent(srs):
    if(srs.Dependents=="0"):
        srs.Dependents=0
    elif(srs.Dependents=="1"):
        srs.Dependents=1
    elif(srs.Dependents=="2"):
        srs.Dependents=2
    elif(srs.Dependents=="3+"):
        srs.Dependents=3
  
    return srs

train_df=train_df.apply(redependent, axis='columns')
test_df=test_df.apply(redependent, axis='columns')
train_df
train_df.head()
'''
train_df["Gender"]=train_df.Gender.fillna(1)
test_df["Gender"]=test_df.Gender.fillna(1)

train_df["Dependents"]=train_df.Dependents.fillna(0)
test_df["Dependents"]=test_df.Dependents.fillna(0)

train_df["Self_Employed"]=train_df.Self_Employed.fillna(2)
test_df["Self_Employed"]=test_df.Self_Employed.fillna(2)

train_df["LoanAmount"]=train_df.LoanAmount.fillna(146.369)
test_df["LoanAmount"]=test_df.LoanAmount.fillna(136.13)

train_df["Loan_Amount_Term"]=train_df.Loan_Amount_Term.fillna(360)
test_df["Loan_Amount_Term"]=test_df.Loan_Amount_Term.fillna(360)

train_df["Credit_History"]=train_df.Credit_History.fillna(1)
test_df["Credit_History"]=test_df.Credit_History.fillna(1)


'''


train_df = train_df.dropna(subset=['Married'])
train_df.isnull().sum()
original_data_train=train_df.copy()
original_data_test=test_df.copy()
original_data_train.set_index("Loan_ID")
original_data_test.set_index("Loan_ID")
Loan_ID_train=original_data_train.Loan_ID
Loan_ID_test=original_data_test.Loan_ID
original_data_train=original_data_train.drop("Loan_ID",axis=1)
original_data_test=original_data_test.drop("Loan_ID",axis=1)
original_data_train.head()
# make new columns indicating what will be imputed
from sklearn.preprocessing import Imputer
new_data_train = original_data_train.copy()


# Imputation
my_imputer = Imputer()
new_data_train = my_imputer.fit_transform(new_data_train)
from sklearn.preprocessing import Imputer
new_data_test = original_data_test.copy()


# Imputation
my_imputer = Imputer()
new_data_test = my_imputer.fit_transform(new_data_test)
new_data_train
new_df_train=pd.DataFrame(new_data_train)
new_df_test=pd.DataFrame(new_data_test)
new_df_train
new_df_train.columns
new_df_train=new_df_train.rename( columns={0: "Gender", 1: "Married",2:"Dependents",3:"Education",4:"Self_Employed",5:"ApplicantIncome",6:"CoapplicantIncome",7:"LoanAmount",8:"Loan_Amount_Term",9:"Credit_History",10:"Property_Area",11:"Loan_Status"})
new_df_test=new_df_test.rename( columns={0: "Gender", 1: "Married",2:"Dependents",3:"Education",4:"Self_Employed",5:"ApplicantIncome",6:"CoapplicantIncome",7:"LoanAmount",8:"Loan_Amount_Term",9:"Credit_History",10:"Property_Area"})
new_df_test
new_df_train
new_df_train.isnull().sum()
new_df_train.Married.value_counts()
new_df_train['Gender'].value_counts().plot.bar()
new_df_train['Married'].value_counts().plot.bar()
new_df_train['Dependents'].value_counts().plot.bar()
new_df_train['Education'].value_counts().plot.bar()
new_df_train['Self_Employed'].value_counts().plot.bar()

#Credit_History       0
#Property_Area        0
#Loan_Status 
new_df_train['ApplicantIncome'].plot.line()
new_df_train["CoapplicantIncome"].plot.line()
new_df_train["LoanAmount"].plot.line()
new_df_train["Loan_Amount_Term"].value_counts().plot.bar()
new_df_train["Credit_History"].value_counts().plot.bar()
new_df_train["Property_Area"].value_counts().plot.bar()
new_df_train["Loan_Status"].value_counts().plot.bar()
y = new_df_train.Loan_Status
Loan_Predictors = ['Gender', 'Married', 'Education', 'Self_Employed', 
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Dependents'
                  ]
X = new_df_train[Loan_Predictors]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
#XGBoost
from xgboost import XGBRegressor


my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
#my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#my_model.fit(X, y,  
 #             verbose=False)
val_X=new_df_test[Loan_Predictors]
loan_preds = my_model.predict(val_X)
loan_preds=pd.DataFrame(loan_preds)
loan_preds
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
val_X=new_df_test[Loan_Predictors]
forest_model = RandomForestRegressor()
forest_model.fit(X, y)
loan_preds = forest_model.predict(val_X)
loan_preds=pd.DataFrame(loan_preds)
loan_preds
'''
loan_preds=loan_preds.rename( columns={0:"Loan_Status"})
loan_preds
def loan_pred(srs):
    if(srs.Loan_Status<=1.5):
        srs.Loan_Status='Y'
    elif(srs.Loan_Status>1.5):
        srs.Loan_Status='N'
    
  
    return srs

loan_preds=loan_preds.apply(loan_pred, axis='columns')
loan_preds
submission_file=pd.concat([Loan_ID_test,loan_preds], axis=1)
submission_file

submission_file.to_csv('submission.csv', index=False)
