import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb

# for showing all columns
pd.set_option('display.max_columns', None)

%matplotlib inline 
test=pd.read_csv("../input/application_test.csv")
train=pd.read_csv("../input/application_train.csv")
credit_card_balance=pd.read_csv("../input/credit_card_balance.csv")
installments_payments=pd.read_csv("../input/installments_payments.csv")
bureau=pd.read_csv("../input/bureau.csv")
bureau_balance=pd.read_csv("../input/bureau_balance.csv")
pos_cash=pd.read_csv("../input/POS_CASH_balance.csv")
prev_applications=pd.read_csv("../input/previous_application.csv")

# description=pd.read_csv("data/HomeCredit_columns_description.csv",encoding='ISO-8859-1')

print("Application train:",train.shape)
print("Application test:",test.shape)
print("Credit card balance:",credit_card_balance.shape)
print("Installements Payements:",installments_payments.shape)
print("Bureau:",bureau.shape)
print("Bureau balance:",bureau_balance.shape)
print("POS_cash Balance:",pos_cash.shape)
print("Previous Applications:",prev_applications.shape)
sns.countplot(train['TARGET'])
plt.title("Distribution of target variable ")
plt.show()
train.head(10)
test.head(10)
def missing_data(data):
    na=pd.DataFrame()
    na['number']=data.isnull().sum().sort_values(ascending=False)
    na['Percent']=data.isnull().sum()/data.shape[0]*100
    na.drop(index=na.loc[na['number']==0].index,inplace=True)
    return na
print(missing_data(train).shape[0])
missing_data(train).head(10)
print(missing_data(test).shape[0])
missing_data(test).head(10)
def plot(train,test,feature,rot=False):
    plt.subplots(nrows=2,ncols=2,figsize=(12,12))
    
    plt.subplot(221)
    sns.countplot(train[feature])
    if(rot):
        plt.xticks(rotation='90')
    plt.title("For training data set")

    plt.subplot(222)
    sns.countplot(train.loc[train['TARGET']==1,feature])
    if(rot):
        plt.xticks(rotation='90')
    plt.title("Count plot when TARGET=1")

    plt.subplot(223)
    sns.barplot(x=train[feature],y=train['TARGET'])
    if(rot):
        plt.xticks(rotation='90')
    plt.title("Bar Plot between Target and "+feature)

    plt.subplot(224)
    sns.countplot(test[feature])
    if(rot):
        plt.xticks(rotation='90')
    plt.title("For test data set")
    
    plt.show()
    
    # these are numbers showing the stats
#     print(train[feature].value_counts())
#     print(train.loc[train['TARGET']==1,feature].value_counts())
    
plot(train,test,'NAME_CONTRACT_TYPE')
plot(train,test,'CODE_GENDER')
plot(train,test,'FLAG_OWN_CAR')
plot(train,test,'FLAG_OWN_REALTY')
plot(train,test,'CNT_CHILDREN')
plot(train,test,'NAME_TYPE_SUITE',rot=True)
plot(train,test,'NAME_INCOME_TYPE',rot=True)
plot(train,test,'NAME_EDUCATION_TYPE',rot=True)
plot(train,test,'NAME_FAMILY_STATUS',rot=True)
plot(train,test,'NAME_HOUSING_TYPE',rot=True)
plot(train,test,'OCCUPATION_TYPE',rot=True)
plot(train,test,'CNT_FAM_MEMBERS',rot=True)
plot(train,test,'ORGANIZATION_TYPE',rot=True)
def plotc(train,feature):
    plt.subplots(nrows=1,ncols=2,figsize=(12,7))
    
    plt.subplot(121)
    sns.distplot(train[feature])
    
    plt.subplot(122)
    sns.boxplot(x=train['TARGET'],y=train[feature])
    plt.show()
plotc(train,'AMT_INCOME_TOTAL')
plotc(train,'AMT_CREDIT')
plotc(train,'DAYS_BIRTH')
plotc(train,'DAYS_EMPLOYED')
plotc(train,'DAYS_REGISTRATION')