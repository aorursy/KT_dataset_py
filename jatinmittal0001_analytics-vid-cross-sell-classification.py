# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from catboost import CatBoostClassifier

import catboost

from catboost import *

import numpy as np

from sklearn.metrics import roc_auc_score

import seaborn as sns

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

'''

https://datahack.analyticsvidhya.com/contest/janatahack-cross-sell-prediction/#ProblemStatement

'''



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/janatahack-crosssell-prediction/train.csv")

test = pd.read_csv("../input/janatahack-crosssell-prediction/test.csv")

print('Shape of raw train data: ',train.shape)

print('Shape of raw test data: ',test.shape)

print(train.columns)

train.head(5)
'''

Variable	     Definition

id	             Unique ID for the customer

Gender	         Gender of the customer

Age              Age of the customer

Driving_License	0 : Customer does not have DL, 1 : Customer already has DL

Region_Code	      Unique code for the region of the customer

Previously_Insured	1 : Customer already has Vehicle Insurance, 0 : Customer doesn't have Vehicle Insurance

Vehicle_Age	     Age of the Vehicle 

Vehicle_Damage   1 : Customer got his/her vehicle damaged in the past. 0 : Customer didn't get his/her vehicle damaged in the past.

Annual_Premium	 The amount customer needs to pay as premium in the year

Policy_Sales_Channel	Anonymised Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc.

Vintage	         Number of Days, Customer has been associated with the company

Response	1 :  Customer is interested, 0 : Customer is not interested



'''
print("#Rows: ",(train[train.Previously_Insured==1].shape[0]))

print("Sum: ",(train[train.Previously_Insured==1]["Response"].values).sum())
print("#Rows: ",(train.shape[0]))

print("Sum: ",(train["Response"].values).sum())
train_sub = train

test_sub = test
sns.boxplot(x = train_sub.Age)  
sns.boxplot(x = train_sub.Annual_Premium)
sns.boxplot(x = train_sub.Vintage)
def outlier_treatment(data,p1,p99):

    data_X = data.copy()

    col = "Annual_Premium"

    data_X[col][data_X[col] <= p1] = p1

    data_X[col][data_X[col] >= p99] = p99

    

    return data_X



a = train["Annual_Premium"].quantile([0.25,0.75]).values

p_cap = a[1] + 1.5*(a[1]-a[0])

p_clip = a[0] - 1.5*(a[1]-a[0])

train_sub = outlier_treatment(train_sub,p_clip,p_cap)

test_sub = outlier_treatment(test_sub,p_clip,p_cap)
train_sub.columns
sns.countplot(x="Response", data=train[train.Previously_Insured==0])
sns.catplot(x="Response", y='Annual_Premium', data=train[train.Previously_Insured==0]) 
sns.distplot(train_sub['Age'],kde = False)
#segregating the ID and response for later use

target = train_sub[["id","Response"]]

target.head(2)
#appending train and test set for converting few vars to categorical

#train_sub.drop(["Response"],axis=1,inplace=True)

test_sub['Response']=999

total = train_sub.append(test_sub,ignore_index = True)



total["Gender"] = (total["Gender"].astype('string')).astype('category')

total["Driving_License"] = (total["Driving_License"].astype('string')).astype('category')

total["Previously_Insured"] = total["Previously_Insured"].astype('category')

total["Vehicle_Age"] = (total["Vehicle_Age"].astype('string')).astype('category')

total["Vehicle_Damage"] = (total["Vehicle_Damage"].astype('string')).astype('category')

#total["Policy_Sales_Channel"] = (total["Policy_Sales_Channel"].astype('string')).astype('category')

#total["Region_Code"] = (total["Region_Code"].astype('string')).astype('category')



#cols_to_one_hot_dummy = ["Gender","Vehicle_Damage","Vehicle_Age","Previously_Insured"]

#dataframe = pd.get_dummies(total,columns=cols_to_one_hot_dummy)

#dataframe.rename(columns={"Vehicle_Age_1-2 Year": "V_age_1to2", "Vehicle_Age_< 1 Year": "V_age_L1","Vehicle_Age_> 2 Years":"V_age_G2"},inplace=True)

#dataframe.drop(["Gender","Driving_License","Vehicle_Damage","Vehicle_Age","Previously_Insured"],axis=1,inplace=True)

dataframe=total







dataframe['Gender']=dataframe['Gender'].replace({'Male':1,'Female':0})

dataframe['Vehicle_Damage']=dataframe['Vehicle_Damage'].replace({'Yes':1,'No':0})

dataframe['Vehicle_Age']=dataframe['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})



# changing data type because cat_feature in catboost cannot be float

dataframe['Region_Code']=dataframe['Region_Code'].astype(int)

dataframe['Previously_Insured']=dataframe['Previously_Insured'].astype(int)

dataframe['Policy_Sales_Channel']=dataframe['Policy_Sales_Channel'].astype(int)

dataframe['Driving_License']=dataframe['Driving_License'].astype(int)



'''

bin_values = [0,25, 30, 35,40, 50,60,70,100]    #NOTE: no bin will be made for 10000000 to infinity, last bin: 65000 to 10000000

bin_labels = ["1", "2", "3",'4','5','6','7','8']       # text labels for each bin



dataframe['Age_cut'] = pd.cut(x=dataframe['Age'],bins= bin_values,labels=bin_labels)

dataframe['Age_cut'] = (dataframe['Age_cut'].replace({'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8})).astype(int)

'''

dataframe.drop(["Driving_License"],axis=1,inplace=True)



'''

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

obj = (dataframe.dtypes=='object') + (dataframe.dtypes=='category')

col = (dataframe.dtypes[obj].index).tolist()

for i in range(0,len(col)):

    l = col[i]

    dataframe[l] = le.fit_transform(dataframe[l])

'''



dataframe.head()
train_sub.head(2)
a1 = train_sub[((train_sub.Driving_License==1) & (train_sub.Gender=='Female') & (train_sub.Response==1))].shape[0]/(train_sub[((train_sub.Driving_License==1) & (train_sub.Gender=='Female'))].shape[0])

a2 = train_sub[((train_sub.Driving_License==0) & (train_sub.Gender=='Female') & (train_sub.Response==1))].shape[0]/(train_sub[((train_sub.Driving_License==0) & (train_sub.Gender=='Female'))].shape[0])



a3 = train_sub[((train_sub.Driving_License==1) & (train_sub.Gender=='Male') & (train_sub.Response==1))].shape[0]/(train_sub[((train_sub.Driving_License==1) & (train_sub.Gender=='Male'))].shape[0])

a4 = train_sub[((train_sub.Driving_License==0) & (train_sub.Gender=='Male') & (train_sub.Response==1))].shape[0]/(train_sub[((train_sub.Driving_License==0) & (train_sub.Gender=='Male'))].shape[0])
dataframe.drop(['Response'],axis=1,inplace=True)

#"Driving_License",'Gender',

dataframe.head()
t_size = train_sub.shape[0]

train_sub_test_sub_id = dataframe["id"]

dataframe.drop(["id"],axis=1,inplace=True)

train_data = dataframe.iloc[:t_size,:]



test_data = dataframe.iloc[t_size:,:]



print(train_data.shape[0])

print(train_sub.shape[0])


from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(train_data, target['Response'], test_size=.25, random_state=150,stratify=target['Response'],shuffle=True)

X_train.head()
X_validation.dtypes




# categorical column 

cat_col=['Gender','Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']

from catboost import CatBoostClassifier

catb = CatBoostClassifier(

    iterations=200,

    cat_features=cat_col,

    random_seed=4,

    learning_rate=0.2,

    early_stopping_rounds=30,

    #one_hot_max_size =2

)

catb.fit(

    X_train, y_train,

    #cat_features=[1,3],

    eval_set=(X_validation, y_validation),

    logging_level='Silent',

    plot=True

)

print('Model is fitted: ' + str(catb.is_fitted()))

print('Model params:')

print(catb.get_params())
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(X_train, y_train)
p_train_catb = catb.predict_proba(X_train)

p_train_xgb = xgb.predict_proba(X_train)



p_test_catb = catb.predict_proba(X_validation)

p_test_xgb = xgb.predict_proba(X_validation)
print('Catboost train set: ',roc_auc_score(y_train, p_train_catb[:,1]))

print('Catboost test set: ',roc_auc_score(y_validation, p_test_catb[:,1]))





print('XGB train set: ',roc_auc_score(y_train, p_train_xgb[:,1]))

print('XGB test set: ',roc_auc_score(y_validation, p_test_xgb[:,1]))
ensemble_pred_train = 0.35*p_train_xgb[:,1] + 0.65*p_train_catb[:,1]

ensemble_pred_test = 0.35*p_test_xgb[:,1] + 0.65* p_test_catb[:,1]
print('Ensemble train set: ',roc_auc_score(y_train, ensemble_pred_train))

print('Ensemble test set: ',roc_auc_score(y_validation, ensemble_pred_test))
p_test_catb = catb.predict_proba(test_data)

p_test_xgb = xgb.predict_proba(test_data)



ensemble_pred_final_test = 0.35*p_test_xgb[:,1] + 0.65*p_test_catb[:,1]
#p_test_sub_data = model.predict_proba(test_data)

ak = test_sub.reset_index(drop=True).merge(pd.DataFrame(ensemble_pred_final_test,columns=["pred_test"]).reset_index(drop=True), left_index=True, right_index=True)



test_prev_ins_1 = test[test.Previously_Insured==1]

test_prev_ins_1["pred_test"] = 0



test_final_pred = ak[["id","pred_test"]]

test_final_pred.rename(columns = {'pred_test':'Response'},inplace=True)

test_final_pred = test_final_pred.sort_values(by=['id'])

test_final_pred.reset_index(drop=True,inplace=True)

test_final_pred.head()
test_final_pred.to_csv('test_final_pred.csv', index=False)