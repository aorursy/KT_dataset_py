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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sas

%matplotlib inline 
test = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/test.csv")

train = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/train.csv")
#storing id's in some variables so that it can be used in future for prediction

train_id = train["pet_id"]

test_id = test["pet_id"]
train["days"] = (pd.to_datetime(train["listing_date"]) - pd.to_datetime(train["issue_date"])).dt.days

test["days"] = (pd.to_datetime(test["listing_date"]) - pd.to_datetime(test["issue_date"])).dt.days

## here i calculated the difference in listing nd issue date store it in a column named days.
train.head()
train.info()
train.describe()
test.shape
test.info()
test.describe()
train.isnull().sum().sort_values(ascending=False)[0:2]
test.isnull().sum().sort_values(ascending=False)[0:2]
list_drop=["pet_id","issue_date","listing_date"]



for col in list_drop:

    del train[col]

    del test[col]

    
train.condition.value_counts(dropna=False)
test.condition.value_counts(dropna=False)
train.condition.fillna(-1,inplace=True)
test.condition.fillna(-1,inplace=True)
train.head()
mat=train.corr()

fig,ax = plt.subplots(figsize = (10,10))

sas.heatmap(mat,annot = True, annot_kws={'size': 12})
test.shape
test.head()
Y1=train["breed_category"]

Y2=train["pet_category"]

print(Y1.shape)
train.drop("pet_category",axis=1,inplace=True)



train.drop("breed_category",axis=1,inplace=True)
final=pd.concat([train,test],axis=0)
## Appyling one_hot_enconding to convert categorical data into numerical data

def One_hot_encoding(columns):

    final_df=final

    i=0

    for fields in columns:

        df1=pd.get_dummies(final[fields],drop_first=True)

        

        final.drop([fields],axis=1,inplace=True)

        if i==0:

            final_df=df1.copy()

        else:           

            final_df=pd.concat([final_df,df1],axis=1)

        i=i+1

       

        

    final_df=pd.concat([final,final_df],axis=1)

        

    return final_df
columns=["condition","color_type","X1","X2"]

# i applied it to all the columns which have categories in it

## in place of one_hot_encoding you can also use label encoder for encoding but your accuracy should be compromised
df_final=One_hot_encoding(columns)
df_final.head()
df_final.shape
df_final.shape
from sklearn import preprocessing

# Get column names first

names = df_final.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(df_final)

df_final = pd.DataFrame(scaled_df, columns=names)
## As color_type is present in both dataset have maximum no. of classes present in it

## try to analyze it that is there any class diffrence in both train and test data set
color_type_train=pd.get_dummies(train["color_type"])

color_type_test=pd.get_dummies(test["color_type"])
co1=[]

for i in color_type_train:

    if  i not in color_type_test :

        co1.append(i)

co1
## as we can see that train dataset of two extra classes which are not present in test dataset 

## so these are not of our use 

## we can safely drop these columns which we added during applying one_hot_encoding
df_final.drop('Black Tiger',axis=1,inplace=True)

df_final.drop("Brown Tiger",axis=1,inplace=True)
df_final.columns
cols = []

count = 1

for column in df_final.columns:

    cols.append(count)

    count+=1

    continue

    

df_final.columns = cols
df_final.columns
df_train=df_final.iloc[:18834,:]

df_test=df_final.iloc[18834:,:]
X=df_train
df_test.shape
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(X,Y2)
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
## Hyper Parameter Optimization





n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
from xgboost import XGBClassifier
xgb1 = XGBClassifier()
random_cv_01 = RandomizedSearchCV(estimator=xgb1,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv_01.fit(x1_train,y1_train)
random_cv_01.best_estimator_
xgb1=XGBClassifier(base_score=0.25, booster='gbtree', colsample_bylevel=1,

       colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

       importance_type='gain', interaction_constraints='',

       learning_rate=0.1, max_delta_step=0, max_depth=2,

       min_child_weight=1, monotone_constraints='()',

       n_estimators=900, n_jobs=0, num_parallel_tree=1,

       objective='multi:softprob', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=None, subsample=1,

       tree_method='exact', validate_parameters=1, verbosity=None)
xgb1.fit(x1_train,y1_train)
new_feat=xgb1.predict(X)

out_01=xgb1.predict(df_test)

vald_01=xgb1.predict(x1_test)
X2=X
X2["output1"]=new_feat
df_test_2=df_test
df_test_2["output1"]=out_01
x2_train,x2_test,y2_train,y2_test=train_test_split(X2,Y1)
xgb2=XGBClassifier()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
## Hyper Parameter Optimization





n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
random_cv_02 = RandomizedSearchCV(estimator=xgb2,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv_02.fit(x2_train,y2_train)
random_cv_02.best_estimator_
xgb2=XGBClassifier(base_score=0.25, booster='gblinear', colsample_bylevel=None,

       colsample_bynode=None, colsample_bytree=None, gamma=None, gpu_id=-1,

       importance_type='gain', interaction_constraints=None,

       learning_rate=0.1, max_delta_step=None, max_depth=3,

       min_child_weight=4, monotone_constraints=None,

       n_estimators=1100, n_jobs=0, num_parallel_tree=None,

       objective='multi:softprob', random_state=0, reg_alpha=0,

       reg_lambda=0, scale_pos_weight=None, subsample=None,

       tree_method=None, validate_parameters=1, verbosity=None)
xgb2.fit(x2_train,y2_train)
out_02=xgb2.predict(df_test)

vald_02=xgb2.predict(x2_test)
from sklearn.metrics  import f1_score
s1=f1_score(y1_test,vald_01,average='weighted')

s2=f1_score(y2_test,vald_02,average='weighted')

accuracy=100*((s1+s2)/2)

accuracy
sub_new=pd.DataFrame({

    "pet_id":test_id,

    "breed_category":out_02,

    "pet_category":out_01

})

sub_new.to_csv("sub_new_13.csv",index=False)