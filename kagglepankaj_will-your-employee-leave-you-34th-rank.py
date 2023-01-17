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
import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv(r'/kaggle/input/Train.csv')

test=pd.read_csv(r'/kaggle/input/Test.csv')

sample=pd.read_csv(r'/kaggle/input/sample_submission.csv')
train.head()
sample.head()
train.shape
# Just done this to  keep original data for further analysis.

traindf=train.copy()

testdf=test.copy()
# Droped Employee id,as it is unique for all the entries.

traindf.drop('Employee_ID',axis=1,inplace=True)

testdf.drop('Employee_ID',axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

cat_features=traindf.columns[traindf.dtypes==object]

for col in cat_features:

    le.fit(traindf[col])

    testdf[col]=le.transform(testdf[col])

    traindf[col]=le.transform(traindf[col])
feature=traindf.drop('Attrition_rate',axis=1)

label=traindf.Attrition_rate
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

lgb=LGBMRegressor(max_depth=5,learning_rate=.01,n_estimators=100,subsample_for_bin=3000,subsample=.6,colsample_bytree=.6,num_leaves=20)

cv=KFold(n_splits=5,shuffle=True)

error=np.array([])

for trainind,testind in cv.split(feature,label):

    X_train,X_test,y_train,y_test=feature.loc[trainind],feature.loc[testind],label[trainind],label[testind]

    lgb.fit(X_train,y_train)

    y_pred=lgb.predict(X_test)

    error=np.append(error,np.sqrt(mean_squared_error(y_test,y_pred)))

error.mean()  
import xgboost as xgb

dmatrix=xgb.DMatrix(feature,label)

params={'objective':'reg:squarederror','max_depth':5,'colsample_bytree':.8,'eta':.02,'alpha':1}

result=xgb.cv(params,dmatrix,num_boost_round=1000,early_stopping_rounds=20,as_pandas=True,nfold=3)

result
lgb.fit(feature,label)

y_pred=lgb.predict(testdf)
y_pred=lgb.predict(testdf)

t=pd.read_csv(r'/kaggle/input/Test.csv')

save=t[['Employee_ID']]

save['Attrition_rate']=y_pred

save.to_csv('absolut_baseline.csv',index=False)
test.isnull().sum()
test.Age.fillna(train.Age.median(),inplace=True)

test.Time_of_service.fillna(1,inplace=True)

test.Education_Level.fillna(1,inplace=True)

test.Work_Life_balance.fillna(1,inplace=True)

test.VAR2.fillna(train.VAR2.median(),inplace=True)

test.VAR4.fillna(train.VAR4.median(),inplace=True)
train.Age.fillna(train.Age.median(),inplace=True)

train.Time_of_service.fillna(1,inplace=True)

train.Education_Level.fillna(1,inplace=True)

train.Work_Life_balance.fillna(1,inplace=True)

train.VAR2.fillna(train.VAR2.median(),inplace=True)

train.VAR4.fillna(train.VAR4.median(),inplace=True)
test.isnull().sum()
test.Pay_Scale.fillna(1,inplace=True)
cat_features=['Education_Level','Post_Level','Pay_Scale','Work_Life_balance']

for col in cat_features:

    train[col]=train[col].astype(str)

    test[col]=test[col].astype(str)
train.drop('Employee_ID',axis=1,inplace=True)

test.drop('Employee_ID',axis=1,inplace=True)
train.describe(exclude='number')
train.describe()
train['isloyal']=(train.Time_of_service>=10).astype(int)

train['isrecentpromoted']=(train.Time_since_promotion<=2).astype(int)

train['isloyalnotpromoted']=((train.Time_of_service>=10)&(train.Time_since_promotion>=2))

train['istopnotpromotes']=((train.Post_Level.astype(int)>=3)&(train.Time_since_promotion>=2)).astype(int)

train['islowernotpromotes']=((train.Post_Level.astype(int)<=3)&(train.Time_since_promotion>=2)).astype(int)
test['isloyal']=(test.Time_of_service>=10).astype(int)

test['isrecentpromoted']=(test.Time_since_promotion<=2).astype(int)

test['isloyalnotpromoted']=((test.Time_of_service>=10)&(test.Time_since_promotion>=2))

test['istopnotpromotes']=((test.Post_Level.astype(int)>=3)&(test.Time_since_promotion>=2)).astype(int)

test['islowernotpromotes']=((test.Post_Level.astype(int)<=3)&(test.Time_since_promotion>=2)).astype(int)
train['Agecat']=train.Age.apply(lambda x:1 if x<27 else 2 if x<37 else 3 if x<52 else 4)

test['Agecat']=test.Age.apply(lambda x:1 if x<27 else 2 if x<37 else 3 if x<52 else 4)
import itertools

cat_features=train.columns[train.dtypes==object]

new=pd.DataFrame(index=train.index)

for col1,col2 in itertools.combinations(cat_features,2):

    new[col1+col2]=train[col1]+train[col2]
new.head()
newt=pd.DataFrame(index=test.index)

for col1,col2 in itertools.combinations(cat_features,2):

    newt[col1+col2]=test[col1]+test[col2]

    
train.groupby('Compensation_and_Benefits')['Attrition_rate'].plot(kind='hist',bins=100)

plt.legend()
train.groupby('Hometown')['Attrition_rate'].plot(kind='hist',bins=100)

plt.legend()
train.groupby('Relationship_Status')['Attrition_rate'].plot(kind='hist',bins=100)

plt.legend()
train.groupby('Agecat')['Attrition_rate'].plot(kind='hist',bins=100)

plt.legend()
train.groupby('Time_since_promotion')['Attrition_rate'].plot(kind='hist',bins=100)

plt.legend()
train.groupby('Unit')['Attrition_rate'].plot(kind='hist',bins=100)

plt.legend()
train=train.join(new)

test=test.join(newt)
cat_features=train.columns[train.dtypes==object]

for col in cat_features:

    le.fit(train[col])

    test[col]=le.transform(test[col])

    train[col]=le.transform(train[col])
_=train.copy()

_.dropna(inplace=True)
feature=_.drop('Attrition_rate',axis=1)

label=_.Attrition_rate
feature.shape
from sklearn.tree import DecisionTreeRegressor

from sklearn.feature_selection import RFE

dt=DecisionTreeRegressor()

rfe=RFE(dt,n_features_to_select=15,step=1).fit(feature,label)
selected_fea=feature.columns[rfe.support_]
import xgboost as xgb

dmatrix=xgb.DMatrix(feature[selected_fea],label)

params={'objective':'reg:squarederror','max_depth':5,'colsample_bytree':.5,'eta':.02,'alpha':8}

result=xgb.cv(params,dmatrix,num_boost_round=1000,early_stopping_rounds=20,as_pandas=True,nfold=3)

result
xgb_cl=xgb.train(params,dmatrix,num_boost_round=231)

xgb.plot_importance(xgb_cl)
selected_col=['Compensation_and_BenefitsWork_Life_balance','Post_LevelPay_Scale','Pay_ScaleWork_Life_balance','Time_of_service','UnitWork_Life_balance','growth_rate','Age','Education_LevelPay_Scale','Decision_skill_possessWork_Life_balance','UnitPay_Scale','HometownPost_Level','HometownPay_Scale','Decision_skill_possessPay_Scale','VAR6']
lgb=LGBMRegressor(max_depth=3,learning_rate=.01,n_estimators=50,subsample_for_bin=5000,subsample=.8,colsample_bytree=.6,num_leaves=20)

cv=KFold(n_splits=5,shuffle=True)

error=np.array([])

for trainind,testind in cv.split(feature[selected_fea],label):

    X_train,X_test,y_train,y_test=feature.loc[trainind],feature.loc[testind],label[trainind],label[testind]

    lgb.fit(X_train,y_train)

    y_pred=lgb.predict(X_test)

    error=np.append(error,np.sqrt(mean_squared_error(y_test,y_pred)))

error.mean()  
lgb.fit(feature[selected_fea],label)

y_pred=lgb.predict(test[selected_fea])
#y_pred=xgb_cl.predict(xgb.DMatrix(test[selected_fea]))

t=pd.read_csv(r'/kaggle/input/Test.csv')

save=t[['Employee_ID']]

save['Attrition_rate']=y_pred

save.to_csv('finalsubmissionlgb.csv',index=False)