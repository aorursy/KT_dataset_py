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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv('/kaggle/input/Train.csv')

test=pd.read_csv('/kaggle/input/Test.csv')

sample=pd.read_csv('/kaggle/input/Sample.csv')

train.head()
train.describe()
len(set(test.ID)-set(train.ID)),test.shape #295 test case contains the new values.
np.intersect1d(train.ID,test.ID)
train.loc[train.ID==458996]
test.loc[test.ID==458996]
train.describe(exclude='number')
train.Segmentation.value_counts()/len(train)*100 #Balanced
train.isnull().sum() ##for the shake of simplicity fill by convention.
sns.heatmap(train.corr(),annot=True)
#Filled based on intution

train.Graduated.fillna('No',inplace=True)

train.Profession.fillna('unk',inplace=True)#unk means unknown.

train.Work_Experience.fillna(0,inplace=True)

train.Family_Size.fillna(5,inplace=True)

train.Var_1.fillna('unk',inplace=True)
_,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(10,5))



sns.distplot(train.loc[(train.Ever_Married=='Yes')&(train.Graduated=='Yes'),'Age'],label='MG',ax=ax1)

sns.distplot(train.loc[(train.Ever_Married=='No')&(train.Graduated=='Yes'),'Age'],label='UMG',ax=ax1)



sns.distplot(train.loc[(train.Ever_Married=='Yes')&(train.Graduated=='No'),'Age'],label='MUG',ax=ax2)

sns.distplot(train.loc[(train.Ever_Married=='No')&(train.Graduated=='No'),'Age'],label='UMUG',ax=ax2)

plt.legend()
sns.distplot(train.loc[(train.Ever_Married.isnull())&(train.Graduated=='Yes'),'Age'],label='G')

sns.distplot(train.loc[(train.Ever_Married.isnull())&(train.Graduated=='No'),'Age'],label='UG')

plt.legend()
# Filling married or not

train.loc[(train.Ever_Married.isnull())&(train.Graduated=='Yes')&(train.Age<35),'Ever_Married']=train.loc[(train.Ever_Married.isnull())&(train.Graduated=='Yes')&(train.Age<35),'Ever_Married'].fillna('No')

train.loc[(train.Ever_Married.isnull())&(train.Graduated=='Yes')&(train.Age>=35),'Ever_Married']=train.loc[(train.Ever_Married.isnull())&(train.Graduated=='Yes')&(train.Age>=35),'Ever_Married'].fillna('Yes')



train.loc[(train.Ever_Married.isnull())&(train.Graduated=='No')&(train.Age<30),'Ever_Married']=train.loc[(train.Ever_Married.isnull())&(train.Graduated=='No')&(train.Age<30),'Ever_Married'].fillna('No')

train.loc[(train.Ever_Married.isnull())&(train.Graduated=='No')&(train.Age>=30),'Ever_Married']=train.loc[(train.Ever_Married.isnull())&(train.Graduated=='No')&(train.Age>=30),'Ever_Married'].fillna('Yes')
train.isnull().sum()
test.isnull().sum()
test.Graduated.fillna('No',inplace=True)

test.Profession.fillna('unk',inplace=True)

test.Work_Experience.fillna(0,inplace=True)

test.Family_Size.fillna(3,inplace=True)

test.Var_1.fillna('unk',inplace=True)
test.loc[(test.Ever_Married.isnull())&(test.Graduated=='Yes')&(test.Age<35),'Ever_Married']=test.loc[(test.Ever_Married.isnull())&(test.Graduated=='Yes')&(test.Age<35),'Ever_Married'].fillna('No')

test.loc[(test.Ever_Married.isnull())&(test.Graduated=='Yes')&(test.Age>=35),'Ever_Married']=test.loc[(test.Ever_Married.isnull())&(test.Graduated=='Yes')&(test.Age>=35),'Ever_Married'].fillna('Yes')



test.loc[(test.Ever_Married.isnull())&(test.Graduated=='No')&(test.Age<30),'Ever_Married']=test.loc[(test.Ever_Married.isnull())&(test.Graduated=='No')&(test.Age<30),'Ever_Married'].fillna('No')

test.loc[(test.Ever_Married.isnull())&(test.Graduated=='No')&(test.Age>=30),'Ever_Married']=test.loc[(test.Ever_Married.isnull())&(test.Graduated=='No')&(test.Age>=30),'Ever_Married'].fillna('Yes')
cat_features=train.columns[train.dtypes==object].drop('Segmentation')
import itertools

new=pd.DataFrame(index=train.index)

for col1,col2 in itertools.combinations(cat_features,2):

    new[col1+'_'+col2]=train[col1]+train[col2]
import itertools

newt=pd.DataFrame(index=test.index)

for col1,col2 in itertools.combinations(cat_features,2):

    newt[col1+'_'+col2]=test[col1]+test[col2]
cat_features_=train.columns[train.dtypes==object].drop('Segmentation')
label=train.Segmentation.copy()

label.replace(dict(zip(['A','B','C','D'],[0,1,2,3])),inplace=True)

import category_encoders

te=category_encoders.TargetEncoder()

te.fit(train[cat_features],label)

encoded=te.transform(train[cat_features])

encodedt=te.transform(test[cat_features])
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for col in cat_features:

    train[col]=le.fit_transform(train[col])

    test[col]=le.transform(test[col])
train.head()
train.shape
feature,label=train.drop(['Segmentation'],axis=1),train.Segmentation
feature=feature.join(encoded.add_suffix('enc')) #adding Target Encoded attributes
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

xgb=XGBClassifier()

lgb=LGBMClassifier()
from sklearn.model_selection import GridSearchCV

xgb=XGBClassifier(n_jobs=-1)

params={'max_depth':[3,5],'subsample':[.8],'clsample_bytree':[.6],'n_estimators':[50,100,200,250,300]}

gs=GridSearchCV(param_grid=params,estimator=xgb,n_jobs=-1)

gs.fit(feature,label)

gs.best_params_
from sklearn.model_selection import GridSearchCV

lgb=LGBMClassifier(n_jobs=-1)

params={'max_depth':[3,5],'subsample':[.8,1],'clsample_bytree':[.6,1],'n_estimators':[50,100,200,250,300],'num_leaves':[20,30,40],'subsample_for_bin':[5000,7000,8000]}

gs=GridSearchCV(param_grid=params,estimator=lgb,n_jobs=-1)

gs.fit(feature,label)

gs.best_params_
from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import SelectKBest

skb=SelectKBest(mutual_info_classif,k=10)

selected_data=skb.fit_transform(feature,label)
selected_data=pd.DataFrame(skb.inverse_transform(selected_data),columns=feature.columns)

selected_col=selected_data.columns[selected_data.var()!=0]
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



#{'clsample_bytree': 0.6,'max_depth': 3,'n_estimators': 250,'num_leaves': 20,'subsample': 0.8,'subsample_for_bin': 5000}

lgb=LGBMClassifier(clsample_bytree=0.6,max_depth=3,n_estimators=250,num_leaves=20,subsample=0.8,subsample_for_bin=5000)



#{'clsample_bytree': 0.6, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8}

xgb=XGBClassifier(max_depth=3,subsample=.8,colsample_bytree=.6,n_estimators=50)



rf=RandomForestClassifier()

knn=KNeighborsClassifier()

svm=SVC(probability=True)



models=(('xgb',xgb),('lgb',lgb),('svm',svm),('knn',knn),('rf',rf))

vc=VotingClassifier(estimators=models,n_jobs=-1)
label=train.Segmentation

skf=StratifiedKFold(n_splits=5,random_state=123)

performance={}

for model,name in zip([lgb,xgb,rf,knn,svm],['lgm','xgb','rf','knn','svm']):

    accuracy=np.array([])

    for trainind,testind in skf.split(feature,label):

        X_train,X_test,y_train,y_test=feature.loc[trainind],feature.loc[testind],label[trainind],label[testind]

        model.fit(X_train,y_train)

        accuracy=np.append(accuracy,accuracy_score(y_test,model.predict(X_test)))

    performance[name] = accuracy.mean()  
pd.Series(performance)
model=xgb

model.fit(feature,label)
sns.barplot(x=model.feature_importances_,y=feature.columns)
def fitpredict(estimator):

    estimator.fit(feature,label)

    return estimator.predict_proba(feature)

df=pd.DataFrame(index=list(range(len(feature))))
df=df.join(pd.DataFrame(fitpredict(lgb)).add_suffix('xgb'))

df=df.join(pd.DataFrame(fitpredict(rf)).add_suffix('rf'))

df=df.join(pd.DataFrame(fitpredict(svm)).add_suffix('svm'))

df=df.join(pd.DataFrame(fitpredict(knn)).add_suffix('knn'))
df.shape
df.head()
modelblend=XGBClassifier(max_depth=3,subsample=.8,colsample_bytree=.6,n_estimators=100)

modelblend.fit(df,label)
def fitpredict(estimator):

    estimator.fit(feature,label)

    return estimator.predict_proba(test.join(encodedt.add_suffix('enc'))[feature.columns])

dft=pd.DataFrame(index=list(range(len(test))))
dft=dft.join(pd.DataFrame(fitpredict(lgb)).add_suffix('xgb'))

dft=dft.join(pd.DataFrame(fitpredict(rf)).add_suffix('rf'))

dft=dft.join(pd.DataFrame(fitpredict(svm)).add_suffix('svm'))

dft=dft.join(pd.DataFrame(fitpredict(knn)).add_suffix('knn'))
dft.shape,test.shape
final_pred=model.predict(test.join(encodedt.add_suffix('enc'))[feature.columns])

#final_pred=modelblend.predict(dft)
pd.Series(final_pred).value_counts()/len(final_pred)*100
t=pd.read_csv('/kaggle/input/Test.csv')

submission=t[['ID']]

submission['Segmentation']=final_pred
common_id=np.intersect1d(train.ID,test.ID)
submission.set_index('ID',inplace=True)
submission.loc[common_id,'Segmentation']=np.array(train.set_index('ID').loc[common_id,'Segmentation'])
submission.reset_index(inplace=True)
submission.to_csv('Targetinc.csv',index=False)