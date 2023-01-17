



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 

sns.set_style(style="darkgrid")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_train=pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

df_train.head()
#Data info

data_dims=df_train.shape

data_info=df_train.info()

print(data_dims,data_info)
#Null Check



null_data=df_train.isnull().sum()

null_data
#Separate Numerical and Categorical features



num_feats=df_train.select_dtypes(['int64','float64']).columns

cat_feats=df_train.select_dtypes(['object']).columns



feats=[num_feats,cat_feats]

feats

#Check Imbalancing of data 

sns.countplot(df_train['Response'],palette="twilight")
#Feature Exploration - > Categorical Types 



#1.Gender

sns.countplot(df_train['Gender'],palette='Set3')





#One Hot Encoding on Gender

df_train['Gender']=pd.get_dummies(df_train['Gender'],drop_first=True)

#2.Vehicle_Damage





sns.countplot(df_train['Vehicle_Damage'],palette='brg')





#One Hot Encoding on Vehicle_Damage

df_train['Vehicle_Damage']=pd.get_dummies(df_train['Vehicle_Damage'],drop_first=True)

#3.Vehicle_Age





va_counts=df_train['Vehicle_Age'].value_counts()



#3.1 String Handling

#3.2 Convert this feature into 3 distinct categories  0-1,1-2,>2 -> 0,1,2 (Label Encoding)









def string_extract(x):

    X=x.split()

    

    if(len(X)==2):

        return 1

    if(X[0]=='<'):

        return 0

    if(X[0]=='>'):

        return 2

    



df_train['Vehicle_Age_n']=df_train['Vehicle_Age'].apply(lambda x:string_extract(x))

sns.countplot(df_train['Vehicle_Age'],palette='brg')
sns.countplot(df_train['Vehicle_Damage'],palette='twilight')
sns.countplot(df_train['Previously_Insured'])
sns.countplot(df_train['Previously_Insured'],hue=df_train['Vehicle_Damage'],palette='twilight')
sns.countplot(df_train['Previously_Insured'],hue=df_train['Vehicle_Age'],palette='twilight')
sns.countplot(df_train['Vehicle_Damage'],hue=df_train['Vehicle_Age'],palette='Set3')
#Feature Exploration - > Numerical Types 



#1.Age 

sns.distplot(df_train['Age'],color='r')

age_desc=df_train['Age'].describe()

age_desc
#2.Annual_Premium

sns.distplot(df_train['Annual_Premium'])

prem_desc=df_train['Annual_Premium'].describe()

prem_desc
plt.scatter(df_train['Vintage'],df_train['Annual_Premium'])
df_train.drop(['id','Vehicle_Age'],inplace=True,axis=1)
imp_features=pd.DataFrame(df_train.corr()['Response'].sort_values(ascending=False))

imp_features.columns=['IMP']

indx=imp_features.index







plt.figure(figsize=(25,10))

b=sns.barplot(x=indx,y=imp_features['IMP'])

b.set_xlabel("Features",fontsize=20)

b.set_ylabel("Co-Relation" ,fontsize=20)
from sklearn.feature_selection import SelectKBest,f_classif



X=df_train.drop('Response',axis=1)

Y=df_train['Response']







selector_model=SelectKBest(score_func=f_classif,k='all')

selector=selector_model.fit(X,Y)



cols=X.columns

df_features = pd.DataFrame(cols)

df_scores = pd.DataFrame(selector.scores_)



df_new = pd.concat([df_features, df_scores], axis=1)

df_new.columns = ['Features', 'Score']



df_new = df_new.sort_values(by='Score', ascending=False)

df_new

imp_feature=df_new['Features']





indx=df_new['Features']

plt.figure(figsize=(25,10))

b=sns.barplot(x=indx,y=df_new['Score'])

b.set_xlabel("Features",fontsize=20)

b.set_ylabel("Co-Relation" ,fontsize=20)





imp_feature

imp_f=imp_feature[:6]

df_train[imp_f].head()
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

import xgboost
X=df_train[imp_f]

Y=df_train['Response']



x_train,x_test,y_train,y_test = train_test_split(X,Y, random_state = 0)
logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred = logreg.predict_proba(x_test)[:,1]

roc_auc_score(y_test,y_pred)
xgb1=xgboost.XGBClassifier()

xgb1.fit(x_train,y_train)

y_pred = xgb1.predict_proba(x_test)[:,1]

roc_auc_score(y_test,y_pred)
xgb1=xgboost.XGBClassifier(min_child_weight= 5, max_depth= 4, learning_rate = 0.25, 

                           gamma= 0.2, colsample_bytree= 0.7)

xgb1.fit(x_train,y_train)

y_pred = xgb1.predict_proba(x_test)[:,1]

roc_auc_score(y_test,y_pred)