import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold,StratifiedKFold,train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

import os

import eli5

from eli5.sklearn import PermutationImportance

import warnings

warnings.filterwarnings('ignore')

path='../input/av-janatahack-crosssell-prediction/'

train_df=pd.read_csv(os.path.join(path,'train.csv'))

test_df=pd.read_csv(os.path.join(path,'test.csv'))

submission_df=pd.read_csv(os.path.join(path,'sample.csv'))
#Check for duplicates and remove



print(train_df.shape)



train_df=train_df.drop_duplicates(subset=[ele for ele in list(train_df.columns) if ele not in ['id']])



print(train_df.shape)
sns.distplot(train_df['Annual_Premium'])
#Data is left Skewed as we can see from above distplot

train_df['Annual_Premium']=np.log(train_df['Annual_Premium'])

sns.distplot(train_df['Annual_Premium'])
#Checking correlation between features

plt.figure(figsize=(10,10))

sns.heatmap(train_df.corr(),annot=True)
sns.barplot(train_df['Response'],train_df['Response'].value_counts())
combine_set=pd.concat([train_df,test_df])

le=LabelEncoder()

combine_set['Gender']=le.fit_transform(combine_set['Gender'])

combine_set['Vehicle_Damage']=le.fit_transform(combine_set['Vehicle_Damage'])



fe=combine_set.groupby('Vehicle_Age').size()/len(combine_set)

combine_set['Vehicle_Age']=combine_set['Vehicle_Age'].apply(lambda x: fe[x])#can even try Label encoding or OHE



combine_set.head(5)
#Data set Preparation

train_df=combine_set[combine_set['Response'].isnull()==False]

test_df=combine_set[combine_set['Response'].isnull()==True]

X=train_df.drop(['id','Response'],axis=1)

y=train_df['Response'] 

X_main_test=test_df.drop(['id','Response'],axis=1)



X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=294)
lg=LGBMClassifier(boosting_type='gbdt',n_estimators=500,depth=10,learning_rate=0.04,objective='binary',metric='auc',is_unbalance=True,

                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=294,n_jobs=-1)



lg.fit(X_train,y_train)

print(roc_auc_score(y_val,lg.predict_proba(X_val)[:,1]))
#Check for Permutation Importance of Features

perm = PermutationImportance(lg,random_state=294).fit(X_val, y_val)

eli5.show_weights(perm,feature_names=X_val.columns.tolist())

submission_df['Response']=np.array(lg.predict_proba(X_main_test)[:,1])

submission_df.to_csv('main_test.csv',index=False)

submission_df.head(5)