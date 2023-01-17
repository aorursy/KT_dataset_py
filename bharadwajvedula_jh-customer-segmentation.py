import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.ensemble import RandomForestClassifier  

from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold,RandomizedSearchCV

from sklearn.metrics import accuracy_score,classification_report

from sklearn.cluster import DBSCAN

from sklearn.pipeline import make_pipeline 

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier
df_train=pd.read_csv('../input/Train.csv')

df_test=pd.read_csv('../input/Test.csv')

df_ss=pd.read_csv('../input/sample_submission.csv')
df_train
df_test
data= pd.concat([df_test,df_train],axis='rows')
data= data.sort_values('ID')
data[0:60]
duplicate = data[data.duplicated('ID',keep=False)] 
duplicate
duplicate.isnull().sum()
duplicate_list_train=[]

duplicate_list_test=[]

for i in range(len(df_train)):

    if df_train['ID'].values[i]in duplicate['ID'].values:

        duplicate_list_train.append(1)

    else:

        duplicate_list_train.append(0)

        

for i in range(len(df_test)):

    if df_test['ID'].values[i]in duplicate['ID'].values:

        duplicate_list_test.append(1)

    else:

        duplicate_list_test.append(0)
np.bincount(duplicate_list_train),np.bincount(duplicate_list_test)
duplicate_rows=pd.Series(duplicate_list_train,name="duplicate_rows")

duplicate_rows
df_train=pd.concat((df_train,pd.Series(duplicate_list_train,name="duplicate_rows")),axis="columns")

df_test=pd.concat((df_test,pd.Series(duplicate_list_test,name="duplicate_rows")),axis="columns")
# handling missing values and categorical data 







#train

df_train['Gender']=df_train['Gender'].map({'Male':1,'Female':0})

df_train['Ever_Married']=df_train['Ever_Married'].fillna(df_train['Ever_Married'].mode(),axis='rows')

df_train["Graduated"]=df_train["Graduated"].fillna(df_train["Graduated"].mode(),axis='rows')

df_train["Profession"]=df_train["Profession"].fillna(df_train["Profession"].mode(),axis='rows')

df_train['Work_Experience']=df_train['Work_Experience'].fillna(df_train['Work_Experience'].median())

df_train['Family_Size']=df_train['Family_Size'].fillna(df_train['Family_Size'].median())

df_train['Var_1']=df_train['Var_1'].fillna('cat_6')



df_train['Segmentation']=df_train['Segmentation'].map({'A':0,'B':1,'C':2,'D':3})  



#test

df_test['Gender']=df_test['Gender'].map({'Male':1,'Female':0})

df_test['Ever_Married']=df_test['Ever_Married'].fillna(df_train['Ever_Married'].mode(),axis='rows')

df_test["Graduated"]=df_test["Graduated"].fillna(df_train["Graduated"].mode(),axis='rows')

df_test["Profession"]=df_test["Profession"].fillna(df_train["Profession"].mode(),axis='rows')

df_test['Work_Experience']=df_test['Work_Experience'].fillna(df_test['Work_Experience'].median())

df_test['Family_Size']=df_test['Family_Size'].fillna(df_test['Family_Size'].median())

df_test['Var_1']=df_test['Var_1'].fillna('cat_6')
#one-hotencoding 

#train

married_dummies=pd.get_dummies(df_train['Ever_Married'],drop_first=True,prefix='m')

grad_dummies=pd.get_dummies(df_train['Graduated'],drop_first=True,prefix='grad')

prof_dummies=pd.get_dummies(df_train['Profession'],drop_first=True,prefix='prof')

spend_dummies=pd.get_dummies(df_train['Spending_Score'],drop_first=True)

cat_dummies=pd.get_dummies(df_train['Var_1'],drop_first=True)



train_dummies=pd.concat([married_dummies,grad_dummies,prof_dummies,spend_dummies,cat_dummies],copy=False,axis='columns')



#test

married_dummies_test=pd.get_dummies(df_test['Ever_Married'],drop_first=True,prefix='m')

grad_dummies_test=pd.get_dummies(df_test['Graduated'],drop_first=True,prefix='grad')

prof_dummies_test=pd.get_dummies(df_test['Profession'],drop_first=True,prefix='prof')

spend_dummies_test=pd.get_dummies(df_test['Spending_Score'],drop_first=True)

cat_dummies_test=pd.get_dummies(df_test['Var_1'],drop_first=True)



test_dummies=pd.concat([married_dummies_test,grad_dummies_test,prof_dummies_test,spend_dummies_test,cat_dummies_test],copy=False,axis='columns')
df_train=pd.concat([df_train,train_dummies],axis='columns',copy=False)

df_test=pd.concat([df_test,test_dummies],axis='columns',copy=False)



df_train.drop(['Ever_Married','Graduated','Profession','Spending_Score','Var_1'],axis='columns',inplace=True)

df_test.drop(['Ever_Married','Graduated','Profession','Spending_Score','Var_1'],axis='columns',inplace=True)
X=df_train.loc[:,df_train.columns!='Segmentation']

y=df_train.loc[:,'Segmentation']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1010,test_size=0.15)

X_train.shape,X_test.shape
inverse_map={0:'A',1:'B',2:'C',3:'D'}
xgc = make_pipeline(StandardScaler(),XGBClassifier(objective="multi:softmax",n_jobs=-1,num_class=4,eval_metric="auc",learning_rate =0.1,

                                                         n_estimators=1000,

                                                         max_depth=5,

                                                         min_child_weight=1,

                                                         gamma=0,

                                                         subsample=0.8,

                                                         colsample_bytree=0.8),verbose=1)

xgc.fit(X_train,y_train)



y_pred_xgc=xgc.predict(X_test)

print(accuracy_score(y_test,y_pred_xgc))

print(classification_report(y_pred_xgc,y_test))
xgc.fit(X,y)

y_sub_xgc=xgc.predict(df_test)



sub1=pd.DataFrame({'ID':df_ss['ID'],'Segmentation':y_sub_xgc})

sub1['Segmentation']=sub1['Segmentation'].map(inverse_map)

sub1.to_csv("xgc_sub2.csv",index=False)

print(sub1.head())
duplicate1= duplicate[duplicate['Segmentation'].notna()]
duplicate1
df_non_duplicate=df_test.loc[df_test['duplicate_rows']==0]

df_non_duplicate
y_pred_hc=xgc.predict(df_non_duplicate)
len(y_pred_hc)
sub2=pd.DataFrame({'ID':df_non_duplicate['ID'],"Segmentation":y_pred_hc})

sub2['Segmentation']=sub2['Segmentation'].map(inverse_map)

sub3=pd.DataFrame({'ID':duplicate1['ID'],"Segmentation":duplicate1['Segmentation']})
sub2,sub3
final_sub=pd.concat((sub2,sub3),axis="rows",copy=False)

final_sub.to_csv("hc_sub.csv",index=False)

print(final_sub.head())
final_sub