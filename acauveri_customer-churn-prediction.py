from numpy import array
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
print(os.listdir("../input"))
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
df_train=os.listdir("../input")
df_train=pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_train.shape
df_train.head()
df_train.nunique()
df_train.isnull().sum()
df_train.dtypes
df_train.drop(['customerID'],inplace=True,axis=1)
df_train.shape
df_train=df_train.astype('category')
df_train.dtypes

def label_encoding_churn(df_train):
    for i in range(len(df_train.columns)):
        col=df_train.columns[i]
        df_train[col]=LabelEncoder().fit_transform(df_train[col])
   
    return df_train
df_train=label_encoding_churn(df_train)
df_train.columns
target=df_train['Churn']
df_train.drop(['Churn'],axis=1,inplace=True)
clf=RandomForestClassifier().fit(df_train,target)
pred=clf.predict(df_train)
accuracy=accuracy_score(target,pred)
print("ACCURACY SCORE:",accuracy)
roc_auc_score(target,pred)
features_value=pd.DataFrame(clf.feature_importances_,columns=['features'])
len(df_train.columns)
features_value.shape
df_train.columns.shape

features_value['feature_name']=array(df_train.columns)
features_value
features_value['feature_code']=array(features_value.index)

features_value
features_value.plot.bar(x='feature_code', y='features', rot=0)

