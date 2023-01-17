import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import scipy.stats as st 

import os

from tqdm import tqdm

#os.chdir('D:\Shashi Katteri\Desktop\data sets to work on\indian-liver-patient-records')
df=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
df.head()
df.info()
df.describe()
df.Dataset.value_counts()
df1=df

df.columns
df1['Dataset']=df1['Dataset'].apply(lambda x:0 if x==2 else x)
df1.Dataset.value_counts()
df1=df1.fillna(method='bfill')
df1.info()
sns.pairplot(data=df1,diag_kind='kde',hue='Dataset')
for i in df1.columns:

    sns.boxplot(y=df1[i],x=df1['Dataset'])

    plt.show()
df1=pd.get_dummies(df1,columns=['Gender'],drop_first=True)
sns.countplot(df1['Gender_Male'],hue=df1['Dataset'])
df1.corr()
df1.info()
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,confusion_matrix,classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,RandomizedSearchCV,cross_val_score
x=df1.drop('Dataset',axis=1)

y=df1['Dataset']
def mod_score(algo,x,y,params=None):

    cv1=cross_val_score(algo,x,y,cv=5,scoring='accuracy')

    cv2=cross_val_score(algo,x,y,cv=5,scoring='roc_auc')

    print('Accuracy : ',cv1.mean())

    print('ROC AUC score : ',cv2.mean())

    

    
def rand_search(algo,params,x,y):

    rs=RandomizedSearchCV(algo,param_distributions=params,random_state=0,n_jobs=-1,n_iter=100,scoring='roc_auc',cv=10)

    mod=rs.fit(x,y)

    print(mod.best_score_)

    return mod.best_params_
rfc_params={'n_estimators':st.randint(50,300),

    'criterion':['gini','entropy'],

    'max_depth':st.randint(2,20),

    'min_samples_split':st.randint(2,100),

    'min_samples_leaf':st.randint(2,100)}

lgb_params={ 'num_leaves':st.randint(31,60),

   'max_depth':st.randint(2,20),

    'learning_rate':st.uniform(0,1),

    'n_estimators':st.randint(50,300),

    'min_split_gain':st.uniform(0,0.3)}
rbp=rand_search(RandomForestClassifier(),rfc_params,x,y)
lbp=rand_search(LGBMClassifier(),lgb_params,x,y)
models={'Logistic Regression':LogisticRegression(solver='liblinear'),'Random Forest':RandomForestClassifier(**rbp),

       'Light GBM(Boosting)':LGBMClassifier(**lbp),'Gausian Naive Bayes':GaussianNB()

       }

params={'Logistic Regression':{'solver':'liblinear'},'Random Forest':rbp,

       'Light GBM(Boosting)':lbp,'Gausian Naive Bayes':None}
for i in models.keys():

    print(i,'\n')

    mod_score(models[i],x,y)

    print('\n')
rfc=RandomForestClassifier(**rbp)

mod=rfc.fit(x,y)

y=mod.feature_importances_
plt.figure(figsize=(10,10))

sns.barplot(y=y,x=x.columns)
df3=df1
#df3=pd.get_dummies(df3,columns=['Gender'],drop_first=True)

#df3=df3.fillna(method='bfill')

df3.info()
x1=df3.drop('Dataset',axis=1)

y=df3['Dataset']
#d=x1

ss=StandardScaler()

d=ss.fit_transform(x1)

x1['sum']=d.sum(axis=1)

x1['min']=d.min(axis=1)

x1['max']=d.max(axis=1)

x1['skew']=st.skew(d,axis=1)

x1['kurt']=st.kurtosis(d,axis=1)

x1['std']=d.std(axis=1)
x1.info()
rbp1=rand_search(RandomForestClassifier(),rfc_params,x1,y)
lbp1=rand_search(LGBMClassifier(),lgb_params,x1,y)
models1={'Logistic Regression':LogisticRegression(solver='liblinear'),'Random Forest':RandomForestClassifier(**rbp1),

       'Light GBM(Boosting)':LGBMClassifier(**lbp1),'Gausian Naive Bayes':GaussianNB()

       }

params={'Logistic Regression':{'solver':'liblinear'},'Random Forest':rbp,

       'Light GBM(Boosting)':lbp,'Gausian Naive Bayes':None}
for i in models1.keys():

    print(i,'\n')

    mod_score(models1[i],x1,y)

    print('\n')
lgb=LGBMClassifier(**lbp1)

mod=lgb.fit(x1,y)

f=mod.feature_importances_

plt.figure(figsize=(15,15))

sns.barplot(y=f,x=x1.columns)