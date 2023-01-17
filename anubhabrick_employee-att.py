import numpy as np

from imblearn.over_sampling import SMOTE

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection

from sklearn import metrics

from sklearn import preprocessing

from datetime import datetime

from sklearn import feature_selection

from sklearn import naive_bayes

from sklearn import tree

from sklearn import utils

from sklearn import ensemble

from sklearn import linear_model

from sklearn import neighbors

import random

import warnings

import xgboost

from scipy import stats

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

#ignoring future warning

warnings.simplefilter(action='ignore', category=FutureWarning)
df_p=pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

df_p=utils.shuffle(df_p,random_state=42)

df_p.info()
df_p["op"]=(df_p["Attrition"]=="Yes").astype(np.int)

df_p.drop("Attrition",axis=1,inplace=True)
for col in df_p.columns.values:

    print(str(col),df_p[str(col)].unique().shape[0])
l_drop=[]

for col in df_p.columns.values:

    if df_p[str(col)].unique().shape[0]==1:

        l_drop.append(str(col))



l_drop.append("EmployeeNumber")

df_p.drop(l_drop,axis=1,inplace=True)
l_str=[]

for col in df_p.columns.values:

    

    if type(df_p[str(col)][0])==str or df_p[str(col)].unique().shape[0]==1:

        l_str.append(str(col))

df_str=df_p[l_str].copy() 

df_str.columns.values
l_con=[]

for col in df_p.drop(l_str,axis=1).columns.values:

    if df_p[str(col)].unique().shape[0]>10:  

     l_con.append(str(col))

df_con=df_p[l_con].copy()  

l_con
l_cat=[]

for col in df_p.drop(l_str,axis=1).columns.values:

    if df_p[str(col)].unique().shape[0]<=10:  

     l_cat.append(str(col))

l_cat.remove("op")

df_cat=df_p[l_cat].copy()  

l_cat
for col in df_p[l_con].columns.values:

    if (df_p[str(col)]<=0).sum()>0:

        print("number of zeros or negative  in columns",str(col),(df_p[str(col)]<=0).sum())

for col in df_p[l_con].columns.values:

    if (df_p[str(col)]<=0).sum()>0:

        print("number of zeros   in columns",str(col),(df_p[str(col)]==0).sum())
def IQR(data):

    upper_quantile=data.quantile(0.75)

    lower_quantile=data.quantile(0.25)

    IQR=upper_quantile-lower_quantile

    outlier1=upper_quantile+1.5*IQR

    outlier2=lower_quantile-1.5*IQR

    return (IQR,outlier1,outlier2)
for col in df_p[l_con].columns.values:

    i,outlier1,outlier2=IQR(df_p[str(col)])

    print("upper_outliers",df_p[df_p[str(col)]>outlier1].shape[0],"column name",str(col),"theroritical_max",outlier1,"max",df_p[str(col)].max())

    print("lower_outliers",df_p[df_p[str(col)]<outlier2].shape[0],"column name",str(col),"theoritical_min",outlier2,"min",df_p[str(col)].min())
fig,axis=plt.subplots(figsize=(50,50))

sns.heatmap(df_p[l_con+["op"]].corr(),annot=True,ax=axis)

plt.tight_layout()
l=[]

l+=list(df_p[l_con+["op"]].corr()["op"][df_p[l_con+["op"]].corr()["op"]>0].index)

l
fig,axes=plt.subplots(nrows=3,ncols=4,figsize=(10,10))

axes1=axes.flatten()

index=0

for col in l_con:

    sns.boxplot(x="op",y=col,data=df_p,ax=axes1[index])

    index+=1

plt.tight_layout()
fig,axis=plt.subplots(figsize=(50,50))

sns.heatmap(df_p[l_cat+["op"]].corr(),annot=True,ax=axis)

plt.tight_layout()
l+=list(df_p[l_cat+["op"]].corr()["op"][df_p[l_cat+["op"]].corr()["op"]>0].index)

l
fig,axes=plt.subplots(nrows=4,ncols=2,figsize=(30,30))

axes1=axes.flatten()

index=0

for col in l_str:

    sns.countplot(y=col,hue="op",data=df_p,ax=axes1[index])

    index+=1

plt.tight_layout()


df_p=pd.get_dummies(df_p,columns=l_str)

df_p.columns.values
lc=['BusinessTravel_Non-Travel',

       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',

       'Department_Human Resources', 'Department_Research & Development',

       'Department_Sales', 'EducationField_Human Resources',

       'EducationField_Life Sciences', 'EducationField_Marketing',

       'EducationField_Medical', 'EducationField_Other',

       'EducationField_Technical Degree', 'Gender_Female', 'Gender_Male',

       'JobRole_Healthcare Representative', 'JobRole_Human Resources',

       'JobRole_Laboratory Technician', 'JobRole_Manager',

       'JobRole_Manufacturing Director', 'JobRole_Research Director',

       'JobRole_Research Scientist', 'JobRole_Sales Executive',

       'JobRole_Sales Representative', 'MaritalStatus_Divorced',

       'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_No',

       'OverTime_Yes']

fig,axis=plt.subplots(figsize=(50,50))

sns.heatmap(df_p[lc+["op"]].corr(),annot=True,ax=axis)

plt.tight_layout()
l+=list(df_p[lc+["op"]].corr()["op"][df_p[lc+["op"]].corr()["op"]>0].index)

y=df_p.op

X=df_p.drop("op",axis=1)

sc = preprocessing.StandardScaler()

x=sc.fit_transform(X)
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.2,random_state=42)
sm=SMOTE(random_state=42,k_neighbors=2)

xtrainsm,ytrainsm=sm.fit_sample(xtrain,ytrain)


lr = linear_model.LogisticRegression()

lr.fit(sc.transform(xtrainsm),ytrainsm)

testprediction=lr.predict(sc.transform(xtest))

print("Accuracy",metrics.accuracy_score(ytest,testprediction))

print("AUC",metrics.roc_auc_score(ytest,testprediction))

print("recall",metrics.recall_score(ytest,testprediction))

print(metrics.confusion_matrix(ytest,testprediction))

print(metrics.classification_report(ytest,testprediction))
rf = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',

                            max_depth=8, max_features='auto', max_leaf_nodes=None,

                            min_impurity_decrease=0.0, min_impurity_split=None,

                            min_samples_leaf=2, min_samples_split=5,

                            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,

                            oob_score=False, random_state=0, verbose=0, warm_start=False)

rf.fit(xtrainsm,ytrainsm)

testprediction=rf.predict(xtest)

trainprediction=rf.predict(xtrainsm)

print("Accuracy",metrics.accuracy_score(ytest,testprediction))

print("AUC",metrics.roc_auc_score(ytest,testprediction))

print("recall",metrics.recall_score(ytest,testprediction))

print("Accuracy train",metrics.accuracy_score(ytrainsm,trainprediction))

print("recall train",metrics.recall_score(ytrainsm,trainprediction))

nb=naive_bayes.GaussianNB()

nb.fit(xtrainsm,ytrainsm)

testprediction=nb.predict(xtest)

print("Accuracy",metrics.accuracy_score(ytest,testprediction))

print("AUC",metrics.roc_auc_score(ytest,testprediction))

print("recall",metrics.recall_score(ytest,testprediction))