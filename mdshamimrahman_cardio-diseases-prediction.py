
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df=pd.read_csv("../input/heart-diseases/datasets_4123_6408_framingham.csv")
df.head(5)
df.shape
df.isnull().sum()
##Handeling Nan

mean=df['education'].mean()


mean
df.education=df.education.fillna(mean)
mc=df['cigsPerDay'].mean()
df.cigsPerDay=df.cigsPerDay.fillna(mc)
mbp=df['BPMeds'].mean()
df.BPMeds=df.BPMeds.fillna(mbp)
mch=df['totChol'].mean()
df.totChol=df.totChol.fillna(mch)
mb=df['BMI'].mean()
df.BMI=df.BMI.fillna(mb)
mr=df['heartRate'].mean()
df.heartRate=df.heartRate.fillna(mr)
mg=df['glucose'].mean()
df.glucose=df.glucose.fillna(mg)
df.info()
df.isnull().sum()
df.describe().round(2)
## To see the correaltion

df.corr()
##Split the features and label

x=df.drop(['TenYearCHD'],axis=1)


y=df['TenYearCHD']
##Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
    
sb=SelectKBest(f_classif)
sb.fit(x,y)
##Find out the score
score=pd.DataFrame(sb.scores_,columns=['Scores'])
score
col_n=pd.DataFrame(x.columns)
final_features=pd.concat([score,col_n],axis=1)
final_features.nlargest(14,'Scores')
##Drop the last 5 features

x=x.drop(['BMI','prevalentStroke','cigsPerDay','education','heartRate'],axis=1)
x.head(2)
##Splited the testsize
from sklearn.model_selection import train_test_split


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.30)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
rf.score(xtest,ytest)
rf.predict(xtest)
##Logistic Regression

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)
##Svm

from sklearn.svm import SVC
sv=SVC()
sv.fit(xtrain,ytrain)
sv.score(xtest,ytest)
##Decission Tree

from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
dt.score(xtest,ytest)
import seaborn as sns

df=pd.read_csv('../input/heart-diseases/datasets_4123_6408_framingham.csv')
df.head(2)
sns.countplot(df['TenYearCHD'],palette=['#137909','#ff0707'])
sns.kdeplot(df['glucose'], shade=True)
sns.countplot(df['currentSmoker'],palette=['#137909','#ff0707'])
df['currentSmoker'].value_counts()
sns.countplot(x='age',hue='TenYearCHD',data=df,palette=['#137909','#ff0707'],edgecolor=sns.color_palette('dark',n_colors=1))
sns.countplot(x='sysBP',hue='TenYearCHD',data=df,palette=['#137909','#ff0707'])
sns.jointplot("heartRate", "totChol", data=df, kind='reg');

