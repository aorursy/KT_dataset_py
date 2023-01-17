import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plote
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
combine=[train,test]
train.head()
features=train.columns.values
features=np.delete(features,np.where(['Survived']))
target=['Survived']
features
f,ax=plote.subplots(figsize=(18,18))
sns.heatmap(train.corr(),annot=True,linewidths=.2,fmt='.1f',ax=ax)
train.describe()
train.shape

train.describe()
train.describe(include=['O'])

train.describe(include=['O'])
train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=True)
train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=True)
train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived')
train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)

train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived')
g=sns.FacetGrid(train,col="Survived")
g.map(sns.distplot,'Age',bins=20)
g=sns.FacetGrid(train,col="Pclass")
g.map(plote.hist,"Age",bins=20)
g=sns.FacetGrid(train,row="Pclass",col="Survived")
g.map(plote.hist,"Age",bins=20)
sns.barplot("Pclass","Survived",data=train)
g=sns.FacetGrid(train,row="Embarked",size=2.2, aspect=1.6)
g.map(sns.pointplot,'Pclass','Survived',"Sex",palette='deep')
g.add_legend()
g=sns.FacetGrid(train,row="Embarked",col='Survived')
g.map(sns.barplot,'Sex','Fare',palette='deep')
g.add_legend()
g=sns.FacetGrid(train,row="Embarked",col="Pclass")
g.map(sns.barplot,"Survived","Fare")
g=sns.FacetGrid(train,col="Sex")
g.map(sns.barplot,"Embarked","Survived")
train=train.drop(["PassengerId","Cabin","Ticket"],axis=1)
test=test.drop(["PassengerId","Cabin","Ticket"],axis=1)
combine=[train,test]
freq=train['Embarked'].dropna().mode()[0]
freq
for data in combine:
    data['Embarked']=data['Embarked'].fillna(freq)
train.describe(include=['O'])
lb=LabelEncoder()
for data in combine:
    data["Sex"]=lb.fit_transform(data["Sex"])
    data["Embarked"]=lb.fit_transform(data["Embarked"])
    #data['Sex']=data['Sex'].map({'male': 0 , 'female': 1}).astype(int)
    #data['Embarked']=data['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)
    
grid = sns.FacetGrid(train, row='Pclass',col="Sex" ,size=2.2, aspect=1.6)
grid.map(plote.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
age_median=np.zeros((2,3))
for data in combine:
    for i in range(0,2):
        for j in range(0,3):
            g_df=data[(data["Sex"]==i) & (data['Pclass']==j+1)]["Age"].dropna()
            g_med=g_df.median()
            g_med=int(g_med/0.5+0.5)*0.5 #rounding to nearest 0.5
            age_median[i,j]=g_med
            
    for i in range(0,2):
        for j in range(0,3):
            data.loc[(data["Age"].isnull()) & (data["Sex"]==i) & (data['Pclass']==j+1),"Age"]=age_median[i,j]
train["Age"]=train["Age"].astype(int)
test["Age"]=test["Age"].astype(int)
train["Age_Groups"]=pd.cut(train["Age"],5)
train[["Age_Groups","Survived"]].groupby(['Age_Groups'],as_index=False).mean().sort_values(by="Survived",ascending=False)
for data in combine:
    data.loc[(data["Age"]<=16),"Age"]=0
    data.loc[(data["Age"]>16) & (data["Age"]<=32),"Age"]=1
    data.loc[(data["Age"]>32) & (data["Age"]<=48),"Age"]=2
    data.loc[(data["Age"]>48) & (data["Age"]<=64),"Age"]=3
    data.loc[(data["Age"]>64) & (data["Age"]<=80),"Age"]=4
    
train.head()
train=train.drop(["Age_Groups"],axis=1)
combine=[train,test]
test.loc[test["Fare"].isnull(),"Fare"]=test["Fare"].dropna().median()
train["Fare_Band"]=pd.qcut(train["Fare"],4)
train[["Fare_Band","Survived"]].groupby(["Fare_Band"],as_index=False).mean().sort_values(by="Survived",ascending=True)
for data in combine:
    data.loc[(data["Fare"]<=7.91),"Fare"]=0
    data.loc[(data["Fare"]>7.91) & (data["Fare"]<=14.454),"Fare"]=1
    data.loc[(data["Fare"]>14.454) & (data["Fare"]<=31.0),"Fare"]=2
    data.loc[(data["Fare"]>31.0),"Fare"]=3
    
train.head(5)
train["Fare"]=train["Fare"].astype(int)
test["Fare"]=test["Fare"].astype(int)
train=train.drop(["Fare_Band"],axis=1)
combine=[train,test]
for data in combine:
    data["Title"]=data["Name"].str.extract("\s([A-Za-z]+\.)\s",expand=False)
pd.crosstab(train["Title"],train["Sex"])

pd.crosstab(test["Title"],test["Sex"])
for data in combine:
    data["Title"]=data["Title"].replace(["Capt.","Col.","Don.","Dr.","Jonkheer.","Lady.","Countess.","Major.","Rev.","Sir.","Dona."],"Others")
    data["Title"]=data["Title"].replace(["Mlle."],"Miss.")
    data["Title"]=data["Title"].replace(["Mme."],"Mrs.")
    data["Title"]=data["Title"].replace(["Ms."],"Miss.")
    
train[["Title","Survived"]].groupby(["Title"],as_index=False).mean().sort_values(by="Survived",ascending=False)
for data in combine:
    data["Title"]=data["Title"].map({"Mr.":1,"Miss.":2,"Mrs.":3,"Master.":4,"Others":5}).astype(int)
train=train.drop(["Name"],axis=1)
test=test.drop(["Name"],axis=1)
combine=[train,test]
for data in combine:
    data["family_members"]=data["Parch"]+data["SibSp"]
train[["family_members","Survived"]].groupby("family_members",as_index=True).mean().sort_values(by="Survived",ascending=False)
for data in combine:
    data.loc[data["family_members"]>0,'isalone']=0
    data.loc[data["family_members"]==0,'isalone']=1
    
for data in combine:
    data["isalone"]=data["isalone"].astype(int)
train[["isalone","Survived"]].groupby(["isalone"],as_index=True).mean().sort_values(by="Survived",ascending=False)
train=train.drop(["family_members","Parch","SibSp"],axis=1)
test=test.drop(["family_members","Parch","SibSp"],axis=1)
combine=[train,test]
train.head()
combine=[train,test]
for data in combine:
    data["age*class"]=data["Age"] * data["Pclass"]
x_train=train.drop(["Survived"],axis=1)
y_train=train["Survived"]
x_test=test
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
pred=logreg.predict(x_test)
conf_score=round(logreg.score(x_train,y_train)*100,2)
conf_score
coeff=pd.DataFrame({"Feature": train.columns.delete(0)})
coeff["Correlation"]=logreg.coef_[0]
coeff.sort_values(by="Correlation",ascending=True)
svc=SVC()
svc.fit(x_train,y_train)
svc_pred=svc.predict(x_test)
svc_conf=round(svc.score(x_train,y_train)*100,2)
svc_conf
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
Y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
Y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
Y_pred = random_forest.predict(test)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest

xx_train,xx_cv,yy_train,yy_cv=train_test_split(x_train,y_train,test_size=0.20,random_state=7)

xx_train.head()
xg=XGBClassifier(n_estimators=300,learning_rate=0.5)
xg.fit(xx_train,yy_train,early_stopping_rounds=10,eval_set=[(xx_cv,yy_cv)])
y_pred=xg.predict(xx_cv)
acc_xgboost=round(xg.score(x_train,y_train)*100,2)
acc_xgboost
xg.feature_importances_
xg_importance=pd.DataFrame({"Feature": train.columns.delete(0)})
xg_importance["Correlation"]=xg.feature_importances_
xg_importance.sort_values(by="Correlation",ascending=True)
f1_score(yy_cv,y_pred)
