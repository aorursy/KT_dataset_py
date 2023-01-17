# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # Data manipulation
import pandas as pd # Data manipulation

import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Data visualization


import random # generate random numbers
import re # match regular expression
from collections import Counter
sns.set_style('whitegrid') # Set plots' style 
%matplotlib inline

# machine learning toolkit
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
df_titanic=pd.concat([df_train,df_test]).reset_index(drop=True)
df_titanic.info()

df_titanic.head()
df_titanic.describe()
fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].set_title("Survival rate vs Sexe")
sns.barplot(x="Sex",y="Survived",data=df_titanic,ax=axs[0])
axs[1].set_title("Survival rate vs Sexe vs Age")
sns.swarmplot(x="Sex",y="Age",data=df_titanic,hue="Survived",ax=axs[1])
df_child=df_titanic[df_titanic["Age"]<=16]
ax=plt.axes()
ax.set_title('Survival Rate vs Sexe')
sns.barplot(x="Sex",y="Survived",data=df_child)
sns.factorplot(x="Pclass",y="Survived",data=df_titanic)
df_age=df_titanic["Age"].dropna()
mean_age=int(df_titanic["Age"].dropna().mean())
std_age=int(df_titanic["Age"].dropna().std())

print("Mean Age :{}".format(mean_age))
print("Age standard deviation: {}".format(std_age))
df_age=df_titanic["Age"].dropna()
mean_age=int(df_titanic["Age"].dropna().mean())
std_age=int(df_titanic["Age"].dropna().std())
df_titanic["Age"]=df_titanic["Age"].apply(lambda x: np.random.randint(int(mean_age-std_age),int(mean_age+std_age)) if pd.isnull(x) else x)
fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].set_title("Age Distribution before imputing missing values")
sns.distplot(df_age,ax=axs[0])
axs[1].set_title("Age Distribution after imputing missing values")
sns.distplot(df_titanic["Age"],ax=axs[1])
f, ax = plt.subplots(1, 1)
sns.distplot(df_titanic[df_titanic["Survived"]==1]["Age"],label='Survivor',hist=False)
sns.distplot(df_titanic[df_titanic["Survived"]==0]["Age"],label='Not Survivor',hist=False)
ax.legend()
sns.factorplot(x="Embarked",y="Fare",data=df_titanic)
print(df_titanic[pd.isnull(df_titanic["Embarked"])])
df_titanic["Embarked"]=df_titanic["Embarked"].fillna("C")
print(df_titanic[pd.isnull(df_titanic["Fare"])])
mean=df_titanic[df_titanic["Embarked"]=='S']["Fare"].mean()
std=df_titanic[df_titanic["Embarked"]=='S']["Fare"].std()
df_titanic["Fare"]=df_titanic["Fare"].fillna(random.randint(int(mean-std),int(mean+std)))
df_titanic.loc[df_titanic["Age"]<=16,"Sex"]="Child"
df_titanic.loc[(df_titanic["Age"]>16) & (df_titanic["Parch"]>0) & (df_titanic["Sex"]=="female"),"Sex"]="Mother"
sns.barplot(x="Sex",y="Survived",data=df_titanic) 
fig, axs = plt.subplots(1,3,figsize=(15,5))
axs[0].set_title('Survival Rate vs Parch')
sns.barplot(x="Parch",y="Survived",data=df_titanic,ax=axs[0])
axs[1].set_title('Survival Rate vs SibSp')
sns.barplot(x="SibSp",y="Survived",data=df_titanic,ax=axs[1])
df_family=df_titanic["Parch"]+df_titanic["SibSp"]
axs[2].set_title('Survival Rate vs Parch + SibSp')
sns.barplot(x=df_family,y=df_titanic["Survived"],ax=axs[2])
df_titanic["Parch"].value_counts()
df_titanic["SibSp"].value_counts()
df_titanic["family"]=df_titanic["Parch"]+df_titanic["SibSp"]
df_titanic.loc[(df_titanic["family"]>=1) & (df_titanic["family"]<=3) ,"family"]=1
df_titanic.loc[df_titanic["family"] >3 | (df_titanic["family"]==0) ,"family"]=0
ax=plt.axes()
ax.set_title('Survival rate vs family size')
sns.barplot(x="family",y="Survived",data=df_titanic)
df_name=pd.DataFrame(df_titanic["Name"].str.extract('([A-Za-z]+\.)'))
print(df_name["Name"].value_counts())
VIP=["Master.","Rev.","Dr.","Col.","Major.","Jonkheer.","Dona.","Capt.","Don.","Sir.","Lady.","Countess."]
df_name.loc[df_name["Name"].isin(VIP),"Name"]="VIP"
df_name.loc[df_name["Name"].isin(["Mlle.","Ms."]),"Name"]="Miss."
df_name.loc[df_name["Name"]=="Mme.","Name"]="Mrs."
df_titanic["Name"]=df_name["Name"]
ax=plt.axes()
ax.set_title("Survival rate vs title")
sns.barplot(x="Name",y="Survived",data=df_titanic)
df_cabin=pd.DataFrame(df_titanic[["Cabin","Pclass","Fare"]].dropna())
df_cabin["Cabin"]=df_cabin["Cabin"].astype(str).str[0]
print(df_cabin["Cabin"].value_counts())
sns.factorplot(x="Cabin",y="Fare",data=df_cabin)
sns.factorplot(x="Cabin",y="Pclass",data=df_cabin)
df_train=df_titanic.loc[0:890,]
df_test=df_titanic.loc[891:1308,]
X=df_train.drop(["PassengerId","Cabin","Ticket","Survived","SibSp","Parch"],axis=1)
y=df_train["Survived"]
Submission_set=df_test.drop(["PassengerId","Cabin","Ticket","Survived","SibSp","Parch"],axis=1)
X=pd.get_dummies(X)
Submission_set=pd.get_dummies(Submission_set)
X.info()
X.describe()
X=scale(X)
Submission_set=scale(Submission_set)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
Neighbors=KNeighborsClassifier()
n_neighbors = np.arange(1,50)
param_grid = {'n_neighbors': n_neighbors, 
             'metric':["cityblock","euclidean"]}
Neighbors_cv = GridSearchCV(Neighbors,param_grid, cv=5)
Neighbors_cv.fit(X_train,y_train)

print("Tuned k-NN classifier Parameters: {}".format(Neighbors_cv.best_params_)) 
print("Best score is {}".format(Neighbors_cv.best_score_))

y_pred=Neighbors_cv.predict(X_test)
print("Confusion Matrix :")
print(confusion_matrix(y_pred,y_test))
print("Classification report :")
print(classification_report(y_pred,y_test))
Sub_pred=Neighbors_cv.predict(Submission_set).astype(int)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Sub_pred
    })
submission.to_csv('titanic.csv', index=False)