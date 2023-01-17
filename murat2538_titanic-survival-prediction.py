# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from collections import Counter

import warnings
warnings.filterwarnings("ignore")


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train=pd.read_csv("/kaggle/input/titanic/train.csv")
data_test=pd.read_csv("/kaggle/input/titanic/test.csv")


train_df=data_train.copy()
test_df=data_test.copy()

train_df.columns
train_df.head()
train_df.describe()
train_df.info()
def bar_plot(variable):
    """
    input: Variable ex:"Sex"
    output: barplot + value_count
    """
    #get feature
    var=train_df[variable]
    #number of categorical varriable
    var_value=var.value_counts()
    
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(var_value.index,var_value)
    plt.xticks(var_value.index,var_value.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}:\n{}".format(variable,var_value))
category1=["Survived","Sex","Pclass","Embarked","SibSp","Parch"]

for c in category1:
    bar_plot(c)
    
def plot_hist(variable):
    plt.figure(figsize=(9,3))
    plt.hist(train_df[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distrubition with hist".format(variable))
    plt.show()
numericVar=["Fare","Age","PassengerId"]

for n in numericVar:
    plot_hist(n)
# Pclass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived",ascending=False)
# Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived",ascending=False)
# SibSb vs Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived",ascending=False)
# SibSb vs Survived
train_df[["Parch","Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived",ascending=False)
def detect_outliers(df,features):
    outlier_indices=[]
    
    for c in features:
        #1st quartile
        Q1=np.percentile(df[c],25)
        #3st quartile
        Q3=np.percentile(df[c],75)
        #IQR
        IQR=Q3-Q1
        #outlier step
        outlier_step=IQR*1.5
        #detect outlier and their indices
        
        outlier_list_col=df[(df[c] < Q1-outlier_step) | (df[c] > Q3+outlier_step )].index
        #store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    
    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)
    
    return multiple_outliers
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
#drop Outliers
train_df=train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis=0).reset_index(drop=True)
train_df_len=len(train_df)
train_df=pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
#Embarked
train_df[train_df.Embarked.isnull()]
train_df.boxplot(column="Fare",by="Embarked")
plt.show()
train_df["Embarked"]=train_df["Embarked"].fillna("C")
train_df[train_df.Embarked.isnull()]
#Fare
train_df[train_df["Fare"].isnull()]
np.mean(train_df[train_df["Pclass"]==3]["Fare"])
train_df["Fare"]=train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))
train_df[train_df["Fare"].isnull()]
list1=["SibSp","Parch","Age","Fare","Survived"]

sns.heatmap(train_df[list1].corr(),annot=True, fmt=".2f");
g = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability");
g = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability");
g = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar",size=6)
g.set_ylabels("Survived Probability");
g = sns.FacetGrid(train_df,col="Survived")
g.map(sns.distplot,"Age",bins=25);
g=sns.FacetGrid(train_df,col="Survived",row="Pclass")
g.add_legend()
g.map(plt.hist,"Age",bins=25);
g=sns.FacetGrid(train_df,row="Embarked")
g.add_legend()
g.map(sns.pointplot,"Pclass","Survived","Sex");
g=sns.FacetGrid(train_df,row="Embarked",col="Survived",size=2.3)
g.add_legend()
g.map(sns.barplot,"Sex","Fare");
train_df[train_df["Age"].isnull()]
sns.factorplot(x="Sex",y="Age",data=train_df,kind="box",size=6);
sns.factorplot(x="Sex",y="Age",hue="Pclass",data=train_df,kind="box",size=6);
sns.factorplot(x="Parch",y="Age",data=train_df,kind="box",size=6);
sns.factorplot(x="SibSp",y="Age",data=train_df,kind="box",size=6);
train_df["Sex"]=[1 if i =="male" else 0 for i in train_df["Sex"]]
sns.heatmap(train_df[["Age","SibSp","Parch","Sex","Pclass"]].corr(),annot=True);
index_nane_age=list(train_df["Age"][train_df["Age"].isnull()].index)

for i in index_nane_age:
    age_pred=train_df["Age"][((train_df["SibSp"]==train_df.iloc[i]["SibSp"]) &(train_df["Parch"]==train_df.iloc[i]["Parch"]) & (train_df["Pclass"]==train_df.iloc[i]["Pclass"]))].median()
    age_med=train_df.Age.median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i]=age_pred
    else:
        train_df["Age"].iloc[i]=age_med
        
train_df[train_df["Age"].isnull()]
train_df["Name"].head(10)
name=train_df["Name"]
train_df["Title"]=[i.split(".")[0].split(",")[-1].strip() for i in name]
train_df["Title"].head(10)
sns.countplot(x="Title",data=train_df)
plt.xticks(rotation=60);
#convert to categorical
train_df["Title"]=train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rav","Sir","Jonkheer","Dona"],"other")
train_df["Title"]=[0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mile" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(10)
sns.countplot(x="Title",data=train_df)
plt.xticks(rotation=60);
g=sns.factorplot(x="Title",y="Survived",data=train_df,kind="bar");
g.set_xticklabels(["Master","Mrs","Mr","other"])
g.set_ylabels("Survival Probability");
train_df.drop(labels=["Name"],axis=1,inplace=True)
train_df.head()
train_df=pd.get_dummies(train_df,columns=["Title"])
train_df.head()
train_df.head()
train_df["Fsize"]=train_df["SibSp"]+train_df["Parch"] + 1
g=sns.factorplot(x="Fsize",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival ");
train_df["Family_size"]=[1 if i < 5 else 0 for i in train_df["Fsize"]]
train_df.head(10)
sns.countplot(x="Family_size",data=train_df);
g=sns.factorplot(x="Family_size",y="Survived",data=train_df,kind="bar")
g.set_ylabels("Survival ");
train_df=pd.get_dummies(train_df,columns=["Family_size"])
train_df.head()
train_df["Embarked"].head()
sns.countplot(x="Embarked",data=train_df);
train_df=pd.get_dummies(train_df,columns=["Embarked"])
train_df.head()
train_df["Ticket"].head()
tickets=[]
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"]=tickets
train_df["Ticket"].head()
train_df=pd.get_dummies(train_df,columns=["Ticket"],prefix="T")
train_df.head()
train_df["Pclass"].head()
sns.countplot(x="Pclass",data=train_df);
train_df["Pclass"]=train_df["Pclass"].astype("category")
train_df=pd.get_dummies(train_df,columns=["Pclass"])
train_df.head()
train_df["Sex"]=train_df["Sex"].astype("category")
train_df=pd.get_dummies(train_df,columns=["Sex"])
train_df.head()
train_df.drop(labels=["PassengerId","Cabin"],axis=1,inplace=True)
train_df.head()
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
train_df_len
test=train_df[train_df_len:]
test.drop(labels=["Survived"],axis=1,inplace=True)
test.head()
train=train_df[:train_df_len]
X_train=train.drop(labels="Survived",axis=1)
y_train=train["Survived"]

X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.15,random_state=42)

print("x_train",len(X_train))
print("x_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))

lg=LogisticRegression().fit(X_train,y_train)

acc_log_train=round(accuracy_score(lg.predict(X_train),y_train)*100,2)
acc_log_test=round(accuracy_score(lg.predict(X_test),y_test)*100,2)
print("Train Accuracy: %",acc_log_train)
print("Test Accuracy : %",acc_log_test)
def model_predict(X_train,y_train,X_test,y_test,alg):#df:datasaet,y:bağımlı değişken,alg:algoritma
    if alg.__name__== "CatBoostClassifier":
        model=alg().fit(X_train,y_train,verbose=False)
        
    elif alg.__name__=="KNeighborsClassifier":
        model=alg(n_neighbors=4).fit(X_train,y_train)
    else:
        model=alg().fit(X_train,y_train)
    y_pred=model.predict(X_test)
    model_name=alg.__name__
    accuracy=accuracy_score(y_test,y_pred)
    print("accuracy for ",model_name,":",accuracy)
    #return accuracy
model_predict(X_train,y_train,X_test,y_test,RandomForestClassifier)
model_list=[LogisticRegression,RandomForestClassifier,KNeighborsClassifier,SVC,GradientBoostingClassifier,LGBMClassifier,
           XGBClassifier,CatBoostClassifier]

for i in model_list:
    model_predict(X_train,y_train,X_test,y_test,i)
def model_tuned(X_train,y_train,X_test,y_test,alg,alg_params,GridSearchCV,cv):    
    if alg.__name__== "CatBoostClassifier":
        model=alg(verbose=False)
    else:
        model=alg()
    model_cv=GridSearchCV(model,alg_params,cv=cv,n_jobs=-1,
                          verbose=2).fit(X_train,y_train)
    model_name=alg.__name__
    print("best params for ",model_name,":")
    return model_cv.best_params_
# Random Forest Tuning
forest_params={"max_features":[1,3,10],
               "n_estimators":[100,300],
               "min_samples_split":[2,3,10],
               "max_features":[1,3,10],
               "bootstrap":[False],
               "criterion":["gini"]}
model_tuned(X_train,y_train,X_test,y_test,RandomForestClassifier,forest_params,GridSearchCV,cv=10)
rforest_tuned=RandomForestClassifier(max_features=10,
                                     min_samples_split=10,
                                     n_estimators=300,
                                     criterion = 'gini',
                                     bootstrap=False).fit(X_train,y_train)
y_pred=rforest_tuned.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
importance=pd.Series(rforest_tuned.feature_importances_,
                     index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10,10))
sns.barplot(x=importance,y=importance.index)
plt.xlabel("Variable İmportance score")
plt.ylabel("Variables")
plt.title("Variable importance level")
plt.gca().legend_=None
#Decission Tree Calssifier

dtree_params={"min_samples_split":range(10,500,20),
              "max_depth":range(1,20,2)}
model_tuned(X_train,y_train,X_test,y_test,DecisionTreeClassifier,dtree_params,GridSearchCV,cv=10)
dtree_tuned=DecisionTreeClassifier(max_depth=5,min_samples_split=10).fit(X_train,y_train)
y_pred=dtree_tuned.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
#SVM Classifier
svm_params={"C":[1,10,50,200,300,1000],
            "kernel":["rbf"],
            "gamma":[0.001,0.01,0.1,1]}
model_tuned(X_train,y_train,X_test,y_test,SVC,svm_params,GridSearchCV,cv=10)
svm_tuned=SVC(kernel="rbf",gamma=0.001,C=1000).fit(X_train,y_train)
y_pred=svm_tuned.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
#KNeighbors Classifier
knn_params={"n_neighbors":np.arange(1,50),
            "weights":["uniform","distance"],
            "metric":["euclidean","manhattan"]}
model_tuned(X_train,y_train,X_test,y_test,KNeighborsClassifier,knn_params,GridSearchCV,cv=10)
knn_tuned=KNeighborsClassifier(metric="manhattan",n_neighbors=5,weights="uniform").fit(X_train,y_train)
y_pred=knn_tuned.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
#GradientBoostingClassifier 
gbm_params={"n_estimators":[100,250,300,350,400,],
            "max_depth":[2,3,5,7,10],
            "learning_rate":[0.1,0.095,0.01,0.03,0.05]}

model_tuned(X_train,y_train,X_test,y_test,GradientBoostingClassifier,gbm_params,GridSearchCV,cv=10)
gbm_tuned=GradientBoostingClassifier(n_estimators=250,
                                     learning_rate=0.03,
                                     max_depth=5).fit(X_train,y_train)
y_pred=gbm_tuned.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
#CatBoostClassifier
cb_params={"iterations":[100,500,1000],
            "depth":[4,5,8],
            "learning_rate":[0.1,0.01,0.03]}

model_tuned(X_train,y_train,X_test,y_test,CatBoostClassifier,cb_params,GridSearchCV,cv=10)
cb_tuned=CatBoostClassifier(iterations=100,
                            depth=4,
                            learning_rate=0.03).fit(X_train,y_train,verbose=False)

y_pred=cb_tuned.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
svm_tuned=SVC(kernel="rbf",gamma=0.001,C=1000).fit(X_train,y_train)
y_pred=svm_tuned.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
test_survived=pd.Series(gbm_tuned.predict(test),name="Survived").astype(int)
passenger_id=test_df["PassengerId"]
results=pd.concat([passenger_id,test_survived],axis=1)
results.to_csv("titanic.csv",index=False)