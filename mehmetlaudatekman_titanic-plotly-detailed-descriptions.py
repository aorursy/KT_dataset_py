# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_PassId = test_df["PassengerId"]

train_df_len = len(train_df)
train_df.info()
train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
train_df.head()
train_df.info()
test_df.info()
train_df.isnull().sum()
survived_feature = train_df.Survived.value_counts()

colors = ["rgba(122,231,23,0.7)","rgba(80,21,235,0.8)"]

trace1 = go.Bar(x=survived_feature.index

               ,y=survived_feature.values

               ,marker=dict(color=colors)

               ,text = survived_feature.index

               ,name= str(survived_feature.index))



layout = go.Layout(title="Countplot - Survived",

                   xaxis=dict(title="Survived")

                  ,yaxis=dict(title="How Many People"))



figure = go.Figure(data=trace1,layout=layout)

iplot(figure)
pclass_feature = train_df.Pclass.value_counts()

colors = ["rgba(212,76,23,0.7)","rgba(0,213,78,0.8)","rgba(0,124,34,0.8)"]

trace1 = go.Bar(x=pclass_feature.index

               ,y=pclass_feature.values

               ,marker=dict(color=colors)

               ,text = pclass_feature.index

               ,name= str(pclass_feature.index))



layout = go.Layout(title="Countplot - Pclass",

                   xaxis=dict(title="Pclass")

                  ,yaxis=dict(title="How Many People"))



figure = go.Figure(data=trace1,layout=layout)

iplot(figure)
sex_feature = train_df.Sex.value_counts()

colors = ["rgba(52,212,184,0.7)","rgba(231,212,21,0.7)"]



trace1 = go.Bar(x=sex_feature.index

               ,y=sex_feature.values

               ,marker=dict(color=colors))



layout = go.Layout(title="Countplot - Sex"

                  ,xaxis=dict(title="Gender")

                  ,yaxis=dict(title="How Many People"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
embarked_feature = train_df.Embarked.value_counts()

colors = ["rgba(12,124,184,0.9)","rgba(124,213,34,0.9)","rgba(142,184,202,0.9)"]

trace1 = go.Bar(x=embarked_feature.index

               ,y=embarked_feature.values

               ,marker=dict(color=colors))



layout = go.Layout(title="Countplot - Embarked"

                  ,xaxis=dict(title="Ports")

                  ,yaxis=dict(title="How Many People"))



figure = go.Figure(data=trace1,layout=layout)

iplot(figure)

trace1 = go.Histogram(x=train_df.Age)



layout = go.Layout(title='Histogram Plot - Age'

                  ,xaxis=dict(title='Age')

                  ,yaxis=dict(title='Count'))



figure = go.Figure(data=trace1,layout=layout)

iplot(figure)
trace1 = go.Histogram(x=train_df.SibSp

                     ,marker=dict(color="rgba(255,0,0,0.7)"))



layout = go.Layout(title='Histogram Plot - SibSp'

                  ,xaxis=dict(title="SibSp")

                  ,yaxis=dict(title="Count"))



figure = go.Figure(data=trace1,layout=layout)

iplot(figure)
trace1 = go.Histogram(x=train_df.Parch

                     ,marker=dict(color="rgba(57,232,0,0.7)"))



layout = go.Layout(title='Histogram Plot - ParCh'

                  ,xaxis=dict(title="ParCh")

                  ,yaxis=dict(title="Count"))



figure = go.Figure(data=trace1,layout=layout)

iplot(figure)
def outlier_detector(df,features):

    rows = len(df)

    

    drop_index_list=[]

    final_index_list = []

    

    for ftr in features:

        

        Q1 = df.describe()[ftr]["25%"]

        Q3 = df.describe()[ftr]["75%"]

        IQR = Q3-Q1

        STEP = IQR*1.5

        drop_index = df[(df[ftr]<Q1-STEP) | (df[ftr]>Q3+STEP)].index.values

        

        for i in drop_index:

            drop_index_list.append(i)

    

    for i in drop_index_list:

        if i<891:

            drop_index_list.remove(i)

            if i in drop_index_list:

                drop_index_list.remove(i)

                if i in drop_index_list:

                    final_index_list.append(i)



        

    rows_end = len(final_index_list)  

    print(f"{rows_end} rows affected")

    return final_index_list

        

    
drop_indexes = outlier_detector(train_df,["Age","SibSp","Parch","Fare"])

print(drop_indexes)
train_df = train_df.drop(drop_indexes,axis=0).reset_index(drop=True)

print("Outliers dropped")
import missingno as msn





msn.matrix(train_df)

plt.show()
msn.bar(train_df)

plt.show()
train_df[(train_df["Embarked"] != "S") & (train_df["Embarked"] != "Q") & (train_df["Embarked"] != "C") ]
common_people = train_df[(train_df.Survived == 1) & (train_df.Pclass == 1) & (train_df.Sex == "female") & (train_df.SibSp == 0) & (train_df.Parch == 0)]
common_people.Embarked.value_counts()
train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df.isnull().sum()
survived = train_df.groupby(by="Survived").median()["Age"]

survived
pclass = train_df.groupby(by="Pclass").median()["Age"]

pclass
sex = train_df.groupby(by="Sex").median()["Age"]

sex
embarked = train_df.groupby(by="Embarked").median()["Age"]

embarked
age_fillna = []

nan_check  = train_df.copy()

nan_check.drop(["Name","PassengerId","SibSp","Parch","Ticket","Fare","Cabin"],axis=1,inplace=True)

nan_check.head()
index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)

for passanger in nan_check.values:

    

    if np.isnan(passanger[3]):

        MedianSurvived = survived[survived.index == passanger[0]].values

    

        MedianPclass = pclass[pclass.index == passanger[1]].values

    

        MedianSex = sex[sex.index == passanger[2]].values

    

        MedianPort = embarked[embarked.index == passanger[4]].values

    

        MedianList = [MedianSurvived,MedianPclass,MedianSex,MedianPort]

    

        pred = np.median(MedianList)

    

        age_fillna.append(pred)

    

    else:

        age_fillna.append(passanger[3])

        

    



print(len(age_fillna))

print(len(train_df))
train_df["Age"] = age_fillna
train_df["Fare"].fillna(train_df["Fare"].median(),inplace=True)
train_df.isnull().sum()
corr = train_df.corr()

corr.drop("PassengerId",axis=1,inplace=True)

corr.drop("PassengerId",axis=0,inplace=True)

fig,ax = plt.subplots(figsize=(10,10))

sns.heatmap(corr,annot=True,linewidths=1.5)

plt.show()

sibsp_survived = train_df.groupby(by="SibSp").mean()["Survived"]

sibsp_survived
colors = ["rgb(123,43,211)"

         ,"rgb(21,211,32)"

         ,"rgb(12,176,90)"

         ,"rgb(48,65,198)"

         ,"rgb(32,87,132)"]



trace1 = go.Bar(x=sibsp_survived.index

               ,y=sibsp_survived.values

               ,marker=dict(color=colors))



layout = go.Layout(title="Survive Possibility by SibSp"

                  ,xaxis=dict(title="SibSp Number")

                  ,yaxis=dict(title="Survive Possibilty"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
parch_survived = train_df.groupby(by="Parch").mean()["Survived"]

parch_survived
import random as rn

def random_color_creator(color_number):

    color_list = []

    for i in range(color_number):

        r = str(rn.randrange(10,255))

        g = str(rn.randrange(10,255))

        b = str(rn.randrange(10,255))

        color = "rgba("+r+","+g+","+b+"," + "0.8"+ ")"

        color_list.append(color)

    return color_list

trace = go.Bar(x=parch_survived.index

              ,y=parch_survived.values

              ,marker=dict(color=random_color_creator(7)))



layout = go.Layout(title="Parch - Survive Possibility"

                  ,xaxis=dict(title="Parch Number")

                  ,yaxis=dict(title="Survived"))



figure = go.Figure(data=trace,layout=layout)



iplot(figure)
pclass_survived = train_df.groupby(by="Pclass").mean()["Survived"]

pclass_survived
trace1 = go.Bar(x=pclass_survived.index

               ,y=pclass_survived.values

               ,marker=dict(color=random_color_creator(3)))



layout = go.Layout(title="Pclass - Survive Possibility"

                  ,xaxis=dict(title="Pclass")

                  ,yaxis=dict(title="Survive Possibility"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
train_df["Name"].head()
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in train_df["Name"]]
train_df.head()
title = train_df["Title"].value_counts()

trace = go.Bar(x=title.index

              ,y=title.values

              ,marker=dict(color=random_color_creator(len(title))))



layout = go.Layout(title="Countplot of Title Feature"

                  ,xaxis=dict(title="Title")

                  ,yaxis=dict(title="Count"))



figure = go.Figure(data=trace,layout=layout)



iplot(figure)
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")

train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]

train_df["Title"].head(20)
train_df.drop("Name",axis=1,inplace=True)

train_df = pd.get_dummies(train_df,columns=["Title"])

train_df.head()
train_df["FamilySize"] = train_df["Parch"] + train_df["SibSp"] + 1 

train_df.head()
familysize = train_df["FamilySize"].value_counts()



trace = go.Bar(x=familysize.index

              ,y=familysize.values

              ,marker=dict(color=random_color_creator(len(familysize))))



layout = go.Layout(title="Countplot of Family Size"

                  ,xaxis=dict(title="Family Size")

                  ,yaxis=dict(title="Count"))



figure = go.Figure(data=trace,layout=layout)



iplot(figure)
train_df["FamilySize"] = [1 if each==1 else 0 for each in train_df["FamilySize"]]
train_df.head()
train_df = pd.get_dummies(train_df,columns=["FamilySize"])

train_df.head()
train_df.drop("SibSp",axis=1,inplace=True)

train_df.drop("Parch",axis=1,inplace=True)
train_df["Sex"] = [1 if i=="female" else 0 for i in train_df.Sex]

train_df["Sex"].head(10)
train_df = pd.get_dummies(train_df,columns=["Sex"])
train_df.head()

train_df["Embarked"] = train_df["Embarked"].astype("category")

train_df = pd.get_dummies(train_df,columns=["Embarked"])

train_df.head()
train_df["Pclass"] = train_df["Pclass"].astype("category")

train_df = pd.get_dummies(train_df,columns=["Pclass"])

train_df.head()
train_df.drop(["Ticket","Cabin","PassengerId"],axis=1,inplace=True)
train_df.head()
train_df[882:].isnull().sum()
test_df = train_df[882:].drop("Survived",axis=1)

test_df.head()
test_df.info()
from sklearn.model_selection import train_test_split

train_df = train_df[:882]

train_df.info()

x = train_df.drop("Survived",axis=1)

y = train_df.Survived



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
print("Len of x_train",len(x_train))

print("Len of x_test",len(x_test))

print("Len of y_train",len(y_train))

print("Len of y_test",len(y_test))

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV
random_state = 1

ML=[DecisionTreeClassifier(random_state = random_state),

    SVC(random_state = random_state),

    RandomForestClassifier(random_state = random_state),

    LogisticRegression(random_state = random_state),

    KNeighborsClassifier()]


dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(ML)):

    clf = GridSearchCV(ML[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose=1)

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
best_estimators[2]
rfc = RandomForestClassifier(bootstrap=False,max_features=10,min_samples_leaf=10,n_estimators=300,random_state=1)

rfc.fit(x_train,y_train)
results = pd.Series(rfc.predict(test_df)).astype(int)

result_csv = pd.concat([test_PassId,results],axis=1)



result_csv.head()

result_csv.to_csv("results.csv",index=False)