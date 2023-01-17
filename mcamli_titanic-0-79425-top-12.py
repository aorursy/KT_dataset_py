import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.head(10)
train.info()
train["Cabin"].value_counts()
train.drop("PassengerId",axis=1,inplace=True)
train.describe()
x = train["Age"]

sns.distplot(x)
print(train["Age"].median())

print(train["Age"].mean())

print(train["Age"].mode()[0])
median = train["Age"].median()

train["Age"].fillna(median,inplace = True)
sns.countplot(train["Embarked"])
train["Embarked"].fillna("S",inplace = True)
train.info()
#correlation map

f,ax=plt.subplots(figsize=(25, 25))

sns.heatmap(train.corr(), annot=True, linewidths=.4, fmt= '.1f',ax=ax)

plt.show()
corr_matrix = train.corr()

corr_matrix["Survived"].sort_values(ascending = False)
train.loc[:,["Name","Ticket"]]
train["Honorific"] = train["Name"].str.split(",",expand = True).iloc[:,1].str.split(".",expand = True).iloc[:,0].str.strip()

train["Honorific"].value_counts()
train = train.assign(Mr = (train["Honorific"] == "Mr").astype(int))

train = train.assign(Miss = (train["Honorific"] == "Miss").astype(int))

train = train.assign(Mrs = (train["Honorific"] == "Mrs").astype(int))

train = train.assign(Master = (train["Honorific"] == "Master").astype(int))
train.drop("Honorific",axis=1,inplace=True)
train["Name"] = train["Name"].str.split(",",expand = True)[0]



family = train.loc[:,["Name","Ticket"]]

family["new"] = family["Name"] + family["Ticket"]



train["new"] = train["Name"] + train["Ticket"]

family_unique = family["new"].unique().tolist()

family_id = [i for i in range(1,len(family_unique)+1)]



merge_df = pd.DataFrame(data={"new":family_unique,"FamilyID":family_id})

    

train = train.join(merge_df.set_index("new"), on='new')

train.drop("new",axis=1,inplace=True)

train["FamilyID"].fillna(0,inplace = True)





train.drop(["Name","Ticket"],axis=1,inplace=True)
import plotly.express as px

fig = px.histogram(train, x="Sex", color='Survived', barmode='group',

             height=400)

fig.show()
train = train.assign(Female = (lambda x: ((x["Sex"] == "female")).astype(int)))

train = train.assign(Male = (lambda x: ((x["Sex"] == "male")).astype(int)))

train.drop("Sex",axis=1,inplace=True)
import plotly.express as px



df_for_bar = train.loc[:,["Pclass","Survived"]]

df_for_bar["Survived"].replace({0:"No",1:"Yes"},inplace = True)



fig = px.histogram(df_for_bar, x="Pclass", color='Survived')



fig.update_layout(

    xaxis_title_text='Pclass', 

    yaxis_title_text='Count', 

    bargap = 0.2)

fig.show()
number_rows = train.count()[0]



rates = train[train["Survived"]==1].groupby(["Pclass"]).count().iloc[:,0]



print(rates / number_rows)
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder()



results = ohe.fit_transform(train["Pclass"].values.reshape(-1,1),).toarray()



train["PClass_1"] = 0

train["PClass_2"] = 0

train["PClass_3"] = 0



train.iloc[:,[10,11,12]] = results

train.iloc[:,[10,11,12]] = train.iloc[:,[10,11,12]].astype(int)
df_for_pie = train[train["Survived"]==1].groupby(["Embarked"]).count()

fig = px.pie(df_for_pie, values='Survived', names=df_for_pie.index, title='Embarked Distribution Among Survived People')

fig.show()
embarked_three = pd.get_dummies(train["Embarked"])

train = pd.concat([train,embarked_three],axis=1)



train.drop("Embarked",axis=1,inplace=True)
train["NoChild"] = train["Parch"].loc[train["Parch"]==0]

train["NoChild"] = train["NoChild"].fillna(1).astype(int)



train["NoSibSp"] = train["SibSp"].loc[train["SibSp"]==0]

train["NoSibSp"] = train["NoSibSp"].fillna(1).astype(int)
train["FamilySize"] = train["Parch"] + train["SibSp"]
train.head()
fig = px.box(train[train["Fare"] != 512.3292], x="Pclass", y="Fare")

fig.show()
train["Pclass"].replace({1:3,3:1},inplace = True)  

train["Fare"] = train["Fare"] * train["Pclass"]

train["Age"] = train["Age"] * train["Pclass"] 

train["Pclass"].replace({1:3,3:1},inplace = True)  
fig = px.box(train[train["Fare"] < 700], x="Pclass", y="Fare")

fig.show()
train.drop(train.loc[:,["Pclass","Parch","SibSp","Cabin"]],axis=1,inplace=True)
y = train["Survived"].copy()

train.drop("Survived",axis=1,inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.13,random_state=42)

for train_index,cv_index in split.split(train,y):

    X_train, X_cv = train.loc[train_index], train.loc[cv_index]

    y_train, y_cv = y.loc[train_index], y.loc[cv_index]
from sklearn.metrics import accuracy_score

def predict_model(model,X_train,y_train,X_cv,y_cv):

    model.fit(X_train,y_train)

    return accuracy_score(y_train,model.predict(X_train)),accuracy_score(y_cv,model.predict(X_cv))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm



log_reg = LogisticRegression(max_iter=50000)

print("Logistic Regression    Train: "+"   CV:".join(map(str,predict_model(log_reg,X_train,y_train,X_cv,y_cv))))



rnd_clf = RandomForestClassifier()

print("Random Forest Classifier    Train: "+"   CV:".join(map(str,predict_model(rnd_clf,X_train,y_train,X_cv,y_cv))))



svc = svm.SVC(max_iter=50000)

print("Support Vector Machine Classifier    Train: "+"   CV:".join(map(str,predict_model(svc,X_train,y_train,X_cv,y_cv))))
train
train2 = pd.read_csv("/kaggle/input/titanic/train.csv")
def all_changes(X):

    X["Honorific"] = X["Name"].str.split(",",expand = True).iloc[:,1].str.split(".",expand = True).iloc[:,0].str.strip()

    X = X.assign(Mr = (X["Honorific"] == "Mr").astype(int))

    X = X.assign(Miss = (X["Honorific"] == "Miss").astype(int))

    X = X.assign(Mrs = (X["Honorific"] == "Mrs").astype(int))

    X = X.assign(Master = (X["Honorific"] == "Master").astype(int))

    X.drop("Honorific",axis=1,inplace=True)

    

    X["Name"]=X["Name"].str.split(",",expand = True)[0]

    family = X.loc[:,["Name","Ticket"]]

    family["new"] = family["Name"] + family["Ticket"]

    X["new"] = X["Name"] + X["Ticket"]

    family_unique = family["new"].unique().tolist()

    family_id = [i for i in range(1,len(family_unique)+1)]

    merge_df = pd.DataFrame(data={"new":family_unique,"FamilyID":family_id})

    

    X = X.join(merge_df.set_index("new"), on='new')

    X.drop("new",axis=1,inplace=True)

    X["FamilyID"].fillna(0,inplace = True)

    

    

    X.drop(X.loc[:,["PassengerId","Name","Ticket","Cabin"]],axis=1,inplace=True)

    

    X = X.assign(Female = (lambda x: ((x["Sex"] == "female")).astype(int)))

    X = X.assign(Male = (lambda x: ((x["Sex"] == "male")).astype(int)))

    

    

    

    mean = X["Age"].mean()

    X["Age"].fillna(mean,inplace = True)

       

    

    

    X = X.assign(P_1 = (X["Pclass"]== 1).astype(int))

    X = X.assign(P_2 = (X["Pclass"]== 2).astype(int))

    X = X.assign(P_3 = (X["Pclass"]== 3).astype(int))

    

    X["Embarked"].fillna("S",inplace = True)

    

    X = X.assign(C = (X["Embarked"]=="C").astype(int))

    X = X.assign(Q = (X["Embarked"]=="Q").astype(int))

    X = X.assign(S = (X["Embarked"]=="S").astype(int))

        



    X["NoChild"] = X["Parch"].loc[X["Parch"]==0]

    X["NoSibSp"] = X["SibSp"].loc[X["SibSp"]==0]

    

    X["NoChild"] = X["NoChild"].fillna(1).astype(int)

    X["NoSibSp"] = X["NoSibSp"].fillna(1).astype(int)

    



    X["FamilySize"] = X["Parch"] + X["SibSp"]

  

 # - - --- - - - - - - -- - - - - - -- - - - -- - - - - -- - - - - - - -- - - - - - -

    X["Pclass"].replace({1:3,3:1},inplace = True)

    

    X["Fare"] = X["Fare"] * X["Pclass"]

    X["Age"] = X["Age"] * X["Pclass"] 

    

    X.rename(columns={"P_1":"PClass_1","P_2":"PClass_2","P_3":"PClass_3"},inplace=True)



    X.drop(X.loc[:,["Sex","Embarked","Pclass","Parch","SibSp"]],axis=1,inplace=True)

    return X





train2 = all_changes(train2)
test.info()
test["Fare"].fillna(11,inplace = True)

test = all_changes(test)
test.head()
X_train.head()
test_result = log_reg.predict(test)

test_result = test_result.reshape(-1,1)



result = pd.DataFrame(data = {"PassengerId":np.arange(892,1310,1)})

result["Survived"]= test_result

result.to_csv(r'result.csv',index = False, header=True)