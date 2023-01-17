import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline
df = pd.read_csv("../input/titanic/train.csv")

df.head(10)
df.info()
df.describe()
import pandas_profiling as pp

pp.ProfileReport(df)

import plotly

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot 

init_notebook_mode(connected=True)
df_sur=df[df['Survived']==0]

df_not=df[df['Survived']==1]
col = "Sex"

v1 = df_sur[col].value_counts().reset_index()

v1 = v1.rename(columns={col:'count',"index":col})

v1["percent"] = v1["count"].apply(lambda x:100*x/sum(v1["count"]))

v1 = v1.sort_values(col)



v2 = df_not[col].value_counts().reset_index()

v2 = v2.rename(columns={col:'count',"index":col})

v2["percent"] = v2["count"].apply(lambda x:100*x/sum(v1["count"]))

v2 = v2.sort_values(col) 



bar1 = go.Bar(x = v1[col], y=v1["count"], name="0")

bar2 = go.Bar(x=v2[col], y=v2["count"], name="1")



data = [bar1, bar2]

layout={'title':"surviving rate male vs female",'xaxis':{'title':"Sex"}}

fig = go.Figure(data=data, layout=layout)

fig.layout.template = "presentation"



iplot(fig)



col = "Pclass"

v1 = df_sur[col].value_counts().reset_index()

v1 = v1.rename(columns={col:'count',"index":col})

v1["percent"] = v1["count"].apply(lambda x:100*x/sum(v1["count"]))

v1 = v1.sort_values(col)



v2 = df_not[col].value_counts().reset_index()

v2 = v2.rename(columns={col:'count',"index":col})

v2["percent"] = v2["count"].apply(lambda x:100*x/sum(v1["count"]))

v2 = v2.sort_values(col) 



bar1 = go.Bar(x = v1[col], y=v1["count"], name="0")

bar2 = go.Bar(x=v2[col], y=v2["count"], name="1")



data = [bar1, bar2]

layout={'title':"surviving rate in Pclass",'xaxis':{'title':"Pclass"},'barmode': 'relative'}

fig = go.Figure(data=data, layout=layout)

fig.layout.template = "presentation"



iplot(fig)
col = "Embarked"

v1 = df_sur[col].value_counts().reset_index()

v1 = v1.rename(columns={col:'count',"index":col})

v1["percent"] = v1["count"].apply(lambda x:100*x/sum(v1["count"]))

v1 = v1.sort_values(col)



v2 = df_not[col].value_counts().reset_index()

v2 = v2.rename(columns={col:'count',"index":col})

v2["percent"] = v2["count"].apply(lambda x:100*x/sum(v1["count"]))

v2 = v2.sort_values(col) 



bar1 = go.Bar(x = v1[col], y=v1["count"], name="0")

bar2 = go.Bar(x=v2[col], y=v2["count"], name="1")



data = [bar1, bar2]

layout={'title':"surviving rate in Embarked",'xaxis':{'title':"Embarked"}}

fig = go.Figure(data=data, layout=layout)

fig.layout.template = "presentation"



iplot(fig)
col = "SibSp"

v1 = df_sur[col].value_counts().reset_index()

v1 = v1.rename(columns={col:'count',"index":col})

v1["percent"] = v1["count"].apply(lambda x:100*x/sum(v1["count"]))

v1 = v1.sort_values(col)



v2 = df_not[col].value_counts().reset_index()

v2 = v2.rename(columns={col:'count',"index":col})

v2["percent"] = v2["count"].apply(lambda x:100*x/sum(v1["count"]))

v2 = v2.sort_values(col) 



bar1 = go.Bar(x = v1[col], y=v1["count"], name="0")

bar2 = go.Bar(x=v2[col], y=v2["count"], name="1")



data = [bar1, bar2]

layout={'title':"surviving rate in SibSp",'xaxis':{'title':"SibSp"}}

fig = go.Figure(data=data, layout=layout)

fig.layout.template = "presentation"



iplot(fig)
col = "Age"

v1 = df_sur[col].value_counts().reset_index()

v1 = v1.rename(columns={col:'count',"index":col})

v1["percent"] = v1["count"].apply(lambda x:100*x/sum(v1["count"]))

v1 = v1.sort_values(col)



v2 = df_not[col].value_counts().reset_index()

v2 = v2.rename(columns={col:'count',"index":col})

v2["percent"] = v2["count"].apply(lambda x:100*x/sum(v1["count"]))

v2 = v2.sort_values(col) 



scat1 = go.Scatter(x = v1[col], y=v1["count"], name="0")

scat2 = go.Scatter(x=v2[col], y=v2["count"], name="1")



data = [scat1, scat2]

layout={'title':"surviving rate on the basic of Age with their names",'xaxis':{'title':"Age"}}

fig = go.Figure(data=data, layout=layout)

fig.layout.template = "presentation"



iplot(fig)
col = "Parch"

v1 = df_sur[col].value_counts().reset_index()

v1 = v1.rename(columns={col:'count',"index":col})

v1["percent"] = v1["count"].apply(lambda x:100*x/sum(v1["count"]))

v1 = v1.sort_values(col)



v2 = df_not[col].value_counts().reset_index()

v2 = v2.rename(columns={col:'count',"index":col})

v2["percent"] = v2["count"].apply(lambda x:100*x/sum(v1["count"]))

v2 = v2.sort_values(col) 



bar1 = go.Bar(x = v1[col], y=v1["count"], name="0")

bar2 = go.Bar(x=v2[col], y=v2["count"], name="1")



data = [bar1, bar2]

layout={'title':"surviving rate on the basic of Parch",'xaxis':{'title':"Parch"},'barmode': 'relative'}

fig = go.Figure(data=data, layout=layout)

fig.layout.template = "presentation"



iplot(fig)
col = "Fare"

v1 = df_sur[col].value_counts().reset_index()

v1 = v1.rename(columns={col:'count',"index":col})

v1["percent"] = v1["count"].apply(lambda x:100*x/sum(v1["count"]))

v1 = v1.sort_values(col)



v2 = df_not[col].value_counts().reset_index()

v2 = v2.rename(columns={col:'count',"index":col})

v2["percent"] = v2["count"].apply(lambda x:100*x/sum(v1["count"]))

v2 = v2.sort_values(col) 



scat1 = go.Scatter(x = v1[col], y=v1["count"], name="0")

scat2 = go.Scatter(x=v2[col], y=v2["count"], name="1")



data = [scat1, scat2]

layout={'title':"surviving rate on the basic of fare with their names",'xaxis':{'title':"Fare"}}

fig = go.Figure(data=data, layout=layout)

fig.layout.template = "presentation"



iplot(fig)
df.corr()
fig,ax = plt.subplots(figsize=(9,8))

ax = sb.heatmap(df.corr(), annot=True,linewidths=.5,fmt='.1f')

plt.show()
df["Parch"].unique()
df["SibSp"].unique()
df.head(10)
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.mean()))

df.head()
df["AgeGrp"] = 1

df.loc[(df.Age.values<12),"AgeGrp"] = 0

df.loc[((df.Age.values>30)*(df.Age.values<50)),"AgeGrp"] = 2

df.loc[(df.Age.values>=50),"AgeGrp"] = 3

df.head()
df.drop(['PassengerId','Name','Age','Parch','Ticket','Cabin'],axis=1,inplace=True)
df = pd.get_dummies(df)

df.head()
X = df.drop(['Survived'], axis = 1)

y = df["Survived"].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_recall_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.svm import SVC

#LogisticRegression

lr_c=LogisticRegression(random_state=0)

lr_c.fit(X_train,y_train)

lr_pred=lr_c.predict(X_test)

lr_cm=confusion_matrix(y_test,lr_pred)

lr_ac=accuracy_score(y_test, lr_pred)



#SVM classifier

svc_c=SVC(kernel='linear',random_state=0)

svc_c.fit(X_train,y_train)

svc_pred=svc_c.predict(X_test)

sv_cm=confusion_matrix(y_test,svc_pred)

sv_ac=accuracy_score(y_test, svc_pred)



#SVM regressor

svc_r=SVC(kernel='rbf')

svc_r.fit(X_train,y_train)

svr_pred=svc_r.predict(X_test)

svr_cm=confusion_matrix(y_test,svr_pred)

svr_ac=accuracy_score(y_test, svr_pred)



#RandomForest

rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rdf_c.fit(X_train,y_train)

rdf_pred=rdf_c.predict(X_test)

rdf_cm=confusion_matrix(y_test,rdf_pred)

rdf_ac=accuracy_score(rdf_pred,y_test)



# DecisionTree Classifier

dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)

dtree_c.fit(X_train,y_train)

dtree_pred=dtree_c.predict(X_test)

dtree_cm=confusion_matrix(y_test,dtree_pred)

dtree_ac=accuracy_score(dtree_pred,y_test)



#KNN

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

knn_pred=knn.predict(X_test)

knn_cm=confusion_matrix(y_test,knn_pred)

knn_ac=accuracy_score(knn_pred,y_test)



#GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=30, max_features=4, max_depth=12, random_state=0)

gbc.fit(X_train,y_train)

gbc_pred=gbc.predict(X_test)

gbc_cm=confusion_matrix(y_test,gbc_pred)

gbc_ac=accuracy_score(gbc_pred,y_test)
print("LogisticRegression_accuracy:",lr_ac)

print("SVM_regressor_accuracy:",svr_ac)

print("RandomForest_accuracy:",rdf_ac)

print("DecisionTree_accuracy",dtree_ac)

print("KNN_accuracy:",knn_ac)

print("SVM_classifier_accuracy:",sv_ac)

print("GradientBoostingClassifier_accuracy:",gbc_ac)
plt.title("LR CM")

sb.heatmap(lr_cm,annot=True)
plt.title("SVMR CM")

sb.heatmap(svr_cm,annot=True)
plt.title("RDF CM")

sb.heatmap(rdf_cm,annot=True)
plt.title("DTREE CM")

sb.heatmap(dtree_cm,annot=True)
plt.title("KNN CM")

sb.heatmap(knn_cm,annot=True)
plt.title("SVC CM")

sb.heatmap(sv_cm,annot=True)
plt.title("GBC CM")

sb.heatmap(gbc_cm,annot=True)
df_test = pd.read_csv("../input/titanic/test.csv")

df_test['Age'] = df_test.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.mean()))

df_test.head()
df_test['Fare'] = df_test.groupby('Sex')['Fare'].apply(lambda x: x.fillna(x.mean()))

df_test.head()
df_test["AgeGrp"] = 1

df_test.loc[(df_test["Age"].values<12),"AgeGrp"] = 0

df_test.loc[((df_test["Age"].values>30)*(df_test.Age.values<50)),"AgeGrp"] = 2

df_test.loc[(df_test["Age"].values>=50),"AgeGrp"] = 3

df_test.head()
df_test.drop(['Name','Age','Parch','Ticket','Cabin'],axis=1,inplace=True)
df_test = pd.get_dummies(df_test)

df_test.head()
survived = dtree_c.predict(df_test.drop(["PassengerId"], axis=1).values)
df_result = pd.DataFrame()

df_result["PassengerId"] = df_test["PassengerId"]

df_result["Survived"] = survived
df_result.head()
df_result.to_csv("submission.csv",index=False)
df["Fare"] = np.log1p(df["Fare"])
df.head()
X = df.drop(['Survived'], axis = 1)

y = df["Survived"].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
gbc = GradientBoostingClassifier(n_estimators=50,max_features=9,max_depth=20)
gbc.fit(X_train,y_train)
pred = gbc.predict(X_test)
accuracy_score(pred,y_test)
f1_score(pred,y_test)
rdf_c = RandomForestClassifier()

rdf_c.fit(X_train,y_train)
pred = rdf_c.predict(X_test)

accuracy_score(pred,y_test)
f1_score(pred,y_test)
df_test["Fare"] = np.log1p(df_test["Fare"])
survied = rdf_c.predict(df_test.drop(["PassengerId"],axis=1))
df_result["Survived"] = survived

df_result.to_csv("submission.csv",index=False)