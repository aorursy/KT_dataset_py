import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt
data = pd.read_csv('../input/titanic/titanic3.csv')
data.head()
data.tail()
data.shape
data.columns.values
data.describe()
data.dtypes
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
data = data.drop(['name','ticket','body','cabin','boat','home.dest'], axis=1)
print("Numero de registros:"+str(data.shape[0]))

for column in data.columns.values:

    print(column + "-NAs:"+ str(pd.isnull(data[column]).values.ravel().sum()))
%%capture

data["age"]=data["age"].fillna(data["age"].mean())
data['embarked'].unique()
data['embarked'].value_counts()
data['fare'].value_counts()
data["fare"]=data["fare"].fillna(8.0500)
data["embarked"]=data["embarked"].fillna('S')
data.head()
print("Cantidad de Hermanos/Hermanastros")

data['sibsp'].value_counts()
print("Quantity of sons or relatives")

data['parch'].value_counts()
data.groupby("pclass").mean()
%matplotlib inline

pd.crosstab(data.pclass,data.survived).plot(kind="bar")

plt.title("Frecuency of survival based on People Class")

plt.xlabel("Social Class")

plt.ylabel("Survive")
table = pd.crosstab(data.pclass,data.survived)

#Hago una division para dejarlo entre 0 y 1

#para que todos queden escalados

table.div(table.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.title("Stack Diagram of class vs survival")

plt.xlabel("Social Class")

plt.ylabel("Survival Proportions")
%matplotlib inline

data.age.hist()

plt.title("Age Histogram")

plt.xlabel("Age")

plt.ylabel("Passengers")
data_sex_plot = data

data_sex_plot["sex"]= np.where(data_sex_plot["sex"]=="female",1,data_sex_plot["sex"])

data_sex_plot["sex"]= np.where(data_sex_plot["sex"]=="male",0,data_sex_plot["sex"])
table = pd.crosstab(data_sex_plot.sex,data_sex_plot.survived)

#Hago una division para dejarlo entre 0 y 1

#para que todos queden escalados

table.div(table.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.title("Stack diagram of sex vs survival")

plt.xlabel("Sex")

plt.ylabel("Survivals Proportion")
data_embarked_plot = data

data_embarked_plot["embarked"]= np.where(data_embarked_plot["embarked"]=="S",0,data_embarked_plot["embarked"])

data_embarked_plot["embarked"]= np.where(data_embarked_plot["embarked"]=="C",1,data_embarked_plot["embarked"])

data_embarked_plot["embarked"]= np.where(data_embarked_plot["embarked"]=="Q",2,data_embarked_plot["embarked"])
table = pd.crosstab(data_embarked_plot.embarked,data_sex_plot.survived)

#Hago una division para dejarlo entre 0 y 1

#para que todos queden escalados

table.div(table.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.title("Stack Diagram of place where people embark against survival")

plt.xlabel("Embark Place")

plt.ylabel("Survivals Proportion")
table = pd.crosstab(data_embarked_plot.embarked,data.pclass)

#Hago una division para dejarlo entre 0 y 1

#para que todos queden escalados

table.div(table.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)

plt.title("Stack Diagram of place where people embark vs social class")

plt.xlabel("Embark Place")

plt.ylabel("Social Class Proportion")
%matplotlib inline

data.pclass.hist()

plt.title("Social Class Histogram")

plt.xlabel("Social Class")

plt.ylabel("Passengers")
dummy_sex = pd.get_dummies(data["sex"],prefix="sex")

data = data.drop(["sex"],axis=1)

data = pd.concat([data,dummy_sex],axis=1)





dummy_embarked = pd.get_dummies(data["embarked"],prefix="embarked")

data = data.drop(["embarked"],axis=1)

data = pd.concat([data,dummy_embarked],axis=1)



dummy_class= pd.get_dummies(data["pclass"],prefix="class")

data = data.drop(["pclass"],axis=1)

data = pd.concat([data,dummy_class],axis=1)
data_vars = data.columns.values.tolist()

Y = ['survived']

X = [v for v in data_vars if v not in Y]



n = len(data.columns.values)

print(n)

from sklearn import datasets

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="lbfgs", max_iter=3500)
rfe = RFE(lr,n)
rfe = rfe.fit(data[X],data[Y].values.ravel())
print(rfe.support_)
print(rfe.ranking_)
z = zip(data_vars,rfe.support_,rfe.ranking_)

list(z)
from sklearn import linear_model
logit_model = linear_model.LogisticRegression(solver="lbfgs", max_iter=3500)

logit_model.fit(data[X],data[Y].values.ravel())
logit_model.score(data[X],data[Y])
pd.DataFrame(list(zip(data[X].columns,np.transpose(logit_model.coef_))))
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data[X],data[Y],test_size = 0.3, random_state=0)
lm = linear_model.LogisticRegression(solver="lbfgs", max_iter=3500)

lm.fit(X_train, Y_train.values.ravel())
from IPython.display import display, Math, Latex

display(Math(r'Y_p=\begin{cases}0& si\ p\leq0.5\\1&si\ p >0.5\end{cases}'))
probs = lm.predict_proba(X_test)
probs
prediction = lm.predict(X_test)
logit_model.score(X_test,Y_test)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(linear_model.LogisticRegression(solver="lbfgs", max_iter=3500), data[X],data[Y].values.ravel(), scoring="accuracy", cv=20)
scores
scores.mean()
X_train,X_test,Y_train,Y_test = train_test_split(data[X],data[Y],test_size=0.3,random_state=0)
lm = linear_model.LogisticRegression(solver="lbfgs", max_iter=3500)

lm.fit(X_train,Y_train.values.ravel())

probs = lm.predict_proba(X_test)

prob = probs[:,1]

prob_df= pd.DataFrame(prob)
thresholds = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.3, 0.4, 0.5]

sensitivities = [1]

especifities_1 = [1]



for t in thresholds:

    prob_df["prediction"] = np.where(prob_df[0]>=t, 1, 0)

    prob_df["actual"] = Y_test.values.ravel()

    prob_df.head()



    confusion_matrix = pd.crosstab(prob_df.prediction, prob_df.actual)

    TN=confusion_matrix[0][0]

    TP=confusion_matrix[1][1]

    FP=confusion_matrix[0][1]

    FN=confusion_matrix[1][0]

    

    sens = TP/(TP+FN)

    sensitivities.append(sens)

    espc_1 = 1-TN/(TN+FP)

    especifities_1.append(espc_1)



sensitivities.append(0)

especifities_1.append(0)
sensitivities
especifities_1
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(especifities_1, sensitivities, marker="o", linestyle="--", color="r")

x=[i*0.01 for i in range(100)]

y=[i*0.01 for i in range(100)]

plt.plot(x,y)

plt.xlabel("1-Specifity")

plt.ylabel("Sensibility")

plt.title("ROC Curve")