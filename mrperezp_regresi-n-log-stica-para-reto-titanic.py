# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

import os

print(os.listdir("../input"))

#Observemos qué hay en la base de "entranamiento"

df = pd.read_csv("../input/train.csv")

ts = pd.read_csv("../input/test.csv")



tr.head()
df.info()



ts.info()
#¿Hay datos nulos?

df.isnull().sum()



ts.isnull().sum()
print(df.columns.values)



print(ts.columns.values)

    

sns.catplot(x="Pclass", y="Age", row="Sex", col= "Embarked", kind="box", hue= "Survived", data =df)              



sns.catplot( x="Age", col= "Pclass", kind="count",  data =df) 
# Display the histogram to undestand the data

df.hist(bins=50, figsize=(20,15)) 



ts.hist(bins=50, figsize=(20,15))# pandas DataFrame

plt.show()


df=df.iloc[:,~df.columns.isin(['Name',  'Ticket' ,'Cabin'])]

ts=ts.iloc[:,~ts.columns.isin(['Name',  'Ticket' ,'Cabin'])]

print(df.columns.values)

print(ts.columns.values)



df["Age"].fillna(df.groupby("Pclass")["Age"].transform("mean"), inplace=True)

df["Embarked"].fillna(method='ffill', inplace=True)



ts["Age"].fillna(ts.groupby("Pclass")["Age"].transform("mean"), inplace=True)

ts["Fare"].fillna(ts.groupby("Pclass")["Fare"].transform("mean"), inplace=True)





df=pd.get_dummies(df,columns=['Pclass', 'Sex','SibSp' ,'Parch' ,'Embarked'],drop_first=True)



ts=pd.get_dummies(ts,columns=['Pclass', 'Sex','SibSp' ,'Parch' ,'Embarked'],drop_first=True)



df.describe()



ts.describe()



df.info()



ts.info()

#Ajustando el modelo de Regreción logistica

X_train = df.drop("Survived", axis=1)

Y_train = df["Survived"]

X_test  = ts.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
#Generate submission file

predictions=Y_pred

StackingSubmission=pd.DataFrame({'PassengerId':ts.PassengerId,'Survived':predictions})



StackingSubmission.to_csv("StackingSubmission.csv", index=False)

StackingSubmission