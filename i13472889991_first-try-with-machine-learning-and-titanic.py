import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/train.csv")
df.columns
sns.FacetGrid(df,hue="Sex").map(plt.scatter,"Age","Fare").add_legend()
ax = sns.boxplot(x="Survived",y="Fare",data=df)
ax = sns.stripplot(x="Survived", y="Fare", data=df, jitter=True)
plt.hist(x="Survived",data=df)
df.drop("PassengerId", axis=1).boxplot(by="Survived")
df['Age'].fillna((df['Age'].mean()),inplace = True)
df.Age.unique()
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

df['Sex'] = LabelEncoder().fit_transform(df["Sex"])
Y = df['Survived']
X = df[['Age','Fare','Sex','Pclass']]
models = []
models.append(("RF",RandomForestClassifier(n_estimators = 10000)))
models.append(("NB",GaussianNB()))
models.append(('KNN',KNeighborsRegressor()))
scoring = 'accuracy'
results = []
kfold = KFold(n_splits = 10, random_state=10)
model = RandomForestClassifier(n_estimators = 10000)
results = cross_val_score(model,X,Y,cv=kfold)
print(results.mean(),results.std())
for i in models:
    model = i[1]
    results = cross_val_score(model,X,Y,cv=kfold)
    print(results.mean(),results.std(),i[0])
    
Y = df['Survived']
df['Embarked'] = LabelEncoder().fit_transform(df["Embarked"])
X = df.drop(columns=['Survived','Name','Ticket','Cabin'])
models = []
models.append(("RF",RandomForestClassifier(n_estimators = 10000)))
models.append(("NB",GaussianNB()))
models.append(('KNN',KNeighborsClassifier()))
scoring = 'accuracy'
results = []
kfold = KFold(n_splits = 10, random_state=10)
for i in models:
    model = i[1]
    results = cross_val_score(model,X,Y,cv=kfold,scoring='accuracy')
    print(results.mean(),results.std(),i[0])
    
X = df.drop(columns=['Survived','Name','Ticket','Cabin'])

X = X.fillna(0)
models[0][1].fit(X,Y)
test = pd.read_csv('../input/test.csv')
test['Sex'] = LabelEncoder().fit_transform(test["Sex"])
test['Embarked'] = LabelEncoder().fit_transform(test["Embarked"])
X = test.drop(columns=['Name','Ticket','Cabin'])
X = X.fillna(0)
predictions = models[0][1].predict(X)
final = pd.DataFrame(data=[test['PassengerId'].values,predictions])
tempfinal = final.transpose()
tempfinal.columns = ['passengerId','Survived']
tempfinal.to_csv("Submission.csv",index=False)
pd.read_csv("../input/gender_submission.csv")
