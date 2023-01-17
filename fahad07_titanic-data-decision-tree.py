import pandas as pd
df = pd.read_csv('../input/decision-tree-on-titanic-data/train.csv')

df
inputs =df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head(10)
inputs = df.drop('Survived',axis = 'columns')
target = df.Survived
inputs.Gender = inputs.Gender.map({'male': 1,'female': 2})
inputs.Age[:10]

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test = train_test_split(inputs,target,test_size=0.2)
len(X_train)
len(X_test)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
model.predict([[1,2,20,7.2500]])