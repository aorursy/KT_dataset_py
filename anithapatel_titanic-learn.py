import pandas as pd
df = pd.read_csv("../input/titanic/titanic.csv")
df.head()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()
inputs = df.drop('Survived',axis='columns')
target = df.Survived
inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
inputs.Age[:10]
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

from sklearn.preprocessing import LabelEncoder
le_Pclass = LabelEncoder()
le_Sex = LabelEncoder()
le_Age = LabelEncoder()
le_Fare=LabelEncoder()
inputs['Pclass_n'] = le_Pclass.fit_transform(inputs['Pclass'])
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
inputs['Age_n'] = le_Age.fit_transform(inputs['Age'])
inputs['Fare_n'] = le_Fare.fit_transform(inputs['Fare'])
inputs
inputs_n = inputs.drop(['Age','Sex','Fare','Pclass'],axis='columns')
inputs_n

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
model.score(inputs_n,target)
model.predict([[0,30,2,80]])
