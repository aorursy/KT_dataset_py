import pandas as pd

df = pd.read_csv("../input/titanicdataset-traincsv/train.csv")

df.head()
df.drop(['PassengerId','Name','SibSp','Parch','Cabin','Ticket','Embarked'],axis='columns',inplace = True)

df.head()

inputs = df.drop(['Survived'],axis='columns')

target = df.Survived
dummies = pd.get_dummies(inputs.Sex)

dummies.head()
inputs = pd.concat([inputs,dummies],axis='columns')

inputs
inputs.drop(['Sex'],axis='columns',inplace=True)

inputs
inputs.columns[inputs.isna().any()]

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

inputs.head(6)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size= 0.2)

len(X_train)
len(X_test)
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,y_train)
model.score(X_test,y_test)
model.predict(X_test[:10])
model.predict_proba(X_test[:10])