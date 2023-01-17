import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#beginners always make a mistake of importing train.csv lile
#this pd.read_csv('train.csv')
#the data source you import is saved in input folder on the cloud
data = pd.read_csv('../input/train.csv')
data.describe()
data.head(15)
y = data.pop("Survived")
y.head()
data.tail()
data["Age"].fillna(data.Age.mean(),inplace=True)
data.tail()
nvar = list(data.dtypes[data.dtypes !='object'].index)
nvar
data[nvar].describe()
model = RandomForestClassifier(n_estimators=100)
model.fit(data[nvar],y)
accuracy_score(y,model.predict(data[nvar]))
test = pd.read_csv('../input/test.csv')
test[nvar].tail()
test["Age"]
test["Age"].fillna(test.Age.mean(),inplace=True)
test = test[nvar].fillna(test.mean()).copy()
y_predictions = model.predict(test[nvar])
survivors = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": y_predictions
})
survivors.to_csv('fk_titanic.csv',index = False)
