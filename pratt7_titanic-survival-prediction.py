import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
data_train.Age.fillna(data_train.Age.mean(), inplace = True)
sex_pivot = data_train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()
ns = data_train.loc[data_train["Survived"] == False] # Didn't survive :(
s  = data_train.loc[data_train["Survived"] == True]  # Survivors

sns.distplot(ns["Age"])
sns.distplot(s["Age"])
plt.show()
data_train.drop("PassengerId", axis = 1, inplace = True)
data_train.drop("Name", axis=1, inplace = True)
data_train.drop("Ticket", axis=1, inplace = True)
data_train.drop("Cabin", axis=1, inplace = True)
data_train.drop("Embarked", axis=1, inplace = True)


data_test.drop("PassengerId", axis = 1, inplace = True)
data_test.drop("Name", axis=1, inplace = True)
data_test.drop("Ticket", axis=1, inplace = True)
data_test.drop("Cabin", axis=1, inplace = True)
data_test.drop("Embarked", axis=1, inplace = True)
data_train.head()
data_train.isnull().any()
data_train.isnull().any()
data_test.isnull().any()
data_test.Age.fillna(data_test.Age.mean(), inplace = True)
data_test.Fare.fillna(data_test.Fare.mean(), inplace = True)
data_test.isnull().any()
data_train.Sex.replace(['male', 'female'], [0, 1], inplace=True)
data_test.Sex.replace(['male', 'female'], [0, 1], inplace=True)
data_train.head()
data_test.head()
data_train.describe()
data_test.describe()
data_whole = pd.concat([data_train, data_test])
data_whole.describe()
data_whole.tail()
del data_whole['Survived']
data_whole.describe()
from sklearn import preprocessing
data_scaled = pd.DataFrame(preprocessing.scale(data_whole))
data_scaled.describe()
titanic_train_x = data_whole.iloc[0:891,:]
titanic_test_x = data_whole.iloc[891:1309,:]
titanic_train_y = data_train.iloc[:,0]

titanic_train_x = titanic_train_x.values
titanic_test_x = titanic_test_x.values
titanic_train_y = titanic_train_y.values
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
model_LR = clf.fit(titanic_train_x, titanic_train_y)
clf.score(titanic_train_x, titanic_train_y)
output = clf.predict(titanic_test_x)
df = pd.DataFrame(output)
data_test_df = pd.read_csv("../input/test.csv")
df["PassengerId"] = data_test_df["PassengerId"]
df.head()
df.columns = ["Survived", "PassengerId"]
result = df.reindex(columns = ["PassengerId", "Survived"])
result.to_csv("titanic_lg.csv", header=True, index=False,  )
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(titanic_train_x, titanic_train_y)
from sklearn import cross_validation
scores = cross_validation.cross_val_score(model, titanic_train_x, titanic_train_y, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predictions = model.predict(titanic_test_x)
submission = pd.DataFrame({
        "PassengerId": data_test_df["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('titanic_rf.csv', index=False)

