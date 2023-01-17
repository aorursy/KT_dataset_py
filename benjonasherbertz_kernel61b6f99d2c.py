import pandas as pd

def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

test = pd.read_csv("/kaggle/input/titanic/test.csv")

clean_data(test)

test["Survived"] = 0
test.loc[(test.Name.str.contains("Mrs")), "Survived"] = 1
test.loc[(test.Name.str.contains("Mrs")) & (test.Pclass > 2) & (test.Age > 38), "Survived"] = 1
test.loc[(test.Name.str.contains("Mrs")) & (test.Pclass < 3) & (test.Age > 24) & (test.Age > 28), "Survived"] = 1
test.loc[(test.Name == "Mr"), "Survived"] = 0
test.loc[(test.Name.str.contains("Mrs") == False) & (test.Name.str.contains("Mr") == False), "Survived"] = 0
test.loc[(test.Name.str.contains("Mrs") == False) & (test.Name.str.contains("Mr") == False) & (test.SibSp < 3), "Survived"] = 1
#test.loc[(test.Sex == 0) & (test.Age < 6.5), "Survived"] = 1

out = pd.DataFrame(columns = ["PassengerId", "Survived"])
out = test[["PassengerId","Survived"]].copy()

out.head()

out.to_csv('submission.csv', index=False)
out.head()
