import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier as rf



train = pd.read_csv("../input/train.csv")



test = pd.read_csv("../input/test.csv")
train["Fam"] = 0

train["Fam"][(train["Parch"] + train["SibSp"]) > 3] = 1

test["Fam"] = 0

test["Fam"][(test["Parch"] + test["SibSp"]) > 3] = 1

train["Embarked"].fillna(value = "S", inplace = True)

train['Age'].fillna(value = np.mean(train['Age']), inplace = True)

test['Age'].fillna(value = np.mean(test['Age']), inplace = True)

test['Fare'].fillna(value = np.mean(test['Fare']), inplace = True)

train['Age_pot'] = np.where(np.logical_and(train['Age']<=8, train['Sex'] == 'male'), 0, np.where(np.logical_and(train['Age']<=15, train['Sex'] == 'male'), 1, np.where(np.logical_and(train['Age']<=25 , train['Sex'] == 'male'), 2, np.where(np.logical_and(train['Age']<=40, train['Sex'] == 'male'), 3, np.where(np.logical_and(train['Age']<=60, train['Sex'] == 'male'), 4, np.where(np.logical_and(train['Age'] > 60, train['Sex'] == 'male'), 5, np.where(np.logical_and(train['Age']<=8, train['Sex'] == 'female'), 6, np.where(np.logical_and(train['Age']<=15, train['Sex'] == 'female'), 7, np.where(np.logical_and(train['Age']<=25, train['Sex'] == 'female'), 8, np.where(np.logical_and(train['Age']<=40, train['Sex'] == 'female'), 9, np.where(np.logical_and(train['Age']<=60, train['Sex'] == 'female'), 10, 11)))))))))))

test['Age_pot'] = np.where(np.logical_and(test['Age']<=8, test['Sex'] == 'male'), 0, np.where(np.logical_and(test['Age']<=15, test['Sex'] == 'male'), 1, np.where(np.logical_and(test['Age']<=25 , test['Sex'] == 'male'), 2, np.where(np.logical_and(test['Age']<=40, test['Sex'] == 'male'), 3, np.where(np.logical_and(test['Age']<=60, test['Sex'] == 'male'), 4, np.where(np.logical_and(test['Age'] > 60, test['Sex'] == 'male'), 5, np.where(np.logical_and(test['Age']<=8, test['Sex'] == 'female'), 6, np.where(np.logical_and(test['Age']<=15, test['Sex'] == 'female'), 7, np.where(np.logical_and(test['Age']<=25, test['Sex'] == 'female'), 8, np.where(np.logical_and(test['Age']<=40, test['Sex'] == 'female'), 9, np.where(np.logical_and(test['Age']<=60, test['Sex'] == 'female'), 10, 11)))))))))))

train.loc[ train['Fare'] <= 9, 'Fare'] 						        = 0

train.loc[(train['Fare'] > 9) & (train['Fare'] <= 30), 'Fare'] = 1

train.loc[(train['Fare'] > 30) & (train['Fare'] <= 40), 'Fare']   = 2

train.loc[(train['Fare'] > 40) & (train['Fare'] <= 80), 'Fare']   = 3

train.loc[ train['Fare'] > 80, 'Fare'] = 4

train['Fare'] = train['Fare'].astype(int)

test.loc[ test['Fare'] <= 9, 'Fare'] 						        = 0

test.loc[(test['Fare'] > 9) & (test['Fare'] <= 30), 'Fare'] = 1

test.loc[(test['Fare'] > 30) & (test['Fare'] <= 40), 'Fare']   = 2

test.loc[(test['Fare'] > 40) & (test['Fare'] <= 80), 'Fare']   = 3

test.loc[ test['Fare'] > 80, 'Fare'] = 4

train['Fare'] = train['Fare'].astype(int)
train.head()

x1 = train[["Sex","Fam", "Embarked", "Pclass", "Age_pot","Fare"]]

y = train["Survived"]

test1 = test[["Sex","PassengerId", "Fam", "Embarked", "Pclass", "Age_pot","Fare"]]
x1 = pd.get_dummies(x1, drop_first=True)

x2 = pd.get_dummies(test1, drop_first=True)
x1.head()
clf = rf(n_estimators = 1000, max_features=3, max_depth = 5, min_samples_leaf = 10)

clf.fit(x1, y)

clf.score(x1,y)
pred = clf.predict(x2.drop("PassengerId", axis = 1))

submission = pd.DataFrame({

        "PassengerId": x2["PassengerId"],

        "Survived": pred

    })

submission.to_csv('submi.csv', index=False)
#pd.set_option('display.max_rows', len(submission))

submission.head()
