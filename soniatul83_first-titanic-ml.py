import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train_titanic = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_titanic = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# showing info of train and test data
train_titanic.info()
print("*****************************************")
test_titanic.info()
# Drop unnecessary columns from train and test, columns won't be useful in analysis and prediction
def DropColumnFromTrainDF():
        train_df=train_titanic.drop(['PassengerId','Name','Ticket'], axis=1)
        return train_df
def DropColumnFromTestDF():
    test_df=test_titanic.drop(['Name','Ticket'],axis=1)
    return test_df

# calling drop column function for train df
train_df=DropColumnFromTrainDF()
#train_df.head()
#calling drop column function for test df
test_df=DropColumnFromTestDF()
#test_df.head()

def fillNA_data():
    train_df["Embarked"] = train_df["Embarked"].fillna("S")
    return train_df

train_df=fillNA_data()
test_df["Embarked"] = test_df["Embarked"].fillna("S")
test_df.head()
#train_df.head()

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert data from float to int
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)
train_df.head()

# So, we can classify passengers as males, females, and child
# Create new column in train and test df
train_df.loc[train_df["Sex"] == "male", "Sex"] = 0
train_df.loc[train_df["Sex"] == "female", "Sex"] = 1

test_df.loc[test_df["Sex"] == "male", "Sex"] = 0
test_df.loc[test_df["Sex"] == "female", "Sex"] = 1

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(train_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

#train_df.drop(['Pclass'],axis=1,inplace=True)
#test_df.drop(['Pclass'],axis=1,inplace=True)
train_df = train_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)
train_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

train_df.head()

train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
train_df["Age"].median()

train_df.loc[train_df["Embarked"] == "S", "Embarked"] = 0
train_df.loc[train_df["Embarked"] == "C", "Embarked"] = 1
train_df.loc[train_df["Embarked"] == "Q", "Embarked"] = 2

test_df.loc[test_df["Embarked"] == "S", "Embarked"] = 0
#test_df.loc[test_df["Embarked"] == "C", "Embarked"] = 1
test_df.loc[test_df["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# machine learning
from sklearn.linear_model import LogisticRegression

from sklearn import cross_validation

# Logistic Regression

alg    = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(
    alg,
    train_df[predictors],
    train_df["Survived"],
    cv=3
)

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = RandomForestClassifier(
    random_state=1,
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2
)

scores = cross_validation.cross_val_score(
    alg,
    train_df[predictors],
    train_df["Survived"],
    cv=3
)

print(scores.mean())

train_df['Age'] = train_df['Age'].astype(int)
def Filesubmission(alg, train, test, predictors, filename):

    alg.fit(train[predictors], train["Survived"])
    predictions = alg.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
    submission.to_csv(filename, index=False)
    
predictors = ["Pclass","Sex"]

Filesubmission(alg, train_df, test_df, predictors, "sample01.csv")