import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
# Train: fill the NaN values
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())
titanic_train['Fare'] = titanic_train['Fare'].fillna(titanic_train['Fare'].mean())
titanic_train['Embarked'] = titanic_train['Embarked'].fillna('0')
# Train: reclassify literals to numeric
titanic_train.loc[titanic_train['Sex'] == 'male', 'Sex'] = 0
titanic_train.loc[titanic_train['Sex'] == 'female', 'Sex'] = 1
titanic_train.loc[titanic_train['Embarked'] == 'S', 'Embarked'] = 1
titanic_train.loc[titanic_train['Embarked'] == 'C', 'Embarked'] = 1
titanic_train.loc[titanic_train['Embarked'] == 'Q', 'Embarked'] = 1
#titanic_train.loc[titanic_train['Embarked'].isnull(), 'Embarked'] = 0

# Test: fill the NaN values
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_train['Age'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].mean())
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('0')
# Test: reclassify literals to numeric
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'S', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked'] == 'Q', 'Embarked'] = 1
#titanic_train.loc[titanic_train['Embarked'].isnull(), 'Embarked'] = 0
titanic_train['Embarked'].isnull().head()
#predictors = ["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
predictors = ["Sex", "Age", "SibSp", "Parch", "Fare"]
alg = LogisticRegression(random_state=1)
alg.fit(titanic_train[predictors], titanic_train["Survived"])
predictions = alg.predict(titanic_test[predictors])
# check statistics for the built model (a la Eviews)
import statsmodels.formula.api as sm
model = sm.ols(formula='Survived ~ Sex+Age+SibSp+Parch+Fare+Embarked', data=titanic_train)
fitted1 = model.fit()
fitted1.summary()
model = sm.ols(formula='Survived ~ Sex+Age+SibSp+Parch+Fare', data=titanic_train)
fitted1 = model.fit()
fitted1.summary()
model = sm.ols(formula='Survived ~ Sex+Age+SibSp+Fare', data=titanic_train)
fitted1 = model.fit()
fitted1.summary()
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("kaggle7.csv", index=False)