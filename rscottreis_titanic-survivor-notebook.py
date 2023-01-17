import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix



train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

full_data = pd.concat([train.drop('Survived', axis=1), test])

train.head()
def preprocess(df):

    # imputation

    df['Embarked'].fillna('S', inplace=True)

    df['Age'].fillna(full_data.Age.mean(), inplace=True)

    # feature creation

    df['high_class_lady'] = df.Pclass.isin([1,2]).bool and df.Sex == 'female'

    df['low_class_man'] = df.Pclass.isin([2,3]).bool and df.Sex == 'male'

    df['child'] = df.Age < 6

    df['free_ride'] = df.Fare == 0

    df['embarked_c'] = df.Embarked == 'C'

    df['travel_alone'] = df.SibSp.eq(0).bool and df.Parch.eq(0)

    df['has_cabin'] = pd.isna(df.Cabin)

    return df



train = preprocess(train)

test = preprocess(test)
y = train['Survived']



features = ['high_class_lady', 'low_class_man', 'child', 'free_ride', 'embarked_c', 'travel_alone', 'has_cabin']

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, random_state=42)

model.fit(X, y)



scores = cross_val_score(model, X, y, cv=3)

print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

print("TN {} FP {} FN {} TP {}".format(*confusion_matrix(y, model.predict(X)).ravel()))



predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test.index, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")