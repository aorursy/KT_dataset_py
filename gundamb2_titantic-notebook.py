import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



train.head()
test.head()
def process_df(df):

    df['Title'] = df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

    normalized_titles = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Mme":        "Mrs",

        "Mlle":       "Miss",

        "Ms":         "Mrs",

        "Mr" :        "Mr",

        "Mrs" :       "Mrs",

        "Miss" :      "Miss",

        "Master" :    "Master",

        "Lady" :      "Royalty"

    }

    df['Title'] = df['Title'].map(normalized_titles)



    grouped = df.groupby(['Sex', 'Pclass', 'Title'])

    df['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))

    df['Cabin'] = df['Cabin'].fillna('U')

    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['FamilySize'] = df['Parch'] + df['SibSp']

    df['Deck'] = df['Cabin'].str.get(0)

    df['Title'] = df['Title'].astype('category').cat.codes

    df['Sex'] = df['Sex'].astype('category').cat.codes

    df['Cabin'] = df['Cabin'].astype('category').cat.codes

    df['Embarked'] = df['Embarked'].astype('category').cat.codes

    df['Pclass'] = df['Pclass'].astype('category').cat.codes

    df['Deck'] = df['Deck'].astype('category').cat.codes

    

    return df



train = process_df(train)
train
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Title', 'FamilySize', 'Deck']



X = train[columns]

y = train['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print("Average Accuracy: ",np.mean(acc_scores_cv))

print("Average F1: ", np.mean(f1_scores_cv))
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



params = dict(     

    max_depth = [n for n in range(9, 14)],     

    min_samples_split = [n for n in range(4, 11)], 

    min_samples_leaf = [n for n in range(2, 5)],     

    n_estimators = [n for n in range(10, 60, 10)],

)



xgb_CV = GridSearchCV(estimator=XGBClassifier(), param_grid=params, scoring='accuracy',cv=5)

xgb_CV.fit(X, y)



lgb_CV = GridSearchCV(estimator=LGBMClassifier(), param_grid=params, scoring='accuracy',cv=5)

lgb_CV.fit(X, y)



random_forest_CV = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, scoring='accuracy',cv=5)

random_forest_CV.fit(X, y)
from sklearn.ensemble import VotingClassifier



model = VotingClassifier(estimators=[('xgb', xgb_CV.best_estimator_),('lgb', lgb_CV.best_estimator_),('rf', random_forest_CV.best_estimator_)], voting='hard')

model.fit(X, y), 

model.score(X, y)
test = process_df(test)
survived = model.predict(test[columns])
submission = pd.DataFrame(columns={'PassengerId', 'Survived'})
submission['PassengerId'] = test['PassengerId']

submission['Survived'] = survived



submission.to_csv('../working/submission.csv', index=False)