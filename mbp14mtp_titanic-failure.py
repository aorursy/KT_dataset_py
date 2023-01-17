import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
def get_titles(df):

    df = df.copy()

    df['title'] = df.Name.str.extract(' ([A-z]+?)\.', expand=True)

    df['title'].replace(

        ['Lady', 'Countess','Capt', 'Col','Don',

         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],

        'rare',

        inplace=True)

    df['title'].replace('Mme', 'Mrs', inplace=True)

    df['title'].replace('Ms', 'Miss', inplace=True)

    df['title'].replace('Mlle', 'Miss', inplace=True)

    return df



def preprocess_df(df):

    df = df.copy()

    df['name_len'] = df['Name'].apply(len)

    df['has_cabin'] = df['Cabin'].apply(

    lambda x: 0 if isinstance(x, float) else 1)

    df['not_alone'] = df['SibSp'] | df['Parch']

    df.drop(['Ticket', 'Cabin', 'Name'], axis=1, inplace=True)

    df['Age'].fillna(np.median(df['Age'].dropna()), inplace=True)

    df['Fare'].fillna(np.median(df['Fare'].dropna()), inplace=True)

    df['Embarked'].fillna('S', inplace=True)

    df = pd.get_dummies(df)

    return df
training_data = preprocess_df(

    pd.read_csv('../input/train.csv', index_col='PassengerId'))

y = training_data.pop('Survived')

X = training_data.values

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    test_size=0.2,

                                                    random_state=10200)



params = dict(max_features=range(5, len(X_train.T), 2),

              max_depth=(None, 20, 10, 5))

rf = RandomForestClassifier(random_state=101, n_estimators=100)

gs_rf = GridSearchCV(estimator=rf, param_grid=params, cv=5).fit(X_train, y_train)

print(gs_rf.best_params_)

print(gs_rf.score(X_test, y_test))
rf_all = RandomForestClassifier(random_state=2101)

gs_rf_all = GridSearchCV(estimator=rf, param_grid=params, cv=5).fit(X, y)

print(gs_rf.best_params_)
test_data = preprocess_df(

    pd.read_csv('../input/test.csv', index_col='PassengerId'))

X_test_proper = test_data.values

test_data.head()
gs_rf_all.predict(X_test_proper)
submission = pd.DataFrame({

        "PassengerId": test_data.index,

        "Survived": gs_rf_all.predict(X_test_proper)

    })

submission.to_csv('submission.csv', index=False)

submission.head()
!head submission.csv