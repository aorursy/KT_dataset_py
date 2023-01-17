%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
def generate_features(df):

    df["Pclass"] = df.Pclass.apply(str)

    

    df["Age_null"] = df.Age.apply(lambda x: 1 if pd.isnull(x) else 0)

    df["Fare_null"] = df.Fare.apply(lambda x: 1 if pd.isnull(x) else 0)

    df["Embarked_null"] = df.Embarked.apply(lambda x: 1 if pd.isnull(x) else 0)

    df["Cabin_null"] = df.Cabin.apply(lambda x: 1 if pd.isnull(x) else 0)

    df["Cabin_letter"] = df.Cabin.apply(lambda x: str(x)[0])

    

    df["Title_mr"] = df.Name.apply(lambda x: 'mr.' in str(x).lower().split())

    df["Title_master"] = df.Name.apply(lambda x: 'master.' in str(x).lower().split())

    df["Name_length"] = df.Name.apply(lambda x: len(str(x)))

    

    df["Ticket_letter"] = df.Ticket.apply(lambda x: str(x)[0]).apply(str)

    df["Ticket_letter"] = np.where(

        (df['Ticket_letter']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), 

         df['Ticket_letter'], np.where((df['Ticket_letter']).isin(

            ['W', '4', '7', '6', 'L', '5', '8']), 'Low', 'Other'))

    

    df["Ticket_length"] = df.Ticket.apply(lambda x: len(str(x)))

    

    df['Fam_size'] = df.SibSp + df.Parch

    df['Fam_type'] = np.where(

        df.Fam_size == 0, 'Single',

        np.where(df.Fam_size <= 3, 'Medium', 

        'Large')

    )

    

    df.Age = df.Age.fillna(df.Age.mean())

    df.Fare = df.Fare.fillna(df.Fare.mean())

    df.Embarked = df.Embarked.fillna('S')

    df.Sex = df.Sex.map({'female': 0, 'male': 1}).astype(int)

        

    df = df.drop(['Ticket', 'Name', 'Cabin', 'Fam_size', 'SibSp', 'Parch'], 1)        

        

    df = pd.get_dummies(df)

    

    return df
train = generate_features(pd.read_csv('../input/train.csv'))

test = generate_features(pd.read_csv('../input/test.csv'))

train.shape, test.shape
train_labels = train['Survived'].values



for column in train.columns:

    if column not in test.columns:

        print("Dropping %s from train" % column)

        train = train.drop(column, axis=1)

        

for column in test.columns:

    if column not in train.columns:

        print("Adding %s to train" % column)

        train[column.title()] = 0

        

train_features = train.drop(['PassengerId'], axis=1).values

test_features = test.drop(['PassengerId'], axis=1).values



train_features.shape, test_features.shape
forest = RandomForestClassifier(

    max_features='auto',

    oob_score=True,

    random_state=1,

    n_jobs=-1)



param_grid = {

    "criterion": ["gini", "entropy"],

    "min_samples_leaf": [6, 8, 10, 12],

    "min_samples_split": [5, 7, 10, 12, 15, 20],

    "n_estimators": [50, 100]

}



gs = GridSearchCV(

    estimator=forest,

    param_grid=param_grid,

    scoring='accuracy',

    cv=3,

    n_jobs=-1

)



gs = gs.fit(train_features, train_labels)



print(gs.best_score_)

print(gs.best_params_)
params = gs.best_params_



clf = RandomForestClassifier(**gs.best_params_)

clf.fit(train_features, train_labels)

output = clf.predict(test_features)
result = np.c_[test.PassengerId.astype(int), output.astype(int)]

df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

#df_result.to_csv('../output/randomforest.csv', index=False)