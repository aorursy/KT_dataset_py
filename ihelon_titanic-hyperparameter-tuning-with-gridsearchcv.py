import os

import random



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import lightgbm as lgbm

import catboost as cb

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
def set_seed(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed_value)

    



SEED = 42

set_seed(SEED)
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
print(f"Train shape: {train_df.shape}")

train_df.sample(3)
print(f"Test shape: {test_df.shape}")

test_df.sample(3)
used_columns = [

    "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"

]

full_df = pd.concat([train_df[used_columns], test_df[used_columns]])

y_train = train_df["Survived"].values
full_df.isna().sum()
full_df = full_df.drop(["Age", "Cabin"], axis=1)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.hist(full_df["Fare"], bins=20)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title("Fare distribution", fontsize=16)



plt.subplot(1, 2, 2)

embarked_info = full_df["Embarked"].value_counts()

plt.bar(embarked_info.index, embarked_info.values)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title("Embarked distribution", fontsize=16);
full_df["Embarked"].fillna("S", inplace=True)

full_df["Fare"].fillna(full_df["Fare"].mean(), inplace=True)
full_df['Title'] = full_df['Name'].str.extract(' ([A-Za-z]+)\.')

full_df['Title'] = full_df['Title'].replace(['Ms', 'Mlle'], 'Miss')

full_df['Title'] = full_df['Title'].replace(['Mme', 'Countess', 'Lady', 'Dona'], 'Mrs')

full_df['Title'] = full_df['Title'].replace(['Dr', 'Major', 'Col', 'Sir', 'Rev', 'Jonkheer', 'Capt', 'Don'], 'Mr')
full_df["Sex"] = full_df["Sex"].map({"male": 1, "female": 0}).astype(int)    

full_df["Embarked"] = full_df["Embarked"].map({"S": 1, "C": 2, "Q": 3}).astype(int)    

full_df['Title'] = full_df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3}).astype(int)   
full_df['TicketNumber'] = full_df['Ticket'].str.split()

full_df['TicketNumber'] = full_df['TicketNumber'].str[-1]

full_df['TicketNumber'] = LabelEncoder().fit_transform(full_df['TicketNumber'])
full_df = full_df.drop(['Name', 'Ticket'], axis=1)
full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1

full_df['IsAlone'] = full_df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
full_df.head()
categorical_columns = ['Pclass', 'Sex', 'Embarked', 'Title', 'TicketNumber', 'IsAlone']
X_train = full_df[:y_train.shape[0]]

X_test = full_df[y_train.shape[0]:]

X_train.shape, y_train.shape, X_test.shape
%%time

parameters = {

    "max_depth": [3, 5, 7, 9, 11, 13],

}



model_desicion_tree = DecisionTreeClassifier(

    random_state=SEED,

    class_weight='balanced',

)



model_desicion_tree = GridSearchCV(

    model_desicion_tree, 

    parameters, 

    cv=5,

    scoring='accuracy',

)



model_desicion_tree.fit(X_train, y_train)



print('-----')

print(f'Best parameters {model_desicion_tree.best_params_}')

print(

    f'Mean cross-validated accuracy score of the best_estimator: ' + \

    f'{model_desicion_tree.best_score_:.3f}'

)

print('-----')
%%time

parameters = {

    "n_estimators": [5, 10, 15, 20, 25], 

    "max_depth": [3, 5, 7, 9, 11, 13],

}



model_random_forest = RandomForestClassifier(

    random_state=SEED,

    class_weight='balanced',

)



model_random_forest = GridSearchCV(

    model_random_forest, 

    parameters, 

    cv=5,

    scoring='accuracy',

)



model_random_forest.fit(X_train, y_train)



print('-----')

print(f'Best parameters {model_random_forest.best_params_}')

print(

    f'Mean cross-validated accuracy score of the best_estimator: '+ \

    f'{model_random_forest.best_score_:.3f}'

)

print('-----')
%%time

parameters = {

    'max_depth': [3, 5, 7, 9], 

    'n_estimators': [5, 10, 15, 20, 25, 50, 100],

    'learning_rate': [0.01, 0.05, 0.1]

}



model_xgb = xgb.XGBClassifier(

    random_state=SEED,

)



model_xgb = GridSearchCV(

    model_xgb, 

    parameters, 

    cv=5,

    scoring='accuracy',

)



model_xgb.fit(X_train, y_train)



print('-----')

print(f'Best parameters {model_xgb.best_params_}')

print(

    f'Mean cross-validated accuracy score of the best_estimator: ' + 

    f'{model_xgb.best_score_:.3f}'

)

print('-----')
%%time

parameters = {

    'n_estimators': [5, 10, 15, 20, 25, 50, 100],

    'learning_rate': [0.01, 0.05, 0.1],

    'num_leaves': [7, 15, 31],

}



model_lgbm = lgbm.LGBMClassifier(

    random_state=SEED,

    class_weight='balanced',

)



model_lgbm = GridSearchCV(

    model_lgbm, 

    parameters, 

    cv=5,

    scoring='accuracy',

)



model_lgbm.fit(

    X_train, 

    y_train, 

    categorical_feature=categorical_columns

)



print('-----')

print(f'Best parameters {model_lgbm.best_params_}')

print(

    f'Mean cross-validated accuracy score of the best_estimator: ' + 

    f'{model_lgbm.best_score_:.3f}'

)

print('-----')
%%time

parameters = {

    'iterations': [5, 10, 15, 20, 25, 50, 100],

    'learning_rate': [0.01, 0.05, 0.1],

    'depth': [3, 5, 7, 9, 11, 13],

}



model_catboost = cb.CatBoostClassifier(

    verbose=False,

)



model_catboost = GridSearchCV(

    model_catboost, 

    parameters, 

    cv=5,

    scoring='accuracy',

)



model_catboost.fit(X_train, y_train)



print('-----')

print(f'Best parameters {model_catboost.best_params_}')

print(

    f'Mean cross-validated accuracy score of the best_estimator: ' + 

    f'{model_catboost.best_score_:.3f}'

)

print('-----')
def create_submission(model, X_test, test_passenger_id, model_name):

    y_pred_test = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame(

        {

            'PassengerId': test_passenger_id, 

            'Survived': (y_pred_test >= 0.5).astype(int),

        }

    )

    submission.to_csv(f"submission_{model_name}.csv", index=False)

    

    return y_pred_test
test_pred_decision_tree = create_submission(

    model_desicion_tree, X_test, test_df["PassengerId"], "decision_tree"

)

test_pred_random_forest = create_submission(

    model_random_forest, X_test, test_df["PassengerId"], "random_forest"

)

test_pred_xgboost = create_submission(

    model_xgb, X_test, test_df["PassengerId"], "xgboost"

)

test_pred_lightgbm = create_submission(

    model_lgbm, X_test, test_df["PassengerId"], "lightgbm"

)

test_pred_catboost = create_submission(

    model_catboost, X_test, test_df["PassengerId"], "catboost"

)
test_pred_merged = (

    test_pred_decision_tree + 

    test_pred_random_forest + 

    test_pred_xgboost + 

    test_pred_lightgbm + 

    test_pred_catboost

)

test_pred_merged = np.round(test_pred_merged / 5)
submission = pd.DataFrame(

    {

        'PassengerId': test_df["PassengerId"], 

        'Survived': test_pred_merged.astype(int),

    }

)

submission.to_csv(f"submission_merged.csv", index=False)