# Load libraries

import pandas as pd

from catboost import CatBoostClassifier

from sklearn.model_selection import KFold
# Load in the train and test datasets

data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.info()
data_test.info()
# Mapping Sex

Sex = {"male": 0, "female": 1}

data_train["SexBinary"] = data_train["Sex"].apply(lambda e: Sex.get(e))

data_test["SexBinary"] = data_test["Sex"].apply(lambda e: Sex.get(e))



# Filling missing values in Age

data_train['Age'] = data_train['Age'].fillna(data_train['Age'].median())

data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())



# Filling missing values in Embarked

data_train['Embarked'] = data_train['Embarked'].fillna("S")



# Mapping Embarked

Embarked = {"C": 0, "Q": 1, "S": 2}

data_train["EmbarkedNumber"] = data_train["Embarked"].apply(lambda e: Embarked.get(e))

data_test["EmbarkedNumber"] = data_test["Embarked"].apply(lambda e: Embarked.get(e))



# Filling missing values in Fare

data_test['Fare'] = data_test['Fare'].fillna(0.0)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'SexBinary', 'EmbarkedNumber']
# Predictions



regr = CatBoostClassifier(

    loss_function="Logloss", 

    eval_metric='AUC', 

    use_best_model=True, 

    random_seed=1, 

    iterations = 1000,

    learning_rate = 0.01,

    verbose=100)



kf = KFold(n_splits=5, random_state=1, shuffle=True)



data_test['Survived'] = 0



idx = 0



for train_index, valid_index in kf.split(data_train):

    idx = idx + 1

    print( "\nFold:", idx)

    

    train = data_train.iloc[train_index]

    valid = data_train.iloc[valid_index]

   

    regr.fit(train[features], train['Survived'], eval_set=(valid[features], valid['Survived']))

    

    data_test['Survived'] += regr.predict_proba(data_test[features])[:,1]



    

data_test['Survived'] /= 5
def get_0_1(x):

    if x < 0.5: return 0

    else: return 1

    

data_test['Survived'] = data_test['Survived'].apply(lambda x: get_0_1(x))



data_test[['PassengerId', 'Survived']].to_csv("Titanic-step0.csv", index=False)



print('Done')