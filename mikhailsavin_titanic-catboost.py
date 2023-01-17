import numpy as np
import pandas as pd 
import catboost

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# base preprocessing

train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test = pd.get_dummies(test, drop_first=True)
train = pd.get_dummies(train, drop_first=True)

train.columns = map(str.lower, train.columns)
test.columns = map(str.lower, test.columns)

mean_age = train[['age', 'sex_male']].groupby(['sex_male']).mean().reset_index()

def fill_age(data):
    if data['age'] > 0:
        return data['age']
    else:
        return mean_age[mean_age['sex_male'] == data['sex_male']]['age'].values[0]
    
    
train['age'] = train.apply(fill_age, axis=1).astype('int')
test['age'] = test.apply(fill_age, axis=1).astype('int')

full_train_t = train['survived']
full_train_f = train.drop(['survived'], axis=1)
    
train, valid = train_test_split(train, test_size=0.2, random_state=1)

train_t = train['survived']
train_f = train.drop(['survived'], axis=1)
valid_t = valid['survived']
valid_f = valid.drop(['survived'], axis=1)

test_f = test.drop('passengerid', axis=1)

# check most influential features

to_drop = ['age', 'sibsp', 'parch', 'fare', 'embarked_q', 'embarked_s']

train_f = train_f.drop(to_drop, axis=1)
valid_f = valid_f.drop(to_drop, axis=1)
test_f = test_f.drop(to_drop, axis=1)
# checking corr

train.corr()
train_f
# create model

cat_features = [0,1]

model = catboost.CatBoostClassifier(iterations=400,
                                    depth=4,
                                    l2_leaf_reg=2,
                                    random_seed=1,
                                    learning_rate=1,
                                    loss_function='Logloss',
                                    verbose=False)

# fit
model.fit(train_f, train_t, cat_features)
# check best score
predictions = model.predict(valid_f)
score = accuracy_score(valid_t, predictions)
score
# train best model on full train sample
model.fit(full_train_f, full_train_t, cat_features)
# test predictions

predictions = model.predict(test_f)
predictions

test['survival'] = pd.Series(predictions)
result = test[['passengerid', 'survival']]
result.columns =  ['PassengerId', 'Survived']
result = result.reset_index(drop=True)

result.to_csv('titanic_cb_3.csv',index=False)
