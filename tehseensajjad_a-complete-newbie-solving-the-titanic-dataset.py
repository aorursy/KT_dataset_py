import pandas as pd



train_set_raw = pd.read_csv('../input/titanic/train.csv')

test_set_raw = pd.read_csv('../input/titanic/test.csv')
train_set_raw.head(10)


train_set = train_set_raw.drop(['Cabin', 'Ticket'], axis=1)

test_set = test_set_raw.drop(['Cabin', 'Ticket'], axis=1)
print(train_set.Age.isna().sum())
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='median')

imputer.fit(train_set[['Age']])

train_set['Age'] = pd.DataFrame(imputer.transform(train_set[['Age']]), columns=['Age'])

test_set['Age'] = pd.DataFrame(imputer.transform(test_set[['Age']]), columns=['Age'])
train_set.head(10)
train_set = train_set.assign(Family_Count=(train_set.SibSp + train_set.Parch))

test_set = test_set.assign(Family_Count=(test_set.SibSp + test_set.Parch))
train_set['IsAlone'] = 0

train_set.loc[train_set.Family_Count == 0, 'IsAlone'] = 1



test_set['IsAlone'] = 0

test_set.loc[test_set.Family_Count == 0, 'IsAlone'] = 1



train_set.head(10)
train_set.info()
from sklearn.preprocessing import LabelEncoder

l_encoder = LabelEncoder()

train_set.Sex = pd.DataFrame(l_encoder.fit_transform(train_set[['Sex']]), columns=['Sex'])

test_set.Sex = pd.DataFrame(l_encoder.fit_transform(test_set[['Sex']]), columns=['Sex'])
train_set.head(10)
print(train_set.info())

print()

print(test_set.info())
train_set['Embarked'].isna().sum()
train_set.loc[

    (train_set['Embarked'].isna())

]
train_set.loc[

    (train_set.Fare == 80.0)

]
train_set.Embarked.dropna().mode()
train_set.Embarked = train_set.Embarked.fillna('S')

test_set.Embarked = test_set.Embarked.fillna('S')

train_set.head(10)
train_embarked = pd.get_dummies(train_set.Embarked)

test_embarked = pd.get_dummies(test_set.Embarked)
train_embarked
train_set = train_set.drop(['Embarked'], axis=1).join(train_embarked)

test_set = test_set.drop(['Embarked'], axis=1).join(test_embarked)
train_set
mean_imputer = SimpleImputer(strategy='mean')

mean_imputer.fit(test_set[['Fare']])



test_set['Fare'] = pd.DataFrame(mean_imputer.transform(test_set[['Fare']]), columns=['Fare'])
drop_cols = ['Name', 'SibSp', 'Parch']

train_set = train_set.drop(drop_cols, axis=1)

test_set  = test_set.drop(drop_cols, axis=1)

print(train_set.info())
train_set.head()
test_set.info()
test_set.head()
from sklearn.model_selection import train_test_split



X = train_set.drop(['Survived'], axis=1)

y = train_set['Survived']



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,

                                                random_state=42,

                                               test_size=0.2,

                                               train_size=0.8)

print(f"""Xtrain:\t{Xtrain.shape}

Xtest:\t{Xtest.shape}

ytrain:\t{ytrain.shape}

ytest:\t{ytest.shape}""")
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



def get_RF_score(est):

    model = RandomForestClassifier(n_estimators=est, random_state=42)

    model.fit(Xtrain, ytrain)

    predictions = model.predict(Xtest)

    return accuracy_score(ytest, predictions)
#checking how many estimators are good

RF_scores = dict()

for estimator in range(50, 1050, 50):

    RF_scores[estimator] = get_RF_score(estimator)
for est, score in RF_scores.items():

    if score == max(RF_scores.values()):

        print(f"""

        Best Score:

        {est}:\t{score}""")

        break
test_set.to_csv('test_set_cleaned.csv')

train_set.to_csv('train_set_cleaned.csv')
RF_Classifier = RandomForestClassifier(n_estimators=50, random_state=42)

RF_Classifier.fit(Xtrain, ytrain)

RF_preds = RF_Classifier.predict(Xtest)

RF_score = accuracy_score(ytest, RF_preds)

RF_score
from sklearn.tree import DecisionTreeClassifier

DC_Classifier = DecisionTreeClassifier(random_state=42)

DC_Classifier.fit(Xtrain, ytrain)
DC_preds = DC_Classifier.predict(Xtest)
DC_score = accuracy_score(ytest, DC_preds)

DC_score
from xgboost import XGBClassifier

XG_Classifier = XGBClassifier(n_estimators=100,

                              early_stopping_rounds=10,

                              random_state=42)

XG_Classifier.fit(Xtrain, ytrain)
XG_preds = XG_Classifier.predict(Xtest)
XG_score = accuracy_score(ytest, XG_preds)
XG_score
from lightgbm import LGBMClassifier

LGBM_Classifier = LGBMClassifier(n_estimators=50,

                                boosting_type='dart',

                                learning_rate=0.1,

                                random_state=42)

LGBM_Classifier.fit(Xtrain, ytrain)
LGBM_preds = LGBM_Classifier.predict(Xtest)

LGBM_score = accuracy_score(ytest, LGBM_preds)

LGBM_score
models = ['Random Forrest', "DecisionTree", 'XGBoost', 'LightGBMClassifier']

score = [RF_score, DC_score, XG_score, LGBM_score]

scores_dataset = pd.DataFrame(score, index=models, columns=['Score'])
scores_dataset.sort_values(by=['Score'], ascending=False)
RF_on_test_set = pd.Series( RF_Classifier.predict(test_set),

                          name='Survived',

                          dtype='int64')
submission = test_set.loc[:, ['PassengerId']].join(RF_on_test_set)
submission.to_csv('my_submission.csv', columns=submission.columns, index=None)