import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import io
import matplotlib.pyplot as plt
def bin_ages(cell):
      if cell < 10:
        return cell
      return cell - (cell % 10)
    
def bin_fares(fare):
  if fare < 8.5:
    return 1
  elif fare < 11:
    return 2
  elif fare < 15:
    return 3
  elif fare < 22:
    return 4
  elif fare < 40:
    return 5
  elif fare < 80:
    return 6
  elif fare < 150:
    return 7
  else:
    return 8
    
def clean_data(file_path):
    df = pd.read_csv(file_path)
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['isAlone'] = np.where((df['SibSp'] == 0) & (df['Parch'] == 0) , 1, 0)
    df['Familysize'] = df['SibSp'] + df['Parch'] + 1
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    #df['Age'] = df['Age'].astype('int').apply(bin_ages)
    df.loc[~df['Cabin'].isnull(), 'Cabin'] = 1
    df.loc[df['Cabin'].isnull(), 'Cabin'] = 0
    df['Cabin'] = df['Cabin'].astype('int')
    df['Embarked'] = df['Embarked'].fillna('N')
    df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2, 'N': 3})
    df['Fare'] = df['Fare'].apply(bin_fares)
    df.drop(['Name','Ticket'], axis=1, inplace=True)
    return df
df = clean_data('../input/train.csv')
df.drop(['PassengerId'], axis=1, inplace=True)
df.head(10)
from sklearn.model_selection import train_test_split
validation_train, validation_test = train_test_split(df, test_size=0.2)
validation_train.head()
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'isAlone', 'Familysize']

def compute_base_models(input_df):
    df_clone = input_df.copy()
    df_clone.reset_index(inplace=True)
    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    for fold in kf.split(df_clone):
        train = df_clone.iloc[fold[0]]
        test =  df_clone.iloc[fold[1]]
        result = next(kf.split(df_clone), None)

        cls_xgboost = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, min_child_weight=0.9, objective='binary:logistic')
        cls_xgboost.fit(train.loc[:, columns], train['Survived'])
        df_clone.loc[fold[1],'M1_xgboost'] = cls_xgboost.predict(test.loc[:, columns])

        cls_ada = AdaBoostClassifier()
        cls_ada.fit(train.loc[:, columns], train['Survived'])
        df_clone.loc[fold[1],'M2_ada'] = cls_ada.predict(test.loc[:, columns])

        cls_gradient = GradientBoostingClassifier()
        cls_gradient.fit(train.loc[:, columns], train['Survived'])
        df_clone.loc[fold[1],'M3_gradient'] = cls_gradient.predict(test.loc[:, columns])
    return [df_clone, cls_xgboost, cls_ada, cls_gradient]
df, cls_xgboost, cls_ada, cls_gradient = compute_base_models(df)
df.head()
from sklearn.ensemble import RandomForestClassifier
cls_random = RandomForestClassifier()
cls_random.fit(df.loc[:, ['Fare', 'Cabin', 'Sex', 'Age', 'M1_xgboost', 'M2_ada', 'M3_gradient']], df['Survived'])
df['Prediction'] = cls_random.predict(df.loc[:, ['Fare', 'Cabin', 'Sex', 'Age', 'M1_xgboost', 'M2_ada', 'M3_gradient']])
test = clean_data('../input/test.csv')
test_ids = test['PassengerId']
columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'isAlone', 'Familysize']
test['M1_xgboost'] = cls_xgboost.predict(test.loc[:, columns])
test['M2_ada'] = cls_ada.predict(test.loc[:, columns])
test['M3_gradient'] = cls_gradient.predict(test.loc[:, columns])
test['Prediction'] = cls_random.predict(test.loc[:, ['Fare', 'Cabin', 'Sex', 'Age', 'M1_xgboost', 'M2_ada', 'M3_gradient']])
df.head(10)
my_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': test['Prediction']})
my_submission.to_csv('submission_stacking.csv', index=False)
validation_train, cls_xgboost, cls_ada, cls_gradient = compute_base_models(validation_train)
cls_random = RandomForestClassifier()
cls_random.fit(validation_train.loc[:, ['M1_xgboost', 'M2_ada', 'M3_gradient']], validation_train['Survived'])

validation_test['M1_xgboost'] = cls_xgboost.predict(validation_test.loc[:, columns])
validation_test['M2_ada'] = cls_ada.predict(validation_test.loc[:, columns])
validation_test['M3_gradient'] = cls_gradient.predict(validation_test.loc[:, columns])
validation_test['Prediction'] = cls_random.predict(validation_test.loc[:, ['M1_xgboost', 'M2_ada', 'M3_gradient']])

from sklearn.metrics import accuracy_score
accuracy_score(validation_test['Survived'], validation_test['Prediction'])
