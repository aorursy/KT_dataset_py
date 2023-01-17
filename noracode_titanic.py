import pandas as pd

import xgboost as xgb

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_train.loc[df_train['Sex'] == 'male', 'Sex'] = 0

df_train.loc[df_train['Sex'] == 'female', 'Sex'] = 1

df_test.loc[df_test['Sex'] == 'male', 'Sex'] = 0

df_test.loc[df_test['Sex'] == 'female', 'Sex'] = 1
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

df_train[features].head()
model = xgb.XGBClassifier(max_depth=1, n_estimators=100, learning_rate=0.1).fit(df_train[features].as_matrix(), df_train['Survived'])
predictions = model.predict(df_test[features].as_matrix())
result = pd.DataFrame({'PassengerId': df_test['PassengerId'],

                       'Survived': predictions})