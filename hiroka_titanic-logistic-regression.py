import pandas as pd

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train
test
gender_submission
X_columns = ['Pclass','Sex','Age','Fare']

Y_column = 'Survived'



sns.pairplot(train[X_columns+[Y_column]], hue='Survived')
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})



X_train = train[X_columns]

Y_train = train[Y_column]



my_imputer = SimpleImputer() # See https://www.kaggle.com/dansbecker/handling-missing-values

X_train = my_imputer.fit_transform(X_train)



lr = LogisticRegression()

lr.fit(X_train, Y_train)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})



X_test = test[X_columns]



X_test = my_imputer.fit_transform(X_test)



Y_pred = lr.predict(X_test)
result = pd.concat([test['PassengerId'], pd.Series(Y_pred)], axis=1)

result.columns = ['PassengerId', 'Survived']

result.to_csv('result.csv', index=False)