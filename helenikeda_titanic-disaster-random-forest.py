import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from numpy import savetxt
base_train = pd.read_csv("/kaggle/input/titanic/train.csv")

base_train = base_train[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]



del base_train['Name']

del base_train['Ticket']

del base_train['Cabin']



base_test = pd.read_csv("/kaggle/input/titanic/test.csv")

del base_test['Name']

del base_test['Ticket']

del base_test['Cabin']
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer = imputer.fit(base_train.iloc[:, 3:4])

base_train.iloc[:, 3:4] = imputer.transform(base_train.iloc[:, 3:4])

base_test.iloc[:, 3:4] = imputer.transform(base_test.iloc[:, 3:4])
labelencoder = LabelEncoder()

base_train['Embarked'] = labelencoder.fit_transform(base_train['Embarked'].astype(str))

base_test['Embarked'] = labelencoder.fit_transform(base_test['Embarked'].astype(str))

base_train.iloc[:, 2] = labelencoder.fit_transform(base_train.iloc[:, 2])

base_test.iloc[:, 2] = labelencoder.fit_transform(base_test.iloc[:, 2])
scaler = StandardScaler()

base_train.iloc[:, 1:8] = scaler.fit_transform(base_train.iloc[:, 1:8])

base_test.iloc[:, 1:8] = scaler.fit_transform(base_test.iloc[:, 1:8])
base_test[base_test==np.inf]=np.nan

base_test.fillna(base_test.mean(), inplace=True)
predictors_train = base_train.iloc[:, 0:8].values

class_train = base_train.iloc[:, 8].values
classificador = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)

classificador.fit(predictors_train, class_train)

predictions = classificador.predict(base_test)
output = pd.DataFrame({'PassengerId': base_test.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)