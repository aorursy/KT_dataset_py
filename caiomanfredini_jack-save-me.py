# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Dataframe operations - Train

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.drop('Name', 1, inplace=True)  # Is really necessary?

train_data.drop('Cabin', 1, inplace=True) # Is really necessary?



train_data['Embarked'] = train_data['Embarked'].fillna('N')
# Dataframe to numpy - Train

previsores = train_data.iloc[:, 2:10].values

classe = train_data.iloc[:, 1].values
# Pre-Processing - Train

from sklearn.preprocessing import LabelEncoder

labelencoder_previsores = LabelEncoder()



previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])

previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])

previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)

imputer = imputer.fit(previsores[:, 0:8])

previsores[:, 0:8] = imputer.transform(previsores[:, 0:8])
# Dataframe operations - Test

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.drop('Name', 1, inplace=True)  # Is really necessary?

test_data.drop('Cabin', 1, inplace=True) # Is really necessary?



test_data['Embarked'] = test_data['Embarked'].fillna('N')
# Dataframe to numpy - Test

previsores_test = test_data.iloc[:, 1:9].values
# Pre Processing - Test

previsores_test[:, 1] = labelencoder_previsores.fit_transform(previsores_test[:, 1])

previsores_test[:, 5] = labelencoder_previsores.fit_transform(previsores_test[:, 5])

previsores_test[:, 7] = labelencoder_previsores.fit_transform(previsores_test[:, 7])



imputer = imputer.fit(previsores_test[:, 0:8])

previsores_test[:, 0:8] = imputer.transform(previsores_test[:, 0:8])
# Random Forest

from sklearn.tree import DecisionTreeClassifier



y = classe



X = previsores

X_test = previsores_test



model = DecisionTreeClassifier()

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")