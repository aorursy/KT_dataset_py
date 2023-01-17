import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder
dataset_train = pd.read_csv('/kaggle/input/titanic/train.csv')
dataset_train.head()
dataset_train['family_size'] = dataset_train['SibSp'] + dataset_train['Parch'] + 1 

dataset_train.head()
dataset_train[['family_size', 'Survived']].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset_train['alone'] = 0

dataset_train.loc[dataset_train['family_size'] == 1, 'alone'] = 1

dataset_train.head()
dataset_train[['alone', 'Survived']].groupby(['alone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
X_train = dataset_train.iloc[:, [2,4,5,9,11,13]].values

y_train = dataset_train.iloc[:, 1].values
print(dataset_train.isnull().sum())
# For Age

imputer_1 = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_1.fit(X_train[:, [2]])

X_train[:, [2]] = imputer_1.transform(X_train[:, [2]])



# For Embarked

imputer_2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_2.fit(X_train[:, [4]])

X_train[:, [4]] = imputer_2.transform(X_train[:, [4]])
# Encoding P Class

ct_1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X_train = np.array(ct_1.fit_transform(X_train))

X_train = X_train[: ,1:]



# Encoding Embarked

ct_2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')

X_train = np.array(ct_2.fit_transform(X_train))

X_train = X_train[: ,[0,1,3,4,5,6,7,8]]



# Encoding Gender

le_train = LabelEncoder()

X_train[:, 4] = le_train.fit_transform(X_train[:, 4])
dataset_test= pd.read_csv('/kaggle/input/titanic/test.csv')
dataset_test.head()
dataset_test['family_size'] = dataset_test['SibSp'] + dataset_test['Parch'] + 1 

dataset_test.head()
dataset_test['alone'] = 0

dataset_test.loc[dataset_train['family_size'] == 1, 'alone'] = 1

dataset_test.head()
X_test = dataset_test.iloc[:, [1,3,4,8,10,12]].values
print(dataset_test.isnull().sum())
# For Age

imputer_3 = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_3.fit(X_test[:, [2]])

X_test[:, [2]] = imputer_3.transform(X_test[:, [2]])



# For Fare

imputer_4 = SimpleImputer(missing_values=np.nan, strategy='median')

imputer_4.fit(X_test[:, [3]])

X_test[:, [3]] = imputer_4.transform(X_test[:, [3]])
# Encoding P Class

ct_3 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X_test = np.array(ct_3.fit_transform(X_test))

X_test = X_test[: ,1:]



# Encoding Embarked

ct_4 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')

X_test = np.array(ct_4.fit_transform(X_test))

X_test = X_test[: ,[0,1,3,4,5,6,7,8]]



# Encoding Gender

le_test = LabelEncoder()

X_test[:, 4] = le_test.fit_transform(X_test[:, 4])
from sklearn.model_selection import train_test_split

X_1, X_2, y_1, y_2 = train_test_split(X_train, y_train, test_size = 0.20)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_1[:, [5,6]] = sc_X.fit_transform(X_1[:, [5,6]])

X_2[:, [5,6]] = sc_X.transform(X_2[:, [5,6]])

X_test[:, [5,6]] = sc_X.transform(X_test[:, [5,6]])
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')

classifier.fit(X_1,y_1)
y_pred_train = classifier.predict(X_2)



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



print('Confusion Matrix :')

print(confusion_matrix(y_2, y_pred_train)) 

print('Accuracy Score :',accuracy_score(y_2, y_pred_train))

print('Report : ')

print(classification_report(y_2, y_pred_train))
y_pred_test = classifier.predict(X_test)



output = pd.DataFrame({'PassengerId': dataset_test.PassengerId, 'Survived': y_pred_test})

output.to_csv('my_submission_4.csv', index=False)

print("Your submission was successfully saved!")