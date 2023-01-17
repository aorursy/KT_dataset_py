# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset_train = pd.read_csv('/kaggle/input/titanic/train.csv')

X_train = dataset_train.iloc[:, [2,4,5,9,11]].values

y_train = dataset_train.iloc[:, 1].values

X_train



# For Age

imputer_1 = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_1.fit(X_train[:, [2]])

X_train[:, [2]] = imputer_1.transform(X_train[:, [2]])



# For Embarked

imputer_2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_2.fit(X_train[:, [4]])

X_train[:, [4]] = imputer_2.transform(X_train[:, [4]])

X_train

# Encoding P Class

ct_1 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X_train = np.array(ct_1.fit_transform(X_train))

X_train = X_train[: ,1:]

print(X_train)



# Encoding Embarked

ct_2 = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')

X_train = np.array(ct_2.fit_transform(X_train))

X_train = X_train[: ,[0,1,3,4,5,6,7]]



# Encoding Gender

le_train = LabelEncoder()

X_train[:, 4] = le_train.fit_transform(X_train[:, 4])
dataset_test= pd.read_csv('/kaggle/input/titanic/test.csv')

X_test = dataset_test.iloc[:, [1,3,4,8,10]].values
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

X_test = X_test[: ,[0,1,3,4,5,6,7]]



# Encoding Gender

le_test = LabelEncoder()

X_test[:, 4] = le_test.fit_transform(X_test[:, 4])
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)

classifier=grid_search_cv.best_estimator_

classifier.fit(X_train,y_train)

y_pred_train = classifier.predict(X_train)



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 



cm = confusion_matrix(y_train, y_pred_train)



print('Confusion Matrix :')

print(confusion_matrix(y_train, y_pred_train)) 

print('Accuracy Score :',accuracy_score(y_train, y_pred_train))

print('Report : ')

print(classification_report(y_train, y_pred_train))
# from sklearn.tree import DecisionTreeClassifier

# classifier = DecisionTreeClassifier(criterion = 'entropy')

# classifier.fit(X_train, y_train)
# y_pred_train = classifier.predict(X_train)



# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import accuracy_score 

# from sklearn.metrics import classification_report 



# cm = confusion_matrix(y_train, y_pred_train)



# print('Confusion Matrix :')

# print(confusion_matrix(y_train, y_pred_train)) 

# print('Accuracy Score :',accuracy_score(y_train, y_pred_train))

# print('Report : ')

# print(classification_report(y_train, y_pred_train))
y_pred_test = classifier.predict(X_test)



output = pd.DataFrame({'PassengerId': dataset_test.PassengerId, 'Survived': y_pred_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")