# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
X = train_data.iloc[:, [2,4,5,6,7,8,9,10,11]].values

y = train_data.iloc[:, 1].values



# Taking care of missing data

from sklearn.impute import SimpleImputer

imputer_string = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_string.fit(X[:, [1,5,7,8]])

X[:, [1,5,7,8]] = imputer_string.transform(X[:, [1,5,7,8]])



imputer_int = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_int.fit(X[:, [2,3,4,6]])

X[:, [2,3,4,6]] = imputer_int.transform(X[:, [2,3,4,6]])



# Encoding the Independent Variable

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,8])], remainder='passthrough')

X = np.array(ct.fit_transform(X))



ct = ColumnTransformer(transformers=[('encoder', OrdinalEncoder(), [9,11])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
#Training

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 100, metric = 'minkowski', p = 2)

classifier.fit(X, y)
X_test = test_data.iloc[:, [1,3,4,5,6,7,8,9,10]].values



# Taking care of missing data

from sklearn.impute import SimpleImputer

imputer_string = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer_string.fit(X_test[:, [1,5,7,8]])

X_test[:, [1,5,7,8]] = imputer_string.transform(X_test[:, [1,5,7,8]])



imputer_int = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_int.fit(X_test[:, [2,3,4,6]])

X_test[:, [2,3,4,6]] = imputer_int.transform(X_test[:, [2,3,4,6]])



# Encoding the Independent Variable

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,8])], remainder='passthrough')

X_test = np.array(ct.fit_transform(X_test))



ct = ColumnTransformer(transformers=[('encoder', OrdinalEncoder(), [9,11])], remainder='passthrough')

X_test = np.array(ct.fit_transform(X_test))
#Predicting

y_pred = classifier.predict(X_test)
PassengerID = pd.read_csv("/kaggle/input/titanic/gender_submission.csv").iloc[:, 0].values

output = pd.DataFrame({'PassengerId': PassengerID, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
print