# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib as plt

pd.set_option("display.max_columns",30)

path = "../input/train.csv"

pathtest = "../input/test.csv"

pathpred ="prediction.csv"

dataset = pd.read_csv(path)





print(dataset.head())

essential_dataset = pd.DataFrame()

#essential_dataset["Pclass"] = dataset["Pclass"]

essential_dataset["Sex"] = dataset["Sex"]

essential_dataset["Age"] = dataset["Age"]

essential_dataset["SibSp"] = dataset["SibSp"]

essential_dataset["Parch"] = dataset["Parch"]

essential_dataset["Embarked"] = dataset["Embarked"]

essential_dataset["Fare"] = dataset["Fare"]



y = dataset.iloc[:, 1].values

X = essential_dataset.iloc[:, :].values

print(essential_dataset.head())

print(f"Independent Variable :\n{pd.DataFrame(X).head()}")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



from sklearn.compose import make_column_transformer

from sklearn.impute import SimpleImputer

le = LabelEncoder()



mt = make_column_transformer((OneHotEncoder(), [0]), remainder='passthrough')

X = mt.fit_transform(X)

X = X[:, 1:]

print(f'After Transformation \n{pd.DataFrame(X).head()}{X.shape}')

sc = SimpleImputer(strategy='most_frequent', missing_values=np.nan, fill_value='most_frequent')

print(f'After Fitting :{pd.DataFrame(sc.fit_transform(X))}')

X = sc.fit_transform(X)



mt = make_column_transformer((OneHotEncoder(), [4]), remainder='passthrough')

X = mt.fit_transform(X)

print(f'After Transformation \n{pd.DataFrame(X).head()}')

X = X[:, 1:]

print(f'After Transformation \n{pd.DataFrame(X).head()}{X.shape}')





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)





from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_pred = y_pred > 0.5

print(y_pred)



from seaborn import heatmap

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix as cm

res = cm(y_true=y_test, y_pred=y_pred)

# plt.show(heatmap(res))

print(((res[0][0]+res[1][1])/(res[0][0]+res[1][0]+res[0][1]+res[1][1]))*100)

print()



print('=========================================TEST.CSV=================================================')

dataset = pd.read_csv(pathtest)

print(dataset.head())

essential_dataset = pd.DataFrame()

#essential_dataset["Pclass"] = dataset["Pclass"]

essential_dataset["Sex"] = dataset["Sex"]

essential_dataset["Age"] = dataset["Age"]

essential_dataset["SibSp"] = dataset["SibSp"]

essential_dataset["Parch"] = dataset["Parch"]

essential_dataset["Embarked"] = dataset["Embarked"]

essential_dataset["Fare"] = dataset["Fare"]





X = essential_dataset.iloc[:, :].values

print(essential_dataset.head())

print(f"Independent Variable :\n{pd.DataFrame(X).head()}")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



from sklearn.compose import make_column_transformer

from sklearn.impute import SimpleImputer

le = LabelEncoder()



mt = make_column_transformer((OneHotEncoder(), [0]), remainder='passthrough')

X = mt.fit_transform(X)

X = X[:, 1:]

print(f'After Transformation \n{pd.DataFrame(X).head()}{X.shape}')

sc = SimpleImputer(strategy='most_frequent', missing_values=np.nan, fill_value='most_frequent')

print(f'After Fitting :{pd.DataFrame(sc.fit_transform(X))}')

X = sc.fit_transform(X)



mt = make_column_transformer((OneHotEncoder(), [4]), remainder='passthrough')

X = mt.fit_transform(X)

print(f'After Transformation \n{pd.DataFrame(X).head()}')

X = X[:, 1:]

print(f'After Transformation \n{pd.DataFrame(X).head()}{X.shape}')







from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X = sc.fit_transform(X)



y_pred_test = classifier.predict(X)



print(f'Answer Predicted\n{y_pred_test}')



dataset_written = pd.DataFrame()

dataset_written['PassengerId'] = dataset['PassengerId']

dataset_written['Survived'] = y_pred_test



dataset_written.to_csv(pathpred, index=False)
